# ==============================================================
# CVAE + FiLM Class-conditioning + Classifier-guided Loss
# ==============================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict
from glob import glob
from PIL import Image
import lpips
import matplotlib.pyplot as plt
import timm  # ✅ torchvision.resnet 대신 timm 사용

# --- 1. 설정 및 하이퍼파라미터 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'data'
CVAE_MODEL_PATH = "best_cvae_clustered_class_only.pth"

IMAGE_SIZE = 224
IMAGE_CHANNEL = 1
LATENT_DIM = 128
CLASSES = {0: 'Non Demented', 1: 'Very mild Dementia', 2: 'Mild Dementia'}
NUM_CLASSES = len(CLASSES)
CLASS_NAMES_MAP = {v: k for k, v in CLASSES.items()}
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# Loss weights
BETA_KLD_WEIGHT = 2.0
BCE_WEIGHT = 1.0
LAMBDA_LPIPS = 1.0
W_CENTER = 10.0
W_SEPARATION = 5.0
MARGIN = 2.0

# Classifier-guided loss
CLASSIFIER_LOSS_WEIGHT = 2.0

OUTPUT_FOLDER = "GEN_SAMPLES"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


# ---------------------------------------------------------------
# Pretrained classifier 로드 (timm 기반, 기존 학습 코드와 동일 구조)
# ---------------------------------------------------------------

class SimpleClassifier(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=NUM_CLASSES):
        super().__init__()
        # ✅ timm으로 resnet18 생성 (in_chans=1, num_classes=3)
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=1
        )

    def forward(self, x):
        return self.model(x)


def load_classifier(
    path="classification_results/best_classifier_resnet18_weights_42.pth",
    model_name="resnet18"
):
    cls = SimpleClassifier(model_name=model_name, num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    cls.load_state_dict(ckpt)
    cls.eval()
    for p in cls.parameters():
        p.requires_grad = False
    return cls


# ---------------------------------------------------------------
# Dataset with stratified SUBJECT split
# ---------------------------------------------------------------

class OASISContrastiveDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', val_ratio=0.2, seed=42):
        self.transform = transform
        self.root_dir = root_dir
        self.data = []

        subject_to_imgs = defaultdict(list)

        for class_folder in os.listdir(root_dir):
            if class_folder not in CLASS_NAMES_MAP:
                continue

            class_label = CLASS_NAMES_MAP[class_folder]

            for img_path in glob(os.path.join(root_dir, class_folder, "*.jpg")):
                name = os.path.basename(img_path).split('.')[0]
                parts = name.split('_')
                subj = parts[1] if len(parts) > 1 else name
                subject_to_imgs[subj].append((img_path, class_label))

        self.subject_to_imgs = subject_to_imgs

        subject_main_label = {}
        for subj, items in subject_to_imgs.items():
            labels = [lbl for _, lbl in items]
            subject_main_label[subj] = max(set(labels), key=labels.count)

        random.seed(seed)
        class_to_subjects = defaultdict(list)
        for subj, lbl in subject_main_label.items():
            class_to_subjects[lbl].append(subj)

        train_subjects, val_subjects = [], []
        for lbl, subj_list in class_to_subjects.items():
            random.shuffle(subj_list)
            n_val = int(len(subj_list) * val_ratio)
            val_subjects.extend(subj_list[:n_val])
            train_subjects.extend(subj_list[n_val:])

        self.train_subjects = sorted(train_subjects)
        self.val_subjects = sorted(val_subjects)

        chosen = train_subjects if split == 'train' else val_subjects

        for subj in chosen:
            for img_path, lbl in subject_to_imgs[subj]:
                self.data.append((subj, img_path, lbl))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subj, img_path, y = self.data[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, y, subj


# ---------------------------------------------------------------
# Improved Latent Clustering Loss
# ---------------------------------------------------------------

class ImprovedClusteringLoss(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, margin: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.centers = nn.Parameter(torch.randn(num_classes, latent_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        centers_batch = self.centers[labels]
        center_loss = F.mse_loss(features, centers_batch)

        separation_loss = 0
        num_pairs = 0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                dist = F.pairwise_distance(
                    self.centers[i].unsqueeze(0),
                    self.centers[j].unsqueeze(0)
                )
                separation_loss += F.relu(self.margin - dist)
                num_pairs += 1

        if num_pairs > 0:
            separation_loss = separation_loss / num_pairs

        return center_loss, separation_loss


# ---------------------------------------------------------------
# FiLM (Scale/Shift Conditioning)
# ---------------------------------------------------------------

class FiLMCond(nn.Module):
    def __init__(self, embed_dim, num_features):
        super().__init__()
        self.scale = nn.Linear(embed_dim, num_features)
        self.shift = nn.Linear(embed_dim, num_features)

    def forward(self, h, class_emb):
        """
        h: (B, C, H, W)
        class_emb: (B, embed_dim)
        """
        gamma = self.scale(class_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.shift(class_emb).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + gamma) + beta


# ---------------------------------------------------------------
# CVAE + FiLM
# ---------------------------------------------------------------

def up_block(cin, cout):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(cin, cout, 3, 1, 1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ImprovedCVAE(nn.Module):
    def __init__(self, latent_dim, img_size, img_channel, num_classes):
        super().__init__()

        self.latent_dim = latent_dim
        self.cls_dim = 32                      # class latent
        self.content_dim = latent_dim - self.cls_dim

        # Encoder
        self.enc_conv1 = nn.Sequential(nn.Conv2d(img_channel, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc_conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc_conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc_conv4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())

        self.flat = 256 * 14 * 14
        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_log = nn.Linear(self.flat, latent_dim)

        # class embedding for decoder FiLM
        self.class_embed = nn.Embedding(num_classes, self.cls_dim)

        # FiLM modules
        self.film1 = FiLMCond(self.cls_dim, 128)
        self.film2 = FiLMCond(self.cls_dim, 64)
        self.film3 = FiLMCond(self.cls_dim, 32)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, self.flat)
        self.dec1 = up_block(256 + 128, 128)
        self.dec2 = up_block(128 + 64, 64)
        self.dec3 = up_block(64, 32)
        self.out = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid())

    def reparam(self, mu, logv):
        std = (0.5 * logv).exp()
        return mu + torch.randn_like(std) * std

    def forward(self, x, class_label):
        # ----- Encoder -----
        e1 = self.enc_conv1(x)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)

        h = e4.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logv = self.fc_log(h)
        z = self.reparam(mu, logv)

        # ----- Latent split -----
        z_content = z[:, :self.content_dim]
        z_class_raw = z[:, self.content_dim:]
        class_emb = self.class_embed(class_label)

        # Replace predicted class latent with target class embedding
        z_final = torch.cat([z_content, class_emb], dim=1)

        # ----- Decoder -----
        d = self.dec_fc(z_final).view(x.size(0), 256, 14, 14)

        d = F.interpolate(d, size=e3.shape[2:], mode="bilinear")
        d = self.dec1(torch.cat([d, e3], dim=1))
        d = self.film1(d, class_emb)

        d = F.interpolate(d, size=e2.shape[2:], mode="bilinear")
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = self.film2(d, class_emb)

        d = F.interpolate(d, size=(112, 112), mode="bilinear")
        d = self.dec3(d)
        d = self.film3(d, class_emb)

        out = self.out(F.interpolate(d, size=(224, 224), mode="bilinear"))

        return out, mu, logv, z_content, z_class_raw


# ---------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------

lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
for p in lpips_fn.parameters():
    p.requires_grad = False


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, lambda_recon, beta, clustering_loss_fn, classifier):
    model.eval()
    total = 0
    sum_loss = 0

    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        recon, mu, logv, z_content, z_class_raw = model(x, y)

        bce = F.binary_cross_entropy(recon, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
        lp = lpips_fn(recon.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1)).mean()
        center_loss, sep_loss = clustering_loss_fn(z_class_raw, y)

        # classifier-guided loss (정규화 맞춰주기)
        recon_norm = (recon - 0.456) / 0.224
        cls_logits = classifier(recon_norm)
        cls_loss = F.cross_entropy(cls_logits, y)

        loss = (lambda_recon * bce +
                beta * kld +
                LAMBDA_LPIPS * lp +
                W_CENTER * center_loss +
                W_SEPARATION * sep_loss +
                CLASSIFIER_LOSS_WEIGHT * cls_loss)

        bs = x.size(0)
        total += bs
        sum_loss += loss.item() * bs

    return sum_loss / total


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------

def train_improved_cvae(model, train_loader, val_loader,
                        epochs, lr, lambda_recon, beta_max, classifier):

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    clustering_loss_fn = ImprovedClusteringLoss(
        latent_dim=model.cls_dim,
        num_classes=NUM_CLASSES,
        margin=MARGIN
    ).to(DEVICE)

    best_val = 1e9
    patience = 10
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        beta = min(beta_max, beta_max * epoch / max(1, epochs // 2))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y, _ in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            recon, mu, logv, z_content, z_class_raw = model(x, y)

            bce = F.binary_cross_entropy(recon, x, reduction='mean')
            kld = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
            lp = lpips_fn(recon.repeat(1, 3, 1, 1), x.repeat(1, 3, 1, 1)).mean()
            center_loss, sep_loss = clustering_loss_fn(z_class_raw, y)

            # classifier-guided loss (정규화 맞춰주기)
            recon_norm = (recon - 0.456) / 0.224
            cls_logits = classifier(recon_norm)
            cls_loss = F.cross_entropy(cls_logits, y)

            loss = (lambda_recon * bce +
                    beta * kld +
                    LAMBDA_LPIPS * lp +
                    W_CENTER * center_loss +
                    W_SEPARATION * sep_loss +
                    CLASSIFIER_LOSS_WEIGHT * cls_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "cls": f"{cls_loss.item():.3f}"
            })

        scheduler.step()

        val_loss = validate(
            model, val_loader, lambda_recon, beta,
            clustering_loss_fn, classifier
        )

        print(f"\n====> Epoch {epoch}/{epochs} | Val Loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "cluster": clustering_loss_fn.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, CVAE_MODEL_PATH)
            print("🔥 Saved best model\n")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    return clustering_loss_fn


# ---------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------

@torch.no_grad()
def generate_conditional_image(model, image, orig_label, target_label):
    model.eval()

    img = image.to(DEVICE).unsqueeze(0)
    target = torch.tensor([target_label], device=DEVICE)

    recon, _, _, _, _ = model(img, target)

    arr = (recon.squeeze().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr, 'L')
    fname = f"from_{CLASSES[orig_label]}_to_{CLASSES[target_label]}_{np.random.randint(9999)}.png"
    path = os.path.join(OUTPUT_FOLDER, fname)
    pil.save(path)
    return path


@torch.no_grad()
def save_comparison(model, image, orig_label, target_label, save_path):
    model.eval()

    img = image.to(DEVICE).unsqueeze(0)
    target = torch.tensor([target_label], device=DEVICE)

    recon, _, _, _, _ = model(img, target)

    x0_np = image.squeeze().cpu().numpy()
    gen_np = recon.squeeze().cpu().clamp(0, 1).numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(x0_np, cmap='gray')
    plt.title(f"Original ({CLASSES[orig_label]})", fontsize=12)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gen_np, cmap='gray')
    plt.title(f"Generated ({CLASSES[target_label]})", fontsize=12)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":

    train_dataset = OASISContrastiveDataset(DATA_DIR, transform=transform, split='train')
    val_dataset = OASISContrastiveDataset(DATA_DIR, transform=transform, split='val')

    print("\n--- Train subjects:", len(train_dataset.train_subjects))
    print("--- Val subjects:", len(val_dataset.val_subjects))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ImprovedCVAE(
        LATENT_DIM, IMAGE_SIZE, IMAGE_CHANNEL,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # 🔥 timm 기반 pretrained classifier 로드
    classifier = load_classifier()

    Training (필요시)
    clustering_loss_fn = train_improved_cvae(
        model, train_loader, val_loader,
        epochs=NUM_EPOCHS, lr=LEARNING_RATE,
        lambda_recon=BCE_WEIGHT, beta_max=BETA_KLD_WEIGHT,
        classifier=classifier
    )

    # Load best model
    if os.path.exists(CVAE_MODEL_PATH):
        ckpt = torch.load(CVAE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"Best model loaded (epoch {ckpt['epoch']} | val {ckpt['val_loss']:.4f})")
    else:
        print("⚠ 저장된 모델 없음")

    # Example generation
    print("\nGenerating sample image...")
    x0, orig_label, _ = val_dataset[0]
    target = (orig_label + 1) % NUM_CLASSES
    gen_path = generate_conditional_image(model, x0, orig_label, target)
    print("Generated:", gen_path)

    print("\nGenerating comparison...")
    save_comparison(model, x0, orig_label, target, "comparison_sample.png")
    print("Saved comparison figure.")
