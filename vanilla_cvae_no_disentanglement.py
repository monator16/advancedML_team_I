# ==============================================================
# BASELINE (ABLATION): CVAE + FiLM
# Trained WITHOUT Disentanglement Loss
# Evaluated WITH Disentanglement Metrics (MIG, Silhouette, Modularity)
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
import timm
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Metrics Imports (RESTORED)
from sklearn.metrics import silhouette_score, mutual_info_score
from scipy.stats import entropy

# --- 1. Configuration & Hyperparameters ---

GPU_ID = 3
DEVICE = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

DATA_DIR = '/home/juhyeong/AML/Data'
CVAE_MODEL_PATH = "best_cvae_baseline_ablation_hyperparameter.pth" # Distinct name for the baseline model

IMAGE_SIZE = 224
IMAGE_CHANNEL = 1
LATENT_DIM = 128
CLASSES = {0: 'Non Demented', 1: 'Very mild Dementia', 2: 'Mild Dementia'}
NUM_CLASSES = len(CLASSES)
CLASS_NAMES_MAP = {v: k for k, v in CLASSES.items()}
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
BATCH_SIZE = 32

# Loss weights
BETA_KLD_WEIGHT = 2.0
BCE_WEIGHT = 1.0
LAMBDA_LPIPS = 2.0

# --- ABLATION: Disentanglement weights set to 0 or removed ---
# W_CENTER = 0.0
# W_SEPARATION = 0.0
# MARGIN = 0.0

# Classifier-guided loss (Kept as it is usually part of the semi-supervised baseline)
CLASSIFIER_LOSS_WEIGHT = 3.0

OUTPUT_FOLDER = "GEN_SAMPLES_BASELINE_HYPERPARAMETER"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


# ---------------------------------------------------------------
# Pretrained Classifier
# ---------------------------------------------------------------

class SimpleClassifier(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=NUM_CLASSES):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=1
        )

    def forward(self, x):
        return self.model(x)


def load_classifier(path="/home/juhyeong/AML/best_classifier_resnet18_weights_42.pth"):
    cls = SimpleClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    cls.load_state_dict(ckpt)
    cls.eval()
    for p in cls.parameters():
        p.requires_grad = False
    return cls


# ---------------------------------------------------------------
# Dataset
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

        # Stratified split by SUBJECT
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
# CVAE Model Components (Architecture remains same)
# ---------------------------------------------------------------

class FiLMCond(nn.Module):
    def __init__(self, embed_dim, num_features):
        super().__init__()
        self.scale = nn.Linear(embed_dim, num_features)
        self.shift = nn.Linear(embed_dim, num_features)

    def forward(self, h, class_emb):
        gamma = self.scale(class_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.shift(class_emb).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + gamma) + beta

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
        self.cls_dim = 32
        self.content_dim = latent_dim - self.cls_dim

        self.enc_conv1 = nn.Sequential(nn.Conv2d(img_channel, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc_conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc_conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc_conv4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())

        self.flat = 256 * 14 * 14
        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_log = nn.Linear(self.flat, latent_dim)

        self.class_embed = nn.Embedding(num_classes, self.cls_dim)

        self.film1 = FiLMCond(self.cls_dim, 128)
        self.film2 = FiLMCond(self.cls_dim, 64)
        self.film3 = FiLMCond(self.cls_dim, 32)

        self.dec_fc = nn.Linear(latent_dim, self.flat)
        self.dec1 = up_block(256 + 128, 128)
        self.dec2 = up_block(128 + 64, 64)
        self.dec3 = up_block(64, 32)
        self.out = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid())

    def reparam(self, mu, logv):
        std = (0.5 * logv).exp()
        return mu + torch.randn_like(std) * std

    def forward(self, x, class_label):
        e1 = self.enc_conv1(x)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)

        h = e4.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logv = self.fc_log(h)
        z = self.reparam(mu, logv)

        z_content = z[:, :self.content_dim]
        z_class_raw = z[:, self.content_dim:]
        class_emb = self.class_embed(class_label)

        z_final = torch.cat([z_content, class_emb], dim=1)

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
# Helper: LPIPS
# ---------------------------------------------------------------
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
for p in lpips_fn.parameters():
    p.requires_grad = False


# ---------------------------------------------------------------
# Helper: Validation Loop (ABLATION VERSION)
# ---------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, lambda_recon, beta, classifier):
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
        
        # --- ABLATION: No Clustering Loss Calculation ---
        # center_loss, sep_loss = clustering_loss_fn(z_class_raw, y)

        recon_norm = (recon - 0.456) / 0.224
        cls_logits = classifier(recon_norm)
        cls_loss = F.cross_entropy(cls_logits, y)

        loss = (lambda_recon * bce +
                beta * kld +
                LAMBDA_LPIPS * lp +
                # W_CENTER * center_loss +       <-- REMOVED
                # W_SEPARATION * sep_loss +      <-- REMOVED
                CLASSIFIER_LOSS_WEIGHT * cls_loss)

        bs = x.size(0)
        total += bs
        sum_loss += loss.item() * bs

    return sum_loss / total


# ---------------------------------------------------------------
# Training Loop (ABLATION VERSION)
# ---------------------------------------------------------------
def train_improved_cvae(model, train_loader, val_loader, epochs, lr, lambda_recon, beta_max, classifier):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # --- ABLATION: No Clustering Loss Initialization ---
    
    best_val = 1e10
    patience = 20
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
            
            # --- ABLATION: No Clustering Loss Calculation ---
            # center_loss, sep_loss = clustering_loss_fn(z_class_raw, y)

            recon_norm = (recon - 0.456) / 0.224
            cls_logits = classifier(recon_norm)
            cls_loss = F.cross_entropy(cls_logits, y)

            loss = (lambda_recon * bce +
                    beta * kld +
                    LAMBDA_LPIPS * lp +
                    # W_CENTER * center_loss +      <-- REMOVED
                    # W_SEPARATION * sep_loss +     <-- REMOVED
                    CLASSIFIER_LOSS_WEIGHT * cls_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "cls": f"{cls_loss.item():.3f}"})

        scheduler.step()
        val_loss = validate(model, val_loader, lambda_recon, beta, classifier)
        print(f"\n====> Epoch {epoch}/{epochs} | Val Loss {val_loss:.4f}") 

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
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


# ---------------------------------------------------------------
# Advanced Disentanglement Evaluation (RESTORED)
# ---------------------------------------------------------------

def discretize_latents(data, bins=20):
    discretized = np.zeros_like(data, dtype=int)
    for i in range(data.shape[1]):
        discretized[:, i] = pd.qcut(data[:, i], q=bins, labels=False, duplicates='drop')
    return discretized

def calculate_advanced_metrics(model, loader, device):
    model.eval()
    
    z_full_list, z_class_list, labels_list = [], [], []
    
    print("\n[Advanced Eval] Extracting latents...")
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Encoding"):
            x = x.to(device)
            y = y.to(device)
            
            _, _, _, z_content, z_class_raw = model(x, y)
            
            z_full = torch.cat([z_content, z_class_raw], dim=1)
            z_full_list.append(z_full.cpu().numpy())
            z_class_list.append(z_class_raw.cpu().numpy())
            labels_list.append(y.cpu().numpy())
            
    Z_full = np.concatenate(z_full_list, axis=0)
    Z_class = np.concatenate(z_class_list, axis=0)
    Y = np.concatenate(labels_list, axis=0)
    
    # 1. Silhouette
    print("  - Calculating Silhouette Score...")
    if len(Y) > 10000:
        indices = np.random.choice(len(Y), 10000, replace=False)
        sil_score = silhouette_score(Z_class[indices], Y[indices])
    else:
        sil_score = silhouette_score(Z_class, Y)

    # 2. MIG
    print("  - Calculating MIG...")
    discretized = np.zeros_like(Z_full, dtype=int)
    for i in range(Z_full.shape[1]):
        discretized[:, i] = pd.qcut(Z_full[:, i], q=20, labels=False, duplicates='drop')
    num_latents = Z_full.shape[1]
    _, counts = np.unique(Y, return_counts=True)
    entropy_y = entropy(counts)
    mi_scores = []
    for j in range(num_latents):
        mi = mutual_info_score(discretized[:, j], Y)
        mi_scores.append(mi)
    mi_scores = np.array(mi_scores)
    sorted_mi = np.sort(mi_scores)[::-1]
    mig_score = (sorted_mi[0] - sorted_mi[1]) / entropy_y if entropy_y > 0 else 0

    # 3. Modularity
    print("  - Calculating Modularity...")
    content_dim = Z_full.shape[1] - Z_class.shape[1] 
    mi_class = mi_scores[content_dim:]
    mi_content = mi_scores[:content_dim]
    
    total_info_class = np.sum(mi_class)
    total_info_content = np.sum(mi_content)
    
    modularity_ratio = total_info_class / (total_info_class + total_info_content + 1e-8)

    # 4. Linear Modularity + PLOTTING
    print("  - Generating Plot...")
    clf = LogisticRegression(max_iter=2000, solver='liblinear')
    clf.fit(Z_class, Y)
    avg_coefs = np.mean(np.abs(clf.coef_), axis=0) 
    lin_mod_score = np.max(avg_coefs) / (np.sum(avg_coefs) + 1e-8)
    
    top_2_idx = np.argsort(avg_coefs)[::-1][:2]
    dim_x, dim_y = top_2_idx[0], top_2_idx[1]
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z_class[:, dim_x], Z_class[:, dim_y], c=Y, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label="Class Label")
    plt.title(f"Baseline Latent Space (No Clustering)\n"
              f"Silhouette: {sil_score:.3f} | Linear Modularity: {lin_mod_score:.3f}")
    plt.xlabel(f"Latent Dim {dim_x}")
    plt.ylabel(f"Latent Dim {dim_y}")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(OUTPUT_FOLDER, "baseline_latent_scatter.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  - Saved plot to {save_path}")

    # Report
    print("\n" + "="*50)
    print(" BASELINE DISENTANGLEMENT METRICS (ABLATION)")
    print("="*50)
    print(f"1. Silhouette Score:       {sil_score:.4f}")
    print(f"2. Mutual Info Gap (MIG):  {mig_score:.4f}")
    print(f"3. Modularity (Info Ratio): {modularity_ratio:.4f}")
    print(f"   - Info in Content (Leakage): {total_info_content:.4f}")
    print(f"   - Info in Class Latents:     {total_info_class:.4f}")
    print("="*50 + "\n")
    
    return {
        "silhouette": sil_score,
        "mig": mig_score,
        "modularity": modularity_ratio
    }


# ---------------------------------------------------------------
# Image Generation Helpers
# ---------------------------------------------------------------
@torch.no_grad()
def generate_conditional_image(model, image, orig_label, target_label):
    model.eval()
    img = image.to(DEVICE).unsqueeze(0)
    target = torch.tensor([target_label], device=DEVICE)
    recon, _, _, _, _ = model(img, target)
    arr = (recon.squeeze().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr, 'L')
    fname = f"baseline_from_{CLASSES[orig_label]}_to_{CLASSES[target_label]}_{np.random.randint(9999)}.png"
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
    plt.title(f"Baseline Generated ({CLASSES[target_label]})", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    
    # --- CONTROL FLAGS -----------------------------------------
    TRAIN_MODE = True  
    # -----------------------------------------------------------

    # 1. Load Data
    train_dataset = OASISContrastiveDataset(DATA_DIR, transform=transform, split='train')
    val_dataset = OASISContrastiveDataset(DATA_DIR, transform=transform, split='val')

    print("\n--- Train subjects:", len(train_dataset.train_subjects))
    print("--- Val subjects:", len(val_dataset.val_subjects))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Initialize Model
    model = ImprovedCVAE(
        LATENT_DIM, IMAGE_SIZE, IMAGE_CHANNEL,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # 3. Load Classifier
    classifier = load_classifier()

    # 4. Logic Split: Train vs Load
    if TRAIN_MODE:
        print(f"\n🚀 Starting Baseline Training (Epochs: {NUM_EPOCHS})...")
        # Note: calling the modified train function that DOES NOT compute clustering loss
        train_improved_cvae(
            model, train_loader, val_loader,
            epochs=NUM_EPOCHS, lr=LEARNING_RATE,
            lambda_recon=BCE_WEIGHT, beta_max=BETA_KLD_WEIGHT,
            classifier=classifier
        )
    else:
        print(f"\n⏩ Skipping Training. Loading pre-trained model from {CVAE_MODEL_PATH}...")
        
        if os.path.exists(CVAE_MODEL_PATH):
            ckpt = torch.load(CVAE_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            print(f"✅ Successfully loaded model from epoch {ckpt['epoch']} (Val Loss: {ckpt['val_loss']:.4f})")
        else:
            raise FileNotFoundError(f"Could not find model at {CVAE_MODEL_PATH}. Set TRAIN_MODE = True first.")

    # 5. Run Evaluation Metrics on the Baseline Model
    print("\n" + "#"*60)
    print(" STARTING BASELINE (ABLATION) EVALUATION")
    print("#"*60)
    
    # This will now print the scores for the non-disentangled model
    metrics = calculate_advanced_metrics(model, val_loader, DEVICE)

    # 6. Generate Samples
    print("\nGenerating sample image...")
    if len(val_dataset) > 0:
        x0, orig_label, _ = val_dataset[0]
        target = (orig_label + 1) % NUM_CLASSES
        
        # Generate
        gen_path = generate_conditional_image(model, x0, orig_label, target)
        print("Generated:", gen_path)

        # Save Comparison
        print("Generating comparison...")
        save_path = os.path.join(OUTPUT_FOLDER, "baseline_comparison.png")
        save_comparison(model, x0, orig_label, target, save_path)
        print(f"Saved comparison figure to {save_path}")
    else:
        print("Validation dataset is empty, skipping generation.")
