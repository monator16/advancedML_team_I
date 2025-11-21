import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F 

# === 기존 코드에서 import해야 하는 것들 ===
from cvae_exp import (
    OASISContrastiveDataset,
    ImprovedCVAE,
    ImprovedClusteringLoss,
    transform,
    NUM_CLASSES,
    CLASSES,
    DEVICE,
    LATENT_DIM,
    IMAGE_CHANNEL,
    IMAGE_SIZE,
    CVAE_MODEL_PATH,
    MARGIN
)

# ----------------------------------------------------
# 1) Intra-class variance of z_class_raw
# ----------------------------------------------------
@torch.no_grad()
def compute_intra_class_variance(model, loader):
    print("\n[1] Computing intra-class variance of z_class ...")
    model.eval()
    zs = {c: [] for c in range(NUM_CLASSES)}

    for x, y, _ in tqdm(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        _, _, _, z_content, z_class_raw = model(x, y)

        z_class_norm = F.normalize(z_class_raw, dim=1)

        for zi, yi in zip(z_class_raw, y):
            zs[int(yi.item())].append(zi.cpu().numpy())

    variances = {}
    for c in zs:
        arr = np.stack(zs[c])
        variances[c] = np.mean(np.var(arr, axis=0))

    return variances


# ----------------------------------------------------
# 2) Center-to-center distance
# ----------------------------------------------------
def compute_center_distances(cluster_loss_fn):
    print("\n[2] Computing center distances ...")

    # 1) torch tensor로 꺼내기
    centers = cluster_loss_fn.centers.detach().cpu()

    # 2) ★ L2-norm normalize 적용
    centers = F.normalize(centers, dim=1)

    # 3) numpy 변환
    centers = centers.numpy()

    dists = {}
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[f"{CLASSES[i]} vs {CLASSES[j]}"] = d

    avg_dist = np.mean(list(dists.values()))
    return avg_dist, dists


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":

    # ===== Validation dataset =====
    val_dataset = OASISContrastiveDataset("data", transform=transform, split='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # ===== Load trained CVAE =====
    print("\nLoading trained CVAE ...")
    model = ImprovedCVAE(
        LATENT_DIM, IMAGE_SIZE, IMAGE_CHANNEL,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    ckpt = torch.load(CVAE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"✔ Loaded model (epoch {ckpt['epoch']} | val_loss {ckpt['val_loss']:.4f})")

    # ===== Load trained centers =====
    print("Loading trained class centers ...")
    cluster_loss_fn = ImprovedClusteringLoss(
        latent_dim=model.cls_dim,
        num_classes=NUM_CLASSES,
        margin=MARGIN
    ).to(DEVICE)
    cluster_loss_fn.load_state_dict(ckpt["cluster"])
    print("✔ Loaded centers")

    # ===== Compute metrics =====
    intra_var = compute_intra_class_variance(model, val_loader)
    avg_dist, dist_dict = compute_center_distances(cluster_loss_fn)

    # ===== Print results =====
    print("\n==================== Results ====================")
    print("Intra-class variance (mean):")
    for c, v in intra_var.items():
        print(f"  {CLASSES[c]}: {v:.4f}")

    print("\nPairwise center distances:")
    for pair, dist in dist_dict.items():
        print(f"  {pair}: {dist:.4f}")

    print(f"\nAverage center distance = {avg_dist:.4f} (margin={MARGIN})")
    print("=================================================\n")
