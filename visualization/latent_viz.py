import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from cvae_exp import (
    ImprovedCVAE,
    OASISContrastiveDataset,
    transform,
    DEVICE,
    CLASSES,
    LATENT_DIM,
)

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
CVAE_MODEL_PATH = "best_cvae_clustered_class_only.pth"
BATCH_SIZE = 32
NUM_CLASSES = len(CLASSES)

# -----------------------------
# Load dataset
# -----------------------------
test_dataset = OASISContrastiveDataset(DATA_DIR, transform=transform, split='val')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Load CVAE
# -----------------------------
model = ImprovedCVAE(
    latent_dim=LATENT_DIM,
    img_size=224,
    img_channel=1,
    num_classes=NUM_CLASSES
).to(DEVICE)

ckpt = torch.load(CVAE_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"Loaded CVAE checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

# -----------------------------
# Collect latent vectors
# -----------------------------
lat_z_content = []
lat_z_class = []
labels = []

with torch.no_grad():
    for x, y, _ in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        # recon, mu, logv, z_content, z_class_raw
        _, _, _, zc, zcls = model(x, y)

        lat_z_content.append(zc.cpu())
        lat_z_class.append(zcls.cpu())
        labels.append(y.cpu())

lat_z_content = torch.cat(lat_z_content).numpy()
lat_z_class = torch.cat(lat_z_class).numpy()
labels = torch.cat(labels).numpy()

print("z_content shape:", lat_z_content.shape)
print("z_class_raw shape:", lat_z_class.shape)

# -----------------------------
# dimensionality reduction utils
# -----------------------------

def plot_2d(feats, labels, title, save_path):
    plt.figure(figsize=(7, 6))
    for cls_id, cls_name in CLASSES.items():
        idx = labels == cls_id
        plt.scatter(
            feats[idx, 0],
            feats[idx, 1],
            s=10, alpha=0.7,
            label=cls_name
        )
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")

# -----------------------------
# PCA
# -----------------------------
pca = PCA(n_components=2)
pca_zc = pca.fit_transform(lat_z_content)
pca_zcls = pca.fit_transform(lat_z_class)

plot_2d(pca_zc, labels, "PCA of z_content", "pca_z_content.png")
plot_2d(pca_zcls, labels, "PCA of z_class_raw", "pca_z_class.png")

# -----------------------------
# t-SNE
# -----------------------------
tsne = TSNE(n_components=2, learning_rate="auto", init="random")

tsne_zc = tsne.fit_transform(lat_z_content)
tsne_zcls = tsne.fit_transform(lat_z_class)

plot_2d(tsne_zc, labels, "t-SNE of z_content", "tsne_z_content.png")
plot_2d(tsne_zcls, labels, "t-SNE of z_class_raw", "tsne_z_class.png")

print("Done. Saved PCA & t-SNE plots.")
