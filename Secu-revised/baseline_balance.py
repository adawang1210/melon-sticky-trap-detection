"""
baseline_balance.py — cluster-balance comparison: SeCu vs naive KMeans on
*raw pretrained* DINOv2 features (no SeCu training).

Rationale: internal indices (Silhouette/DBI/CHI) are not comparable across
different feature spaces, so a head-to-head on those would be confounded.
Cluster *balance* (size distribution) IS space-independent and directly
showcases the SeCu size constraint. We report, for each method:
  - per-cluster sizes
  - normalized entropy  H = -sum p_i log p_i / log K   (1.0 = perfectly balanced)
  - min/max size ratio  (1.0 = perfectly balanced)

Usage (from Secu-revised/):
    python baseline_balance.py --data-path ./data/adaptive_output
"""
import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from inference import ImageDataset

os.environ["OMP_NUM_THREADS"] = "1"

# SeCu released-model coarse-cluster sizes (DINOv2 + size-mml, K=8); verified
# against output/cluster8/cluster_X and Table 2 of the paper.
SECU_SIZES = [862, 524, 745, 329, 480, 649, 370, 346]


def balance_stats(sizes):
    sizes = np.asarray(sizes, dtype=np.float64)
    n = sizes.sum()
    p = sizes / n
    p = p[p > 0]
    H = -(p * np.log(p)).sum() / np.log(len(sizes))     # normalized entropy
    ratio = sizes.min() / sizes.max()
    cv = sizes.std() / sizes.mean()                      # coefficient of variation
    return H, ratio, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="./data/adaptive_output")
    ap.add_argument("--k", default=8, type=int)
    ap.add_argument("--batch-size", default=64, type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import timm
    print("[model] loading frozen pretrained DINOv2 (no SeCu)...")
    model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m",
                              pretrained=True, num_classes=0, img_size=224)
    model = model.to(device).eval()

    # DINOv2 / ImageNet normalization (NOT the CIFAR stats used for SeCu input)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(ImageDataset(args.data_path, transform=transform),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    feats = []
    with torch.no_grad():
        for _, images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            feats.append(model(images).cpu().numpy())   # (B, 768) pooled embedding
    X = normalize(np.concatenate(feats), norm="l2")
    print(f"[data] N={len(X)} raw-DINOv2 dim={X.shape[1]}")

    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(X)
    km_sizes = [int((km == c).sum()) for c in range(args.k)]

    secu_H, secu_r, secu_cv = balance_stats(SECU_SIZES)
    km_H, km_r, km_cv = balance_stats(km_sizes)

    print("\n================= CLUSTER BALANCE (K=8) =================")
    print(f"SeCu (DINOv2+size, ours) sizes : {sorted(SECU_SIZES, reverse=True)}")
    print(f"KMeans on raw DINOv2     sizes : {sorted(km_sizes, reverse=True)}")
    print("-" * 56)
    print(f"{'metric':<26}{'SeCu':>12}{'KMeans-raw':>14}")
    print(f"{'norm. entropy (1=bal.)':<26}{secu_H:>12.3f}{km_H:>14.3f}")
    print(f"{'min/max ratio (1=bal.)':<26}{secu_r:>12.3f}{km_r:>14.3f}")
    print(f"{'coeff. of variation (0=bal.)':<26}{secu_cv:>12.3f}{km_cv:>14.3f}")
    print("========================================================")


if __name__ == "__main__":
    main()
