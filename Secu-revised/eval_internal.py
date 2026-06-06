"""
eval_internal.py
----------------
Label-free clustering quality evaluation for the SeCu / DINOv2 pipeline.

Computes three *internal* validation indices that need NO ground-truth labels:
    * Silhouette score          (higher = better; range [-1, 1])
    * Davies-Bouldin index      (lower  = better; >= 0)
    * Calinski-Harabasz index   (higher = better)

It rebuilds the model exactly like inference.py, extracts the 128-d
L2-normalized embedding (model.get_feature) and the cluster assignment
(model.get_pred) for every crop, then scores the partition.

Each run appends one row to a shared CSV (--out, default internal_metrics.csv)
tagged by --tag, so the metrics for every ablation variant accumulate in one
table that maps directly onto the paper's Table 1 / Table 3.

Example (main model = DINOv2 + size-mml, the existing best_model):
    python eval_internal.py \
      --model-path model/best_model.pth.tar \
      --backbone dinov2 --secu-cst size-mml \
      --secu-num-ins 4305 --secu-alpha 517 --secu-k 8 9 10 \
      --secu-tx 0.07 --secu-lratio 0.7 \
      --data-name custom --data-path .\\data\\adaptive_output \
      --tag dinov2_size-mml
"""
import os
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

import secu.builder
from inference import ImageDataset          # reuse the exact same dataset / preprocessing
from config import clusters_amount

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def build_parser():
    p = argparse.ArgumentParser(description="Label-free internal clustering metrics")
    p.add_argument("--model-path", required=True, type=str)
    p.add_argument("--data-path", required=True, type=str)
    p.add_argument("--data-name", default="custom", type=str)
    p.add_argument("--backbone", default="dinov2", type=str,
                   help="resnet18 / vit / dinov2 (must match training)")
    p.add_argument("--secu-cst", default="size-mml", type=str,
                   help="size / size-mml / entropy (must match training)")
    p.add_argument("--secu-k", default=[8, 9, 10], type=int, nargs="+")
    p.add_argument("--secu-dim", default=128, type=int)
    p.add_argument("--secu-num-ins", default=4305, type=int)
    p.add_argument("--secu-tx", default=0.07, type=float)
    p.add_argument("--secu-tw", default=0.05, type=float)
    p.add_argument("--secu-dual-lr", default=0.1, type=float)
    p.add_argument("--secu-lratio", default=0.7, type=float)
    p.add_argument("--secu-alpha", default=517, type=float)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--tag", required=True, type=str,
                   help="row label for the output CSV, e.g. dinov2_size-mml")
    p.add_argument("--out", default="internal_metrics.csv", type=str)
    return p


def get_base_encoder(backbone, data_name):
    if backbone == "resnet18":
        if data_name == "stl10":
            from nets.resnet_stl import resnet18
        elif data_name == "custom":
            from nets.resnet_custom import resnet18
        else:
            from nets.resnet_cifar import resnet18
        return resnet18
    elif backbone == "vit":
        from nets.vit import ViT
        return ViT
    elif backbone == "dinov2":
        from nets.vit import DINOv2
        return DINOv2
    raise ValueError(f"Unsupported backbone: {backbone}")


def main():
    args = build_parser().parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- rebuild model exactly like inference.py ----
    base_encoder = get_base_encoder(args.backbone, args.data_name)
    model = secu.builder.SeCu(
        base_encoder=base_encoder,
        K=args.secu_k,
        tx=args.secu_tx,
        tw=args.secu_tw,
        dim=args.secu_dim,
        num_ins=args.secu_num_ins,
        alpha=args.secu_alpha,
        dual_lr=args.secu_dual_lr,
        lratio=args.secu_lratio,
        constraint=args.secu_cst,
    )
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model.eval()

    ckpt = torch.load(args.model_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.load_param()

    # ---- same transform as inference.py (bands == 3) ----
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = ImageDataset(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    feats, preds = [], []
    with torch.no_grad():
        for names, images, targets, file_paths in loader:
            images = images.to(device, non_blocking=True)
            feats.append(model.get_feature(images).cpu().numpy())
            preds.append(torch.argmax(model.get_pred(images), dim=1).cpu().numpy())

    X = np.concatenate(feats).astype(np.float64)   # (N, 128) L2-normalized
    y = np.concatenate(preds)                        # (N,) in [0, K0)

    n_clusters = len(np.unique(y))
    print(f"[data] N={len(y)}  clusters_found={n_clusters}  feat_dim={X.shape[1]}")
    if n_clusters < 2:
        raise SystemExit("Need >= 2 non-empty clusters to score; got 1.")

    # Silhouette: cosine is the natural metric for L2-normalized embeddings.
    sil_cos = silhouette_score(X, y, metric="cosine")
    sil_euc = silhouette_score(X, y, metric="euclidean")
    dbi = davies_bouldin_score(X, y)        # lower better
    chi = calinski_harabasz_score(X, y)     # higher better

    print("\n========== Internal metrics ==========")
    print(f"tag                  : {args.tag}")
    print(f"Silhouette (cosine)  : {sil_cos:.4f}   (higher better)")
    print(f"Silhouette (euclid)  : {sil_euc:.4f}")
    print(f"Davies-Bouldin       : {dbi:.4f}      (lower  better)")
    print(f"Calinski-Harabasz    : {chi:.2f}      (higher better)")
    print("======================================\n")

    header = ["tag", "backbone", "loss", "N", "clusters",
              "silhouette_cos", "silhouette_euc", "davies_bouldin", "calinski_harabasz"]
    row = [args.tag, args.backbone, args.secu_cst, len(y), n_clusters,
           f"{sil_cos:.4f}", f"{sil_euc:.4f}", f"{dbi:.4f}", f"{chi:.2f}"]
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"[out] appended row to {args.out}")


if __name__ == "__main__":
    main()
