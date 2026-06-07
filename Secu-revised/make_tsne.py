"""
make_tsne.py — render a 2-D t-SNE of the released model's embedding,
colored by the 8 SeCu clusters, and save it as a print-ready PNG for Fig. 7.

Usage (from Secu-revised/):
    python make_tsne.py --model-path model/best_dinov2-mml.pth.tar \
        --out ../投稿格式/tsne.png
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import secu.builder
from inference import ImageDataset
from eval_internal import get_base_encoder

os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="model/best_dinov2-mml.pth.tar")
    p.add_argument("--data-path", default="./data/adaptive_output")
    p.add_argument("--backbone", default="dinov2")
    p.add_argument("--secu-cst", default="size-mml")
    p.add_argument("--secu-k", default=[8, 9, 10], type=int, nargs="+")
    p.add_argument("--secu-num-ins", default=4305, type=int)
    p.add_argument("--secu-alpha", default=517, type=float)
    p.add_argument("--secu-tx", default=0.07, type=float)
    p.add_argument("--secu-lratio", default=0.7, type=float)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--out", default="../投稿格式/tsne.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_encoder = get_base_encoder(args.backbone, "custom")
    model = secu.builder.SeCu(
        base_encoder=base_encoder, K=args.secu_k, tx=args.secu_tx, tw=0.05,
        dim=128, num_ins=args.secu_num_ins, alpha=args.secu_alpha,
        dual_lr=0.1, lratio=args.secu_lratio, constraint=args.secu_cst,
    )
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device).eval()
    ckpt = torch.load(args.model_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.load_param()

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(), normalize])
    loader = DataLoader(ImageDataset(args.data_path, transform=transform),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    feats, preds = [], []
    with torch.no_grad():
        for _, images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            feats.append(model.get_feature(images).cpu().numpy())
            preds.append(torch.argmax(model.get_pred(images), dim=1).cpu().numpy())
    X = np.concatenate(feats)
    y = np.concatenate(preds)
    print(f"[data] N={len(y)} feat_dim={X.shape[1]} clusters={len(np.unique(y))}")

    print("[tsne] running 2-D t-SNE (cosine)...")
    emb = TSNE(n_components=2, metric="cosine", init="pca",
               perplexity=30, random_state=0).fit_transform(X)

    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap("tab10")
    for c in sorted(np.unique(y)):
        m = y == c
        plt.scatter(emb[m, 0], emb[m, 1], s=6, color=cmap(c % 10),
                    label=f"cluster {c}", alpha=0.7, linewidths=0)
    plt.xticks([]); plt.yticks([])
    plt.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.9, ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"[out] saved {args.out}")


if __name__ == "__main__":
    main()
