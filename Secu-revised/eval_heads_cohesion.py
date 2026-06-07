"""
eval_heads_cohesion.py — two label-free analyses on the released model:

  (Item 2) K-selection: score each of the multi-head partitions (K=8,9,10)
           with Silhouette/DBI/CHI to justify reporting K=8.
  (Item 3) Coarse-cluster cohesion: per-cluster mean intra-cluster cosine
           similarity (to the cluster's mean unit vector) for the K=8 head,
           a label-free "how tight is each cluster" measure for Table 2.

Usage (from Secu-revised/):
    python eval_heads_cohesion.py --model-path model/best_dinov2-mml.pth.tar
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

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

    feats = []
    with torch.no_grad():
        for _, images, _, _ in loader:
            feats.append(model.get_feature(images.to(device)).cpu().numpy())
    X = np.concatenate(feats).astype(np.float64)
    Xt = torch.from_numpy(X).float()
    print(f"[data] N={len(X)} dim={X.shape[1]}")

    # ---------- Item 2: K-selection across heads ----------
    print("\n===== K-selection (multi-head) =====")
    print(f"{'K':>3} {'Silh(cos)':>10} {'DBI':>7} {'CHI':>9} {'#empty':>7}")
    head_preds = {}
    for h, K in enumerate(args.secu_k):
        c = F.normalize(getattr(model, f"center_{h}").detach().cpu(), dim=0)  # [128,K]
        pred = torch.argmax(Xt @ c, dim=1).numpy()
        head_preds[K] = pred
        n_found = len(np.unique(pred))
        sil = silhouette_score(X, pred, metric="cosine")
        dbi = davies_bouldin_score(X, pred)
        chi = calinski_harabasz_score(X, pred)
        print(f"{K:>3} {sil:>10.4f} {dbi:>7.3f} {chi:>9.2f} {K-n_found:>7}")

    # ---------- Item 3: per-cluster cohesion for K=8 ----------
    print("\n===== Per-cluster cohesion (K=8): mean intra-cluster cosine =====")
    y = head_preds[8]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rows = []
    for k in range(8):
        m = y == k
        cnt = int(m.sum())
        if cnt == 0:
            rows.append((k, 0, float("nan"))); continue
        sub = Xn[m]
        centroid = sub.mean(0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        coh = float((sub @ centroid).mean())
        rows.append((k, cnt, coh))
    print(f"{'cluster':>7} {'count':>6} {'cohesion':>9}")
    for k, cnt, coh in rows:
        print(f"{k:>7} {cnt:>6} {coh:>9.3f}")
    overall = np.nanmean([r[2] for r in rows])
    print(f"{'mean':>7} {'':>6} {overall:>9.3f}")


if __name__ == "__main__":
    main()
