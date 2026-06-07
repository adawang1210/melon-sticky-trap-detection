"""
subcluster_stats.py — quantify the hierarchical sub-clustering stage (Item 1).

For each coarse cluster, run the ensemble sub-clustering and collect:
  - number of sub-clusters discovered
  - ensemble Silhouette (consensus space)
  - per-sample stability distribution: stable (>=0.7), uncertain (0.4-0.7), unstable (<0.4)

Then aggregate across all coarse clusters into paper-ready numbers.

Usage (from scripts/):
    python subcluster_stats.py -i ../Secu-revised/output/cluster8
"""
import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np

import subcluster as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="../Secu-revised/output/cluster8")
    ap.add_argument("--color-weight", type=float, default=1.0)
    ap.add_argument("--size-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    feat_args = SimpleNamespace(device=args.device, batch_size=args.batch_size,
                                color_weight=args.color_weight,
                                size_weight=args.size_weight, dr="pca_umap")

    root = Path(args.input)
    cluster_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("cluster_") and "subcluster" not in p.name
    )

    tot_imgs = tot_sub = tot_stable = tot_uncertain = tot_unstable = 0
    sil_weighted = 0.0
    per = []
    for cd in cluster_dirs:
        paths = sc.collect_images(cd)
        feats, _ = sc.extract_all_features(paths, feat_args)
        labels, k, score, stability = sc.run_ensemble(
            feats, paths, min_cluster_size=15, min_samples=5)
        n = len(paths)
        st = int((stability >= 0.7).sum())
        un = int(((stability >= 0.4) & (stability < 0.7)).sum())
        us = int((stability < 0.4).sum())
        per.append((cd.name, n, k, score, st, un, us))
        tot_imgs += n; tot_sub += k
        tot_stable += st; tot_uncertain += un; tot_unstable += us
        if score is not None:
            sil_weighted += score * n

    print("\n================= SUB-CLUSTERING STATS (ensemble) =================")
    print(f"{'cluster':>10}{'imgs':>6}{'#sub':>5}{'silh':>7}{'stable%':>9}{'uncert%':>9}{'unstab%':>9}")
    for name, n, k, score, st, un, us in per:
        s = f"{score:.3f}" if score is not None else "  -  "
        print(f"{name:>10}{n:>6}{k:>5}{s:>7}{100*st/n:>8.1f}%{100*un/n:>8.1f}%{100*us/n:>8.1f}%")
    print("-" * 64)
    print(f"coarse clusters processed : {len(per)}")
    print(f"total images              : {tot_imgs}")
    print(f"total sub-clusters        : {tot_sub}")
    print(f"weighted mean Silhouette  : {sil_weighted / max(tot_imgs,1):.3f}")
    print(f"stability  stable / uncertain / unstable :"
          f" {100*tot_stable/tot_imgs:.1f}% / {100*tot_uncertain/tot_imgs:.1f}% / {100*tot_unstable/tot_imgs:.1f}%")
    print("===================================================================")


if __name__ == "__main__":
    main()
