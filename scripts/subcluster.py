"""
subcluster.py — 對單一 cluster 進行子聚類（Hierarchical Clustering）

使用 frozen DINOv2 提取特徵，再用 K-Means 進行子分群。
不需要 GPU 訓練，幾分鐘即可完成。

用法：
  python subcluster.py -i ../Secu-revised/output/cluster9/cluster_1 -k 3
  python subcluster.py -i ../Secu-revised/output/cluster9/cluster_1 -k 2 3 4 5 --preview
  python subcluster.py -i <cluster資料夾> -k <子分群數> -o <輸出資料夾>

輸出：
  output/
  ├── k3/
  │   ├── sub_0/    ← 子群 0 的圖片
  │   ├── sub_1/
  │   └── sub_2/
  └── k3_preview/
      ├── sub_0.jpg ← 預覽圖（--preview 時產生）
      └── ...
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


def collect_images(input_dir: Path):
    """收集所有圖片路徑。"""
    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )
    if not images:
        raise FileNotFoundError(f"在 '{input_dir}' 找不到任何圖片")
    log.info("找到 %d 張圖片", len(images))
    return images


def extract_features(image_paths, device="cuda", batch_size=32):
    """用 frozen DINOv2 (timm) 提取特徵。"""
    import timm
    from torchvision import transforms

    log.info("載入 DINOv2 模型...")
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=True,
        num_classes=0,
        img_size=224,
    )
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    log.info("提取特徵中（batch_size=%d）...", batch_size)

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(transform(img))

            batch = torch.stack(batch_tensors).to(device)
            features = model(batch)  # (B, 768)
            all_features.append(features.cpu().numpy())

            if (i // batch_size) % 5 == 0:
                log.info("  進度：%d / %d", min(i + batch_size, len(image_paths)), len(image_paths))

    features = np.concatenate(all_features, axis=0)
    log.info("特徵提取完成，shape: %s", features.shape)
    return features


def run_kmeans(features, k):
    """執行 K-Means 聚類，並計算 Silhouette Score。"""
    from sklearn.metrics import silhouette_score

    log.info("執行 K-Means (k=%d)...", k)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)

    # Silhouette Score：-1（最差）到 1（最好），越高代表分群越清晰
    score = silhouette_score(features, labels)
    log.info("  Silhouette Score: %.4f", score)

    # 印出每個子群的大小
    for c in range(k):
        count = (labels == c).sum()
        log.info("  sub_%d: %d 張", c, count)

    return labels, score


def save_results(image_paths, labels, k, output_dir: Path):
    """將圖片複製到對應的子群資料夾。"""
    out = output_dir / f"k{k}"
    if out.exists():
        shutil.rmtree(out)

    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        sub_dir = out / f"sub_{label}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        dst = sub_dir / path.name
        shutil.copy2(path, dst)

    log.info("結果存於 %s", out)
    return out


def make_preview(result_dir: Path, output_dir: Path, cols=10, rows=5, size=128):
    """為每個子群產生預覽圖。"""
    import random

    preview_dir = output_dir
    preview_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted(p for p in result_dir.iterdir() if p.is_dir())
    n = cols * rows

    for subdir in subdirs:
        tiles = sorted(
            p for p in subdir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        )
        if not tiles:
            continue

        selected = random.sample(tiles, min(n, len(tiles)))

        total_w = cols * size + (cols - 1) * 4
        total_h = rows * size + (rows - 1) * 4
        canvas = Image.new("RGB", (total_w, total_h), (50, 50, 50))

        for idx, path in enumerate(selected):
            r, c = idx // cols, idx % cols
            x, y = c * (size + 4), r * (size + 4)
            try:
                img = Image.open(path).convert("RGB").resize((size, size))
                canvas.paste(img, (x, y))
            except Exception:
                pass

        out_path = preview_dir / f"{subdir.name}.jpg"
        canvas.save(out_path, quality=95)
        log.info("  預覽 → %s", out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="對單一 cluster 進行子聚類",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="要子聚類的 cluster 資料夾")
    parser.add_argument("-o", "--output", default=None, help="輸出資料夾（預設：<input>_subcluster）")
    parser.add_argument("-k", "--k", type=int, nargs="+", default=[3], help="子分群數（可指定多個，如 -k 2 3 4）")
    parser.add_argument("--device", default="cuda", help="計算裝置")
    parser.add_argument("--batch-size", type=int, default=32, help="特徵提取 batch size")
    parser.add_argument("--preview", action="store_true", help="產生預覽圖")
    parser.add_argument("--preview-cols", type=int, default=10, help="預覽圖每排張數")
    parser.add_argument("--preview-rows", type=int, default=5, help="預覽圖排數")
    parser.add_argument("--preview-size", type=int, default=128, help="預覽圖每張大小")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir.parent / f"{input_dir.name}_subcluster"

    # 收集圖片 & 提取特徵（只做一次）
    image_paths = collect_images(input_dir)
    features = extract_features(image_paths, device=args.device, batch_size=args.batch_size)

    # 對每個 k 值跑 K-Means
    scores = {}
    for k in args.k:
        labels, score = run_kmeans(features, k)
        scores[k] = score
        result_dir = save_results(image_paths, labels, k, output_dir)

        if args.preview:
            preview_dir = output_dir / f"k{k}_preview"
            make_preview(result_dir, preview_dir,
                         cols=args.preview_cols,
                         rows=args.preview_rows,
                         size=args.preview_size)

    # 印出總結：推薦最佳 K
    if len(scores) > 1:
        log.info("=" * 50)
        log.info("Silhouette Score 總結（越高越好）：")
        for k, s in sorted(scores.items()):
            marker = " ← 推薦" if s == max(scores.values()) else ""
            log.info("  K=%d : %.4f%s", k, s, marker)
        best_k = max(scores, key=scores.get)
        log.info("建議使用 K=%d（Silhouette Score 最高）", best_k)
        log.info("=" * 50)

    log.info("✅ 全部完成！")
