"""
preview_clusters.py — 將 cluster 資料夾內每個 cluster 各印出 N 張預覽圖

用法：
  python preview_clusters.py                                # 預設從 data/train/cluster7 讀取
  python preview_clusters.py --input data/train/cluster7   # 指定資料夾
  python preview_clusters.py --n 50                        # 每個 cluster 印 50 張
  python preview_clusters.py --cols 10 --rows 10           # 10x10 排列

輸出：
  cluster_previews/
  ├── cluster_0.jpg
  ├── cluster_1.jpg
  └── ...
"""

import argparse
import logging
import random
from pathlib import Path

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def make_grid(tiles: list, cols: int, rows: int, gap: int, tile_size: int) -> Image.Image:
    n = cols * rows
    selected = tiles[:n]

    total_w = cols * tile_size + (cols - 1) * gap
    total_h = rows * tile_size + (rows - 1) * gap
    canvas = Image.new("RGB", (total_w, total_h), color=(50, 50, 50))

    for idx, path in enumerate(selected):
        row = idx // cols
        col = idx % cols
        x = col * (tile_size + gap)
        y = row * (tile_size + gap)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize((tile_size, tile_size), Image.LANCZOS)
                canvas.paste(img, (x, y))
        except Exception as e:
            log.warning("略過 %s：%s", path.name, e)

    return canvas


def preview_clusters(input_dir: Path, output_dir: Path,
                     n: int, cols: int, rows: int,
                     gap: int, tile_size: int, seed: int) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())

    if not cluster_dirs:
        log.warning("在 '%s' 找不到任何子資料夾", input_dir)
        return

    log.info("找到 %d 個 cluster", len(cluster_dirs))

    for cluster_dir in cluster_dirs:
        tiles = sorted(
            p for p in cluster_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        )

        if not tiles:
            log.warning("  %s：沒有圖片，略過", cluster_dir.name)
            continue

        log.info("  %s：共 %d 張，抽取 %d 張", cluster_dir.name, len(tiles), min(n, len(tiles)))

        selected = random.sample(tiles, min(n, len(tiles)))
        selected.sort()

        preview = make_grid(selected, cols, rows, gap, tile_size)
        out_path = output_dir / f"{cluster_dir.name}.jpg"
        preview.save(out_path, quality=95)
        log.info("  ✓ → %s", out_path)

    log.info("✅ 完成！預覽圖存於 %s/", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="將每個 cluster 的照片印成預覽圖",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",  default=r"..\Secu-revised\data\train\cluster7", help="cluster 根資料夾")
    parser.add_argument("-o", "--output", default="cluster_previews",     help="輸出資料夾")
    parser.add_argument("-n", "--n",      type=int, default=100,          help="每個 cluster 抽幾張")
    parser.add_argument("--cols",    type=int, default=10,  help="每排張數")
    parser.add_argument("--rows",    type=int, default=10,  help="排數")
    parser.add_argument("--gap",     type=int, default=4,   help="圖片間距（px）")
    parser.add_argument("--size",    type=int, default=128, help="每張圖顯示大小（px）")
    parser.add_argument("--seed",    type=int, default=42,  help="隨機種子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preview_clusters(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        n=args.n,
        cols=args.cols,
        rows=args.rows,
        gap=args.gap,
        tile_size=args.size,
        seed=args.seed,
    )