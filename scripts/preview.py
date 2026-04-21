"""
preview.py — 將資料夾內每個子資料夾各抽樣 N 張圖片，排成網格預覽圖

可用於預覽聚類結果、候選圖塊、或任何以子資料夾分組的影像集。

用法：
  python preview.py -i data/train/cluster8           # 預覽聚類結果
  python preview.py -i filtered_output/candidate     # 預覽候選圖塊
  python preview.py -i some/folder --cols 5 --rows 5 # 自訂排列
  python preview.py -i some/folder --size 200 -n 50  # 大圖、每群 50 張

輸出：
  previews/
  ├── <子資料夾名稱>.jpg
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


def make_grid(tiles: list, cols: int, rows: int,
              gap: int, tile_size: int) -> Image.Image:
    """從 tiles 清單取前 cols*rows 張，排成網格圖。"""
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
                img = img.convert("RGB").resize(
                    (tile_size, tile_size), Image.LANCZOS)
                canvas.paste(img, (x, y))
        except Exception as e:
            log.warning("略過 %s：%s", path.name, e)

    return canvas


def preview_all(input_dir: Path, output_dir: Path,
                n: int, cols: int, rows: int,
                gap: int, tile_size: int, seed: int) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted(p for p in input_dir.iterdir() if p.is_dir())

    # 如果沒有子資料夾，把 input_dir 本身當一個群
    if not subdirs:
        subdirs = [input_dir]

    log.info("找到 %d 個子資料夾", len(subdirs))

    n_per_preview = min(n, cols * rows)

    for subdir in subdirs:
        tiles = sorted(
            p for p in subdir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        )

        if not tiles:
            log.warning("  %s：沒有圖片，略過", subdir.name)
            continue

        if len(tiles) > n_per_preview:
            selected = sorted(random.sample(tiles, n_per_preview))
        else:
            selected = tiles

        log.info("  %s：共 %d 張，抽取 %d 張",
                 subdir.name, len(tiles), len(selected))

        preview = make_grid(selected, cols, rows, gap, tile_size)
        out_path = output_dir / f"{subdir.name}.jpg"
        preview.save(out_path, quality=95)
        log.info("  ✓ → %s", out_path)

    log.info("✅ 完成！預覽圖存於 %s/", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="將每個子資料夾的圖片排成網格預覽圖",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",  required=True,       help="輸入資料夾（含子資料夾）")
    parser.add_argument("-o", "--output", default="previews",  help="輸出資料夾")
    parser.add_argument("-n", "--n",      type=int, default=100, help="每個子資料夾最多抽幾張")
    parser.add_argument("--cols",  type=int, default=10,  help="每排張數")
    parser.add_argument("--rows",  type=int, default=10,  help="排數")
    parser.add_argument("--gap",   type=int, default=4,   help="圖片間距（px）")
    parser.add_argument("--size",  type=int, default=128, help="每張圖顯示大小（px）")
    parser.add_argument("--seed",  type=int, default=42,  help="隨機種子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preview_all(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        n=args.n,
        cols=args.cols,
        rows=args.rows,
        gap=args.gap,
        tile_size=args.size,
        seed=args.seed,
    )
