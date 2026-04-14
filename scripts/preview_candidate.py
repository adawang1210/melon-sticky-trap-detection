"""
preview_candidate.py — 將 candidate 資料夾內每個子資料夾各印出 50 張預覽圖
排列方式：一排 10 張，共 5 排，圖片之間有間隔

用法：
  python preview_candidate.py                                      # 預設讀 filtered_output/candidate
  python preview_candidate.py --input filtered_output/candidate   # 指定資料夾
  python preview_candidate.py --cols 10 --rows 5 --gap 8          # 自訂排列與間隔

輸出：
  previews/
  └── preview_<子資料夾名稱>.jpg   ← 每個子資料夾一張預覽圖
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


def make_preview(tiles: list, cols: int, rows: int, gap: int, tile_display_size: int) -> Image.Image:
    """
    從 tiles 清單中取前 cols*rows 張，排成網格圖。
    tile_display_size: 每張 tile 在預覽圖中的顯示大小（px）
    """
    n = cols * rows
    selected = tiles[:n]

    cell = tile_display_size
    total_w = cols * cell + (cols - 1) * gap
    total_h = rows * cell + (rows - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), color=(240, 240, 240))

    for idx, path in enumerate(selected):
        row = idx // cols
        col = idx % cols
        x = col * (cell + gap)
        y = row * (cell + gap)

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize((cell, cell), Image.LANCZOS)
                canvas.paste(img, (x, y))
        except Exception as e:
            log.warning("略過 %s：%s", path.name, e)

    return canvas


def process_candidate(input_dir: Path, output_dir: Path,
                      cols: int, rows: int, gap: int,
                      tile_display_size: int, seed: int) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找出所有子資料夾（如果沒有子資料夾，就把 input_dir 本身當一個群）
    subdirs = sorted(p for p in input_dir.iterdir() if p.is_dir())

    if not subdirs:
        # 沒有子資料夾，直接處理 input_dir
        subdirs = [input_dir]

    log.info("找到 %d 個子資料夾", len(subdirs))

    n_per_preview = cols * rows

    for subdir in subdirs:
        tiles = sorted(
            p for p in subdir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        )

        if not tiles:
            log.warning("  %s：沒有圖片，略過", subdir.name)
            continue

        # 隨機抽樣（若不足 n_per_preview 張則全取）
        if len(tiles) > n_per_preview:
            selected = random.sample(tiles, n_per_preview)
            selected.sort()
        else:
            selected = tiles
            log.warning("  %s：只有 %d 張（少於 %d），全部使用",
                        subdir.name, len(tiles), n_per_preview)

        log.info("  處理 %s（%d 張）...", subdir.name, len(tiles))

        preview = make_preview(selected, cols, rows, gap, tile_display_size)

        out_path = output_dir / f"preview_{subdir.name}.jpg"
        preview.save(out_path, quality=95)
        log.info("  ✓ 儲存 → %s", out_path)

    log.info("✅ 完成！預覽圖存於 %s/", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="將 candidate 資料夾內每個子資料夾各印出 50 張預覽圖",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",  default="filtered_output/candidate", help="candidate 資料夾路徑")
    parser.add_argument("-o", "--output", default="previews",                  help="輸出資料夾")
    parser.add_argument("--cols",  type=int, default=10,  help="每排張數")
    parser.add_argument("--rows",  type=int, default=10,   help="排數")
    parser.add_argument("--gap",   type=int, default=6,   help="圖片間距（px）")
    parser.add_argument("--size",  type=int, default=64,  help="每張 tile 的顯示大小（px）")
    parser.add_argument("--seed",  type=int, default=42,  help="隨機種子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_candidate(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        cols=args.cols,
        rows=args.rows,
        gap=args.gap,
        tile_display_size=args.size,
        seed=args.seed,
    )