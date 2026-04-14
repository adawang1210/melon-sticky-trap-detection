"""
crop_corners.py — 批次將 Bugdatasets 底下所有 jpg 的右上角和右下角塗黑

用法：
  python crop_corners.py                        # 預設塗黑 2000px
  python crop_corners.py --size 1000            # 改成 1000px
  python crop_corners.py --input Bugdatasets    # 指定輸入資料夾
  python crop_corners.py --output masked        # 指定輸出資料夾
  python crop_corners.py --dry-run              # 預覽不儲存

輸出：
  圖片大小不變，右上角和右下角各塗黑 size x size px
"""

import argparse
import logging
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None


def mask_corners(input_dir: Path, output_dir: Path, size: int, dry_run: bool) -> None:
    images = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() == ".jpg" and p.is_file()
    )

    if not images:
        log.warning("在 '%s' 找不到任何 jpg 檔案", input_dir)
        return

    log.info("找到 %d 張圖片，塗黑範圍：%d x %d px", len(images), size, size)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    success = errors = 0

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img.load()
                img = ImageOps.exif_transpose(img)  # 修正 EXIF 旋轉方向
                img = img.convert("RGB")
                w, h = img.size

                fill_w = min(size, w)
                fill_h = min(size, h)

                draw = ImageDraw.Draw(img)
                # 右上角塗黑
                draw.rectangle([(w - fill_w, 0), (w, fill_h)], fill=(0, 0, 0))
                # 右下角塗黑
                draw.rectangle([(w - fill_w, h - fill_h), (w, h)], fill=(0, 0, 0))

                if dry_run:
                    log.info("[dry-run] %s (%dx%d) → 右上+右下各塗黑 %dpx",
                             img_path.name, w, h, size)
                else:
                    out_path = output_dir / img_path.name
                    img.save(out_path, quality=95)
                    log.info("✓ %s (%dx%d)", img_path.name, w, h)

            success += 1

        except Exception as e:
            log.error("✗ %s：%s", img_path.name, e)
            errors += 1

    elapsed = time.perf_counter() - t0
    log.info("─" * 50)
    log.info("✅ 完成！耗時 %.1f 秒", elapsed)
    log.info("成功：%d 張，失敗：%d 張", success, errors)
    if not dry_run:
        log.info("輸出資料夾：%s", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="批次將 jpg 右上角和右下角塗黑",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",  default="Bugdatasets",  help="輸入資料夾")
    parser.add_argument("-o", "--output", default="masked_output", help="輸出資料夾")
    parser.add_argument("--size", type=int, default=3000,          help="塗黑大小（px），預設 2000")
    parser.add_argument("--dry-run", action="store_true",          help="預覽不儲存")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mask_corners(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        size=args.size,
        dry_run=args.dry_run,
    )