"""
crop_border.py — 批次裁切圖片四邊的邊框

用法：
  python crop_border.py                              # 四邊各裁 100px
  python crop_border.py --all 200                    # 四邊各裁 200px
  python crop_border.py --top 100 --bottom 100       # 只裁上下
  python crop_border.py --top 50 --bottom 50 --left 30 --right 30  # 分別指定
  python crop_border.py --dry-run                    # 預覽不儲存

輸出：
  cropped_border/ 資料夾，檔名與原始相同
"""

import argparse
import logging
import time
from pathlib import Path

from PIL import Image, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None


def crop_border(input_dir: Path, output_dir: Path,
                top: int, bottom: int, left: int, right: int,
                dry_run: bool) -> None:

    images = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        and p.is_file()
    )

    if not images:
        log.warning("在 '%s' 找不到任何圖片", input_dir)
        return

    log.info("找到 %d 張圖片", len(images))
    log.info("裁切範圍 — 上:%dpx  下:%dpx  左:%dpx  右:%dpx", top, bottom, left, right)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    success = errors = 0

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img.load()
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                w, h = img.size

                # 確保裁切範圍不超過圖片大小
                l = min(left,  w // 4)
                r = min(right, w // 4)
                t = min(top,   h // 4)
                b = min(bottom, h // 4)

                new_w = w - l - r
                new_h = h - t - b

                cropped = img.crop((l, t, w - r, h - b))

                if dry_run:
                    log.info("[dry-run] %s (%dx%d) → (%dx%d)",
                             img_path.name, w, h, new_w, new_h)
                else:
                    out_path = output_dir / img_path.name
                    cropped.save(out_path, quality=95)
                    log.info("✓ %s (%dx%d) → (%dx%d)",
                             img_path.name, w, h, new_w, new_h)

            success += 1

        except Exception as e:
            log.error("✗ %s：%s", img_path.name, e)
            errors += 1

    elapsed = time.perf_counter() - t0
    log.info("─" * 50)
    mode = "[dry-run] " if dry_run else ""
    log.info("✅ %s完成！耗時 %.1f 秒", mode, elapsed)
    log.info("成功：%d 張，失敗：%d 張", success, errors)
    if not dry_run:
        log.info("輸出資料夾：%s", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="批次裁切圖片四邊邊框",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",   default="masked_output",   help="輸入資料夾")
    parser.add_argument("-o", "--output",  default="cropped_border", help="輸出資料夾")
    parser.add_argument("--all",    type=int, default=None, help="四邊裁切相同大小（px）")
    parser.add_argument("--top",    type=int, default=100,  help="上方裁切（px）")
    parser.add_argument("--bottom", type=int, default=100,  help="下方裁切（px）")
    parser.add_argument("--left",   type=int, default=100,  help="左方裁切（px）")
    parser.add_argument("--right",  type=int, default=100,  help="右方裁切（px）")
    parser.add_argument("--dry-run", action="store_true",   help="預覽不儲存")

    args = parser.parse_args()

    # --all 覆蓋四邊個別設定
    if args.all is not None:
        args.top = args.bottom = args.left = args.right = args.all

    return args


if __name__ == "__main__":
    args = parse_args()
    crop_border(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        top=args.top,
        bottom=args.bottom,
        left=args.left,
        right=args.right,
        dry_run=args.dry_run,
    )