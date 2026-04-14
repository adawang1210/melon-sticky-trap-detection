"""
adaptive_tile.py — 自適應切割：自動找出非黃色區域並擴張到四周都是黃色背景

邏輯：
  1. 用小格子（probe_size）掃描整張大圖
  2. 發現非黃色的格子（可能有蟲）
  3. 以該格子為中心往外擴張，直到四周都是黃色為止
  4. 加上 padding 後裁出，存成獨立圖片

用法：
  python adaptive_tile.py                          # 預設從 cropped_border 讀取
  python adaptive_tile.py --input cropped_border   # 指定輸入資料夾
  python adaptive_tile.py --probe 32               # 探測格子大小（預設 32px）
  python adaptive_tile.py --padding 20             # 擴張後額外加的邊距（預設 20px）
  python adaptive_tile.py --yellow-threshold 0.85  # 格子黃色比例超過此值視為背景
  python adaptive_tile.py --dry-run                # 預覽不儲存
  python adaptive_tile.py --preview                # 輸出標記框的預覽圖
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# 黃色定義（HSV）
YELLOW_H_MIN = 30
YELLOW_H_MAX = 55
YELLOW_S_MIN = 40
YELLOW_V_MIN = 80

# 黑色定義
BLACK_V_MAX = 30


def is_yellow(h, s, v):
    return (
        (h >= YELLOW_H_MIN) & (h <= YELLOW_H_MAX) &
        (s >= YELLOW_S_MIN) &
        (v >= YELLOW_V_MIN)
    )


def is_black(v):
    return v <= BLACK_V_MAX


def get_yellow_ratio(hsv_arr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """計算指定區域的黃色像素比例。"""
    region = hsv_arr[y1:y2, x1:x2]
    if region.size == 0:
        return 1.0
    h, s, v = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    yellow = is_yellow(h, s, v).sum()
    black  = is_black(v).sum()
    # 黑色也算背景
    bg = yellow + black
    return float(bg) / h.size


def find_blobs(hsv_arr: np.ndarray, img_w: int, img_h: int,
               probe: int, yellow_threshold: float) -> list:
    """
    用探測格子掃描整張圖，找出所有非背景的格子，
    然後把相鄰的格子合併成一個 bounding box。
    回傳 list of (x1, y1, x2, y2)。
    """
    # 第一步：找出所有非背景格子
    non_bg = []
    for y in range(0, img_h, probe):
        for x in range(0, img_w, probe):
            x2 = min(x + probe, img_w)
            y2 = min(y + probe, img_h)
            ratio = get_yellow_ratio(hsv_arr, x, y, x2, y2)
            if ratio < yellow_threshold:
                non_bg.append((x, y, x2, y2))

    if not non_bg:
        return []

    # 第二步：把相鄰格子合併（Union-Find 風格的簡單合併）
    def overlaps_or_adjacent(a, b, gap=1):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        # 允許 gap 個 probe 的距離內視為同一群
        return (ax1 - gap * probe <= bx2 and bx1 - gap * probe <= ax2 and
                ay1 - gap * probe <= by2 and by1 - gap * probe <= ay2)

    blobs = []
    for box in non_bg:
        merged = False
        for i, blob in enumerate(blobs):
            if overlaps_or_adjacent(blob, box):
                # 合併：取聯集
                blobs[i] = (
                    min(blob[0], box[0]),
                    min(blob[1], box[1]),
                    max(blob[2], box[2]),
                    max(blob[3], box[3]),
                )
                merged = True
                break
        if not merged:
            blobs.append(box)

    # 第三步：再跑一次合併，確保所有重疊的 blob 都合併
    changed = True
    while changed:
        changed = False
        new_blobs = []
        used = [False] * len(blobs)
        for i in range(len(blobs)):
            if used[i]:
                continue
            cur = blobs[i]
            for j in range(i + 1, len(blobs)):
                if used[j]:
                    continue
                if overlaps_or_adjacent(cur, blobs[j]):
                    cur = (
                        min(cur[0], blobs[j][0]),
                        min(cur[1], blobs[j][1]),
                        max(cur[2], blobs[j][2]),
                        max(cur[3], blobs[j][3]),
                    )
                    used[j] = True
                    changed = True
            new_blobs.append(cur)
            used[i] = True
        blobs = new_blobs

    return blobs


def process_image(img_path: Path, output_dir: Path,
                  probe: int, padding: int,
                  yellow_threshold: float,
                  dry_run: bool, preview: bool) -> int:
    """處理單張圖片，回傳找到的 blob 數量。"""
    with Image.open(img_path) as img:
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        hsv_arr = np.array(img.convert("HSV"), dtype=np.uint8)

    blobs = find_blobs(hsv_arr, w, h, probe, yellow_threshold)

    if not blobs:
        log.info("  %s → 未找到目標", img_path.name)
        return 0

    log.info("  %s (%dx%d) → 找到 %d 個目標", img_path.name, w, h, len(blobs))

    # 輸出預覽圖
    if preview and not dry_run:
        preview_img = img.copy()
        draw = ImageDraw.Draw(preview_img)
        for bx1, by1, bx2, by2 in blobs:
            px1 = max(0, bx1 - padding)
            py1 = max(0, by1 - padding)
            px2 = min(w, bx2 + padding)
            py2 = min(h, by2 + padding)
            draw.rectangle([(px1, py1), (px2, py2)], outline=(255, 0, 0), width=3)
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        # 縮小預覽圖
        MAX = 2048
        scale = min(MAX / max(w, h), 1.0)
        if scale < 1.0:
            preview_img = preview_img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS)
        preview_img.save(preview_dir / f"{img_path.stem}_preview.jpg", quality=85)

    if dry_run:
        return len(blobs)

    # 儲存每個 blob
    stem = img_path.stem
    pad_digits = len(str(len(blobs)))

    for idx, (bx1, by1, bx2, by2) in enumerate(blobs, 1):
        # 加 padding
        px1 = max(0, bx1 - padding)
        py1 = max(0, by1 - padding)
        px2 = min(w, bx2 + padding)
        py2 = min(h, by2 + padding)

        cropped = img.crop((px1, py1, px2, py2))
        out_name = f"{stem}_obj{idx:0{pad_digits}d}.jpg"
        img_out_dir = output_dir / stem
        img_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = img_out_dir / out_name
        cropped.save(out_path, quality=95)

    return len(blobs)


def process_all(input_dir: Path, output_dir: Path,
                probe: int, padding: int,
                yellow_threshold: float,
                dry_run: bool, preview: bool) -> None:

    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )

    if not images:
        log.warning("在 '%s' 找不到任何圖片", input_dir)
        return

    log.info("找到 %d 張圖片", len(images))
    log.info("探測格子：%dpx  邊距：%dpx  黃色閾值：%.2f", probe, padding, yellow_threshold)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    total_blobs = 0
    errors = 0

    for img_path in images:
        try:
            count = process_image(img_path, output_dir, probe, padding,
                                  yellow_threshold, dry_run, preview)
            total_blobs += count
        except Exception as e:
            log.error("✗ %s：%s", img_path.name, e)
            errors += 1

    elapsed = time.perf_counter() - t0
    log.info("─" * 50)
    mode = "[dry-run] " if dry_run else ""
    log.info("✅ %s完成！耗時 %.1f 秒", mode, elapsed)
    log.info("找到目標總數：%d 個", total_blobs)
    if errors:
        log.warning("失敗：%d 張", errors)
    if not dry_run:
        log.info("輸出資料夾：%s", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="自適應切割：自動找出非黃色區域並裁出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",   default="../cropped_border", help="輸入資料夾")
    parser.add_argument("-o", "--output",  default="../adaptive_output", help="輸出資料夾")
    parser.add_argument("--probe",    type=int,   default=32,   help="探測格子大小（px）")
    parser.add_argument("--padding",  type=int,   default=20,   help="目標區域外加的邊距（px）")
    parser.add_argument("--yellow-threshold", type=float, default=0.85,
                        help="格子黃色+黑色比例超過此值視為背景（預設 0.85）")
    parser.add_argument("--preview",  action="store_true", help="輸出標記框的預覽圖")
    parser.add_argument("--dry-run",  action="store_true", help="預覽不儲存")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        probe=args.probe,
        padding=args.padding,
        yellow_threshold=args.yellow_threshold,
        dry_run=args.dry_run,
        preview=args.preview,
    )