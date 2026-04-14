"""
filter_yellow.py — 將 tiles 依黃色背景比例分類到不同資料夾

用法：
  python filter_yellow.py                                      # 預設設定
  python filter_yellow.py --input tiles_output                 # 指定輸入資料夾
  python filter_yellow.py --threshold 0.9                      # 調整黃色閾值
  python filter_yellow.py --filter-gray                        # 額外過濾含灰白邊緣的 tiles
  python filter_yellow.py --threshold 0.9 --filter-gray        # 同時套用兩種過濾
  python filter_yellow.py --dry-run                            # 預覽結果不移動檔案

黃色判斷邏輯（HSV）：
  H=20~55、S>=40、V>=80

灰白過濾邏輯：
  S<=40 且 V>=180，比例超過 gray-ratio 視為邊緣 tile

黑色過濾邏輯：
  V<=30，比例超過 black-ratio 視為黑色遮罩區域，直接歸入背景

輸出結構：
  filtered_output/
  ├── candidate/    ← 可能含有目標的 tiles（送去訓練）
  ├── background/   ← 黃色或黑色背景比例高的 tiles（備份）
  └── edge/         ← 含灰白邊緣的 tiles（僅 --filter-gray 時產生）
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

PRESETS = {
    "loose":  0.5,
    "normal": 0.6,
    "strict": 0.75,
}

# 黃色範圍
YELLOW_H_MIN = 20
YELLOW_H_MAX = 55
YELLOW_S_MIN = 40
YELLOW_V_MIN = 80

# 灰白範圍
GRAY_S_MAX = 40
GRAY_V_MIN = 180
GRAY_RATIO_THRESHOLD = 0.10

# 黑色範圍
BLACK_V_MAX = 30
BLACK_RATIO_THRESHOLD = 0.10  # 黑色像素超過 10% 視為黑色遮罩區域


def is_yellow_pixel(h, s, v):
    return (
        (h >= YELLOW_H_MIN) & (h <= YELLOW_H_MAX) &
        (s >= YELLOW_S_MIN) &
        (v >= YELLOW_V_MIN)
    )


def is_gray_pixel(s, v):
    return (s <= GRAY_S_MAX) & (v >= GRAY_V_MIN)


def is_black_pixel(v):
    return v <= BLACK_V_MAX


def analyze_tile(img_path):
    with Image.open(img_path) as img:
        arr = np.array(img.convert("HSV"), dtype=np.uint8)
        h, s, v = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        total = h.size
        yellow = float(is_yellow_pixel(h, s, v).sum()) / total
        gray   = float(is_gray_pixel(s, v).sum()) / total
        black  = float(is_black_pixel(v).sum()) / total
    return {"yellow": yellow, "gray": gray, "black": black}


def filter_tiles(input_dir, output_dir, threshold, filter_gray, dry_run, gray_ratio, black_ratio):
    candidate_dir  = output_dir / "candidate"
    background_dir = output_dir / "background"
    edge_dir       = output_dir / "edge"

    if not dry_run:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        background_dir.mkdir(parents=True, exist_ok=True)
        if filter_gray:
            edge_dir.mkdir(parents=True, exist_ok=True)

    all_tiles = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )

    if not all_tiles:
        log.warning("在 '%s' 找不到任何圖片。", input_dir)
        return

    log.info("共找到 %d 張 tiles，開始過濾...", len(all_tiles))
    log.info("黃色閾值：%.2f  黑色閾值：%.2f", threshold, black_ratio)
    if filter_gray:
        log.info("已啟用灰白邊緣過濾（灰白像素 > %.0f%%）", gray_ratio * 100)

    t0 = time.perf_counter()
    candidate_count = background_count = edge_count = error_count = 0

    for i, tile_path in enumerate(all_tiles, 1):
        try:
            result = analyze_tile(tile_path)
            yellow_r = result["yellow"]
            gray_r   = result["gray"]
            black_r  = result["black"]

            # 判斷優先順序：黑色 > 灰白邊緣 > 黃色 > candidate
            if black_r >= black_ratio:
                target_dir = background_dir
                background_count += 1
                tag = "bg(black)"
            elif filter_gray and gray_r >= gray_ratio:
                target_dir = edge_dir
                edge_count += 1
                tag = "edge"
            elif yellow_r >= threshold:
                target_dir = background_dir
                background_count += 1
                tag = "bg(yellow)"
            else:
                target_dir = candidate_dir
                candidate_count += 1
                tag = "candidate"

            if dry_run:
                if i <= 20 or i % 5000 == 0:
                    log.info("[dry-run] %s  黃=%.2f  灰=%.2f  黑=%.2f  → %s",
                             tile_path.name, yellow_r, gray_r, black_r, tag)
            else:
                rel  = tile_path.relative_to(input_dir)
                dest = target_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tile_path, dest)

        except Exception as e:
            log.error("✗ 處理失敗：%s — %s", tile_path.name, e)
            error_count += 1

        if i % 5000 == 0:
            log.info("  進度：%d / %d", i, len(all_tiles))

    elapsed = time.perf_counter() - t0
    total = len(all_tiles)

    log.info("─" * 50)
    mode = "[dry-run] " if dry_run else ""
    log.info("✅ %s完成！耗時 %.1f 秒", mode, elapsed)
    log.info("候選 tiles（送去訓練）：%d 張  →  %s", candidate_count, candidate_dir)
    log.info("背景 tiles（備份）    ：%d 張  →  %s", background_count, background_dir)
    if filter_gray:
        log.info("邊緣 tiles（灰白）    ：%d 張  →  %s", edge_count, edge_dir)
    if error_count:
        log.warning("處理失敗：%d 張", error_count)
    log.info("保留比例：%.1f%%", 100 * candidate_count / max(total, 1))


def parse_args():
    parser = argparse.ArgumentParser(
        description="依黃色/黑色背景比例過濾 tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",     default="scripts/tiles_output",    help="輸入資料夾")
    parser.add_argument("-o", "--output",    default="filtered_output",  help="輸出根資料夾")
    parser.add_argument("-t", "--threshold", type=float, default=None,   help="黃色閾值（預設 0.6）")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                        help="快速預設：loose(0.5) / normal(0.6) / strict(0.75)")
    parser.add_argument("--filter-gray", action="store_true",
                        help="額外過濾含灰白邊緣的 tiles（移至 edge/ 資料夾）")
    parser.add_argument("--gray-ratio", type=float, default=GRAY_RATIO_THRESHOLD,
                        help="灰白像素比例閾值（預設 0.10）")
    parser.add_argument("--black-ratio", type=float, default=BLACK_RATIO_THRESHOLD,
                        help="黑色像素比例閾值（預設 0.10）")
    parser.add_argument("--dry-run", action="store_true",
                        help="僅預覽分類結果，不實際複製檔案")

    args = parser.parse_args()

    if args.threshold is not None:
        threshold = args.threshold
    elif args.preset:
        threshold = PRESETS[args.preset]
        log.info("使用預設 [%s]：threshold=%.2f", args.preset, threshold)
    else:
        threshold = PRESETS["normal"]
        log.info("使用預設 threshold=%.2f", threshold)

    return args, threshold


if __name__ == "__main__":
    args, threshold = parse_args()
    filter_tiles(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        threshold=threshold,
        filter_gray=args.filter_gray,
        dry_run=args.dry_run,
        gray_ratio=args.gray_ratio,
        black_ratio=args.black_ratio,
    )