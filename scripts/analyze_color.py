"""
analyze_color.py — 從 tiles 隨機抽樣，分析主要顏色分布

用法：
  python analyze_color.py                          # 預設從 tiles_output 抽 200 張
  python analyze_color.py --input tiles_output     # 指定資料夾
  python analyze_color.py --sample 500             # 抽更多樣本
  python analyze_color.py --output color_report    # 指定輸出資料夾

輸出：
  color_report/
  ├── hsv_distribution.png   ← H/S/V 三個頻道的分布圖
  ├── top_colors.png         ← 前 20 個最常見顏色的色塊
  └── summary.txt            ← 建議的黃色 HSV 範圍
"""

import argparse
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_tiles(input_dir: Path, sample_n: int) -> list:
    """遞迴收集所有 tile 路徑，並隨機抽樣。"""
    all_tiles = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )
    if not all_tiles:
        raise FileNotFoundError(f"在 '{input_dir}' 找不到任何圖片")
    
    if len(all_tiles) <= sample_n:
        log.info("圖片總數 %d 張，全部使用", len(all_tiles))
        return all_tiles
    
    sampled = random.sample(all_tiles, sample_n)
    log.info("圖片總數 %d 張，隨機抽樣 %d 張", len(all_tiles), sample_n)
    return sampled


def extract_hsv_pixels(tiles: list, max_pixels_per_tile: int = 50) -> np.ndarray:
    """從每張 tile 隨機取樣像素，回傳 HSV 陣列 (N, 3)。"""
    all_pixels = []
    for path in tiles:
        try:
            with Image.open(path) as img:
                hsv = np.array(img.convert("HSV"), dtype=np.uint8)
                flat = hsv.reshape(-1, 3)
                if len(flat) > max_pixels_per_tile:
                    idx = np.random.choice(len(flat), max_pixels_per_tile, replace=False)
                    flat = flat[idx]
                all_pixels.append(flat)
        except Exception as e:
            log.warning("略過 %s：%s", path.name, e)
    
    return np.vstack(all_pixels)


def suggest_yellow_range(h_values: np.ndarray) -> dict:
    """
    依 H 頻道分布，自動推算黃色範圍。
    PIL HSV 的 H 是 0–255，對應色相：
      0   = 紅
      32  = 橘黃
      43  = 黃
      85  = 綠
      128 = 青
      170 = 藍
      213 = 紫
      255 = 紅（循環）
    黃色大約在 H=20~55（涵蓋橘黃到亮黃）
    """
    # 統計 H=15~60 的像素比例
    yellow_mask = (h_values >= 15) & (h_values <= 60)
    ratio = yellow_mask.sum() / len(h_values)
    
    if ratio > 0.3:
        # 找黃色區間的峰值
        yellow_h = h_values[yellow_mask]
        peak = int(np.median(yellow_h))
        h_min = max(10, peak - 15)
        h_max = min(65, peak + 15)
        confidence = "高"
    else:
        h_min, h_max = 20, 55
        confidence = "低（黃色像素不多，建議手動確認）"
    
    return {
        "h_min": h_min,
        "h_max": h_max,
        "yellow_pixel_ratio": ratio,
        "confidence": confidence,
    }


def plot_hsv_distribution(pixels: np.ndarray, output_path: Path) -> None:
    """繪製 H / S / V 三個頻道的分布長條圖。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("HSV Color Distribution of Sampled Tiles", fontsize=14, fontweight="bold")

    channels = [
        (pixels[:, 0], "Hue (色相)", "hue"),
        (pixels[:, 1], "Saturation (飽和度)", "saturation"),
        (pixels[:, 2], "Value (亮度)", "value"),
    ]

    hue_colors = []
    for i in range(256):
        # 把 HSV H=i 轉成 RGB 顏色給 matplotlib
        from PIL import Image as PILImage
        px = PILImage.new("HSV", (1, 1), (i, 200, 220))
        r, g, b = px.convert("RGB").getpixel((0, 0))
        hue_colors.append((r/255, g/255, b/255))

    for ax, (data, title, kind) in zip(axes, channels):
        counts, edges = np.histogram(data, bins=64, range=(0, 255))
        centers = (edges[:-1] + edges[1:]) / 2
        
        if kind == "hue":
            bar_colors = [hue_colors[min(int(c), 255)] for c in centers]
            ax.bar(centers, counts, width=4, color=bar_colors, edgecolor="none")
            # 標記黃色區間
            ax.axvspan(20, 55, alpha=0.2, color="yellow", label="黃色區間 (H=20~55)")
            ax.legend(fontsize=8)
        else:
            ax.bar(centers, counts, width=4, color="steelblue", edgecolor="none")
        
        ax.set_title(title)
        ax.set_xlabel("值 (0–255)")
        ax.set_ylabel("像素數")
        ax.set_xlim(0, 255)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("✓ HSV 分布圖 → %s", output_path)


def plot_top_colors(pixels: np.ndarray, output_path: Path, top_n: int = 20) -> None:
    """把最常見的顏色量化後，畫成色塊圖。"""
    # 把 H 量化到 32 個桶、S/V 量化到 4 個桶，降維統計
    h_q = (pixels[:, 0] // 8).astype(int)   # 0~31
    s_q = (pixels[:, 1] // 64).astype(int)  # 0~3
    v_q = (pixels[:, 2] // 64).astype(int)  # 0~3

    keys = list(zip(h_q, s_q, v_q))
    counter = Counter(keys)
    top = counter.most_common(top_n)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_title(f"Top {top_n} Most Common Color Groups", fontsize=13, fontweight="bold")
    ax.set_xlim(0, top_n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, ((hq, sq, vq), cnt) in enumerate(top):
        h = int(hq * 8 + 4)
        s = int(sq * 64 + 32)
        v = int(vq * 64 + 32)
        from PIL import Image as PILImage
        px = PILImage.new("HSV", (1, 1), (h, s, v))
        r, g, b = px.convert("RGB").getpixel((0, 0))
        color = (r/255, g/255, b/255)
        rect = mpatches.FancyBboxPatch(
            (i + 0.05, 0.2), 0.85, 0.6,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="gray", linewidth=0.5
        )
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.12, f"H={h}", ha="center", fontsize=6, color="gray")
        pct = 100 * cnt / len(keys)
        ax.text(i + 0.5, 0.88, f"{pct:.1f}%", ha="center", fontsize=6, color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("✓ 主要顏色圖 → %s", output_path)


def write_summary(result: dict, output_path: Path, pixels: np.ndarray) -> None:
    """輸出建議的 filter_yellow.py 參數。"""
    total = len(pixels)
    yellow_ratio = result["yellow_pixel_ratio"]

    lines = [
        "=" * 50,
        "顏色分析結果摘要",
        "=" * 50,
        f"分析像素總數：{total:,}",
        f"黃色像素比例：{yellow_ratio:.1%}",
        f"推算信心度  ：{result['confidence']}",
        "",
        "建議的 filter_yellow.py 黃色 HSV 範圍：",
        f"  YELLOW_H_MIN = {result['h_min']}",
        f"  YELLOW_H_MAX = {result['h_max']}",
        f"  YELLOW_S_MIN = 60   （可視情況調整）",
        f"  YELLOW_V_MIN = 80   （可視情況調整）",
        "",
        "建議執行指令：",
        f"  python filter_yellow.py --input tiles_output --threshold 0.6",
        "",
        "若過濾結果不理想：",
        "  - candidate 太少 → --threshold 0.5（寬鬆）",
        "  - candidate 還有很多黃色 → --threshold 0.75（嚴格）",
        "=" * 50,
    ]

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print("\n" + text)
    log.info("✓ 摘要 → %s", output_path)

    # 同時印出是否需要更新 filter_yellow.py
    if result["h_min"] != 20 or result["h_max"] != 50:
        log.info(
            "💡 提示：推算的黃色範圍 H=%d~%d 與 filter_yellow.py 預設 (20~50) 不同，"
            "建議更新 filter_yellow.py 的 YELLOW_H_MIN / YELLOW_H_MAX",
            result["h_min"], result["h_max"]
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 tiles 的顏色分布，推算黃色範圍",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input",   default="test",  help="tile 資料夾")
    parser.add_argument("-o", "--output",  default="color_report",  help="報告輸出資料夾")
    parser.add_argument("-n", "--sample",  type=int, default=200,   help="隨機抽樣張數")
    parser.add_argument("--seed",          type=int, default=42,    help="隨機種子（可重現）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles = collect_tiles(Path(args.input), args.sample)
    log.info("擷取像素中...")
    pixels = extract_hsv_pixels(tiles)
    log.info("共擷取 %d 個像素樣本", len(pixels))

    log.info("繪製 HSV 分布圖...")
    plot_hsv_distribution(pixels, output_dir / "hsv_distribution.png")

    log.info("繪製主要顏色...")
    plot_top_colors(pixels, output_dir / "top_colors.png")

    result = suggest_yellow_range(pixels[:, 0])
    write_summary(result, output_dir / "summary.txt", pixels)

    log.info("✅ 分析完成！請查看 %s/", output_dir)