import os
import argparse
from pathlib import Path
from collections import defaultdict

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def count_tiles(base_path: Path) -> dict[str, int]:
    """遞迴統計各子資料夾的圖片數量，回傳 {資料夾名稱: 數量}"""
    if not base_path.exists():
        raise FileNotFoundError(f"找不到資料夾：'{base_path}'")
    if not base_path.is_dir():
        raise NotADirectoryError(f"'{base_path}' 不是資料夾")

    counts: dict[str, int] = defaultdict(int)

    for item in base_path.rglob("*"):
        if item.is_file() and item.suffix.lower() in EXTENSIONS and item.parent != base_path:
            # 使用相對路徑作為 key，保留層級結構
            rel = item.parent.relative_to(base_path)
            counts[str(rel)] += 1

    return dict(sorted(counts.items()))


def print_report(counts: dict[str, int], base_path: Path) -> None:
    """格式化輸出統計報告"""
    col_w = max((len(k) for k in counts), default=20) + 2
    col_w = max(col_w, 20)
    sep = "-" * (col_w + 15)

    print(f"\n統計資料夾：{base_path.resolve()}")
    print(sep)
    print(f"{'資料夾路徑':<{col_w}} {'圖片張數':>8}")
    print(sep)

    for folder, count in counts.items():
        print(f"{folder:<{col_w}} {count:>8,}")

    print(sep)
    print(f"{'子資料夾總數':<{col_w}} {len(counts):>8,}")
    print(f"{'圖片總張數':<{col_w}} {sum(counts.values()):>8,}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="統計 tile 資料夾內的圖片數量")
    parser.add_argument(
        "path",
        nargs="?",
        default="tiles_output",
        help="目標資料夾路徑（預設：tiles_output）",
    )
    args = parser.parse_args()
    base_path = Path(args.path)

    try:
        counts = count_tiles(base_path)
        if not counts:
            print(f"在 '{base_path}' 的子資料夾中找不到任何圖片。")
        else:
            print_report(counts, base_path)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"錯誤：{e}")


if __name__ == "__main__":
    main()