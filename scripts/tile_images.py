"""
tile_images.py — 將大圖切割成固定大小的 tiles（支援 overlap）

用法：
  python tile_images.py                              # 預設 8px tiles，overlap 自動
  python tile_images.py --tile-size 16               # 16px tiles
  python tile_images.py --tile-size 32 --overlap 4   # 手動設定 overlap
  python tile_images.py --tile-size 64 --overlap 0   # 無重疊
  python tile_images.py --multi-size 8,16,32         # 同時輸出多種尺寸
  python tile_images.py --preset tiny                # 8px / overlap=0
  python tile_images.py --preset xsmall              # 16px / overlap=2
  python tile_images.py --preset small               # 32px / overlap=4
  python tile_images.py --preset medium              # 64px / overlap=8
  python tile_images.py --min-content 0.05           # 略過近全黑/白的 tile
  python tile_images.py --dry-run                    # 預覽切割結果（不儲存）
  python tile_images.py --preview-grid               # 輸出切割格線預覽圖

overlap 預設行為（不指定 --overlap 時）：
  自動依 tile-size / 8 計算，例如 8px→1, 16px→2, 32px→4, 64px→8

預設組合（--preset）：
  tiny=8px  xsmall=16px  small=32px  medium=64px  large=128px  xlarge=256px
"""

import os
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# 預設組合表
PRESETS = {
    "tiny":   {"tile_size": 8,   "overlap": 0},
    "xsmall": {"tile_size": 16,  "overlap": 2},
    "small":  {"tile_size": 32,  "overlap": 4},
    "medium": {"tile_size": 64,  "overlap": 8},
    "large":  {"tile_size": 128, "overlap": 16},
    "xlarge": {"tile_size": 256, "overlap": 32},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────

@dataclass
class Config:
    input_folder: Path
    output_base: Path
    tile_size: int = 8
    overlap: int = 0
    quality: int = 95
    output_ext: str = ".jpg"
    workers: int = os.cpu_count() or 4
    min_content_ratio: float = 0.0
    overwrite: bool = False
    dry_run: bool = False
    preview_grid: bool = False
    multi_sizes: list = field(default_factory=list)   # 多尺寸同時切割

    @property
    def step(self) -> int:
        s = self.tile_size - self.overlap
        if s <= 0:
            raise ValueError(f"overlap ({self.overlap}) 必須小於 tile_size ({self.tile_size})")
        return s


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def _is_blank(tile: Image.Image, min_ratio: float) -> bool:
    """若 tile 的非空白像素比例低於閾值，視為空白 tile。"""
    if min_ratio <= 0:
        return False
    arr = np.array(tile.convert("L"), dtype=np.uint8)
    non_blank = np.count_nonzero((arr > 10) & (arr < 245))
    return (non_blank / arr.size) < min_ratio


def _estimate_tile_count(w: int, h: int, cfg: Config) -> int:
    """預估切割後的 tile 數量（不含過濾）。"""
    cols = len(range(0, w, cfg.step))
    rows = len(range(0, h, cfg.step))
    return cols * rows


def _draw_preview_grid(img: Image.Image, cfg: Config, save_path: Path) -> None:
    """在原圖上繪製切割格線，輸出預覽圖（縮小至最大 2048px 以利預覽）。"""
    preview = img.copy()
    MAX_PREVIEW = 2048
    scale = min(MAX_PREVIEW / max(img.width, img.height), 1.0)
    if scale < 1.0:
        pw, ph = int(img.width * scale), int(img.height * scale)
        preview = preview.resize((pw, ph), Image.LANCZOS)
    else:
        scale = 1.0

    draw = ImageDraw.Draw(preview)
    w, h = img.size
    color = (255, 80, 80, 180)

    for y in range(0, h, cfg.step):
        draw.line([(0, int(y * scale)), (int(w * scale), int(y * scale))],
                  fill=color, width=1)
    for x in range(0, w, cfg.step):
        draw.line([(int(x * scale), 0), (int(x * scale), int(h * scale))],
                  fill=color, width=1)

    preview.save(save_path)
    log.info("📐 格線預覽圖 → %s", save_path)


# ──────────────────────────────────────────────
# 核心邏輯
# ──────────────────────────────────────────────

def tile_single_image(img_path: Path, cfg: "Config") -> tuple:
    """
    處理單張圖片，回傳 (filename, tile_count, error_message|None)。
    設計為可在子 process 執行。
    """
    filename = img_path.name

    try:
        with Image.open(img_path) as img:
            img.load()
            w, h = img.size

            # ── dry-run / preview 模式 ──
            if cfg.dry_run or cfg.preview_grid:
                count = _estimate_tile_count(w, h, cfg)
                log.info("🔍 [dry-run] %s  [%dx%d]  → 預計 %d tiles  (step=%dpx)",
                         filename, w, h, count, cfg.step)
                if cfg.preview_grid:
                    preview_dir = cfg.output_base / "previews"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    _draw_preview_grid(img, cfg, preview_dir / f"{img_path.stem}_grid.jpg")
                return filename, count, None

            # ── 正式切割 ──
            save_dir = cfg.output_base / img_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)

            positions = [
                (x, y)
                for y in range(0, h, cfg.step)
                for x in range(0, w, cfg.step)
            ]
            total_possible = len(positions)
            pad = len(str(total_possible))

            count = 0
            skipped = 0
            seq = 1

            for x, y in positions:
                box = (x, y, min(x + cfg.tile_size, w), min(y + cfg.tile_size, h))
                tile = img.crop(box)

                if _is_blank(tile, cfg.min_content_ratio):
                    skipped += 1
                    continue

                out_path = save_dir / f"tile_{seq:0{pad}d}{cfg.output_ext}"

                if cfg.overwrite or not out_path.exists():
                    save_kwargs = {}
                    if cfg.output_ext in {".jpg", ".jpeg"}:
                        save_kwargs["quality"] = cfg.quality
                        save_kwargs["subsampling"] = 0
                    tile.save(out_path, **save_kwargs)

                seq += 1
                count += 1

        extra = f"（略過空白 {skipped} 張）" if skipped else ""
        log.info("✓ %s  [%dx%d]  → %d tiles%s", filename, w, h, count,
                 f"  {extra}" if extra else "")
        return filename, count, None

    except Exception as exc:  # noqa: BLE001
        import traceback
        msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log.error("✗ %s：%s", filename, msg)
        return filename, 0, msg


# ──────────────────────────────────────────────
# 多尺寸批次切割
# ──────────────────────────────────────────────

def _run_for_size(images: list, cfg: "Config") -> tuple:
    """針對單一尺寸執行切割，回傳 (total_tiles, errors)。"""
    total_tiles = 0
    errors: list = []

    # dry-run 強制單執行緒，避免 Windows multiprocessing 中文路徑問題
    effective_workers = 1 if cfg.dry_run else cfg.workers

    if effective_workers == 1:
        for img_path in images:
            _, count, err = tile_single_image(img_path, cfg)
            total_tiles += count
            if err:
                errors.append((img_path.name, err))
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(tile_single_image, p, cfg): p for p in images}
            for fut in as_completed(futures):
                _, count, err = fut.result()
                total_tiles += count
                if err:
                    errors.append((futures[fut].name, err))

    return total_tiles, errors


# ──────────────────────────────────────────────
# 批次處理入口
# ──────────────────────────────────────────────

def process_all(cfg: Config) -> None:
    if not cfg.input_folder.exists():
        raise FileNotFoundError(f"找不到輸入資料夾：'{cfg.input_folder}'")

    cfg.output_base.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in cfg.input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )

    if not images:
        log.warning("在 '%s' 中找不到任何支援的圖片。", cfg.input_folder)
        return

    # ── 多尺寸模式 ──
    if cfg.multi_sizes:
        sizes_to_run = cfg.multi_sizes
        log.info("🔀 多尺寸模式：%s", sizes_to_run)
    else:
        sizes_to_run = [cfg.tile_size]

    t0 = time.perf_counter()
    grand_total = 0
    all_errors: list = []

    for size in sizes_to_run:
        # 自動計算對應 overlap（若使用多尺寸，overlap 等比縮放）
        if cfg.multi_sizes:
            auto_overlap = max(0, round(size * cfg.overlap / cfg.tile_size))
            sub_cfg = Config(
                input_folder=cfg.input_folder,
                output_base=cfg.output_base / f"size_{size}",
                tile_size=size,
                overlap=auto_overlap,
                quality=cfg.quality,
                output_ext=cfg.output_ext,
                workers=cfg.workers,
                min_content_ratio=cfg.min_content_ratio,
                overwrite=cfg.overwrite,
                dry_run=cfg.dry_run,
                preview_grid=cfg.preview_grid,
            )
            log.info("── 尺寸 %dpx（overlap=%dpx） ──", size, auto_overlap)
        else:
            sub_cfg = cfg

        count, errors = _run_for_size(images, sub_cfg)
        grand_total += count
        all_errors.extend(errors)

    elapsed = time.perf_counter() - t0
    log.info("─" * 50)
    if cfg.dry_run:
        log.info("✅ [dry-run] 預覽完成！耗時 %.1f 秒", elapsed)
    else:
        log.info("✅ 完成！耗時 %.1f 秒", elapsed)
    log.info("成功處理圖片：%d / %d", len(images) - len({e[0] for e in all_errors}), len(images))
    log.info("產生 tile 總數：%d", grand_total)
    if all_errors:
        log.warning("失敗 %d 筆：", len(all_errors))
        for name, msg in all_errors:
            log.warning("  • %s：%s", name, msg)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="將資料夾內的大圖切割成 tiles（支援多種小尺寸）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 基本路徑
    parser.add_argument("-i", "--input",     default="../cropped_border",  help="輸入資料夾")
    parser.add_argument("-o", "--output",    default=str(Path.cwd() / "tiles_output"), help="輸出資料夾")

    # 尺寸設定（三選一：preset / multi-size / 手動）
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default=None,
        help="快速預設：tiny(8px) / xsmall(16px) / small(32px) / medium(64px) / large(128px) / xlarge(256px)"
    )
    parser.add_argument(
        "--multi-size", metavar="SIZES", default=None,
        help="同時輸出多種尺寸，以逗號分隔，例如 64,128,256"
    )
    parser.add_argument(
        "--tile-size", type=int, default=None,
        help="Tile 邊長（px）；建議值：8 / 16 / 32 / 64 / 128 / 256"
    )

    # 切割參數
    parser.add_argument("--overlap",     type=int,   default=None,
        help="重疊像素數（預設：自動依 tile-size 等比縮放，約 tile-size / 8）")
    parser.add_argument("--quality",     type=int,   default=95,   help="JPEG 品質（1–95）")
    parser.add_argument("--output-ext",  default=".jpg",           help="輸出格式 (.jpg / .png)")
    parser.add_argument("--workers", type=int, default=1, help="平行 worker 數（預設 1，大圖建議保持 1 避免 MemoryError）")
    parser.add_argument("--min-content", type=float, default=0.0,  help="最低非空白像素比例（0=不過濾）")

    # 行為旗標
    parser.add_argument("--overwrite",    action="store_true", help="覆蓋已存在的 tile")
    parser.add_argument("--dry-run",      action="store_true", help="僅預覽切割數量，不實際儲存")
    parser.add_argument("--preview-grid", action="store_true", help="輸出格線預覽圖（可與 --dry-run 合併）")

    args = parser.parse_args()

    # 三選一互斥檢查
    size_opts = [args.preset, args.multi_size, args.tile_size]
    if sum(v is not None for v in size_opts) > 1:
        parser.error("--preset / --multi-size / --tile-size 只能擇一使用")

    # 解析 preset
    tile_size = args.tile_size if args.tile_size is not None else 8
    if args.preset:
        p = PRESETS[args.preset]
        tile_size = p["tile_size"]
        preset_overlap = p["overlap"]
    else:
        preset_overlap = None

    # 自動計算 overlap：若未指定，依 tile_size / 8 等比縮放（最小為 0）
    if args.overlap is not None:
        overlap = args.overlap
    elif preset_overlap is not None:
        overlap = preset_overlap
    else:
        overlap = max(0, tile_size // 8)

    if args.preset:
        log.info("使用預設組合 [%s]：tile=%dpx, overlap=%dpx", args.preset, tile_size, overlap)
    else:
        log.info("tile=%dpx, overlap=%dpx（自動）", tile_size, overlap)

    # 解析 multi-size
    multi_sizes: list = []
    if args.multi_size:
        try:
            multi_sizes = [int(s.strip()) for s in args.multi_size.split(",")]
        except ValueError:
            parser.error("--multi-size 格式錯誤，請輸入以逗號分隔的整數，例如：64,128,256")

    return Config(
        input_folder=Path(args.input),
        output_base=Path(args.output),
        tile_size=tile_size,
        overlap=overlap,
        quality=args.quality,
        output_ext=args.output_ext,
        workers=args.workers,
        min_content_ratio=args.min_content,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        preview_grid=args.preview_grid,
        multi_sizes=multi_sizes,
    )


if __name__ == "__main__":
    process_all(parse_args())