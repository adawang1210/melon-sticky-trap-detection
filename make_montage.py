# -*- coding: utf-8 -*-
"""為 final_result 下每個 sub_* 資料夾生成一張拼貼大圖。

每個 sub 取前 N 張照片，縮成統一縮圖後排成網格，照片之間留間隔。
輸出 final_result_montage/sub_X.jpg。
"""
import os
import math
from PIL import Image

SRC_ROOT = "final_result"
DST_ROOT = "final_result_montage"
PER_SUB = 200          # 每個 sub 最多取幾張
THUMB = 128            # 每張縮圖的邊長(px)
GAP = 2                # 照片之間的間隔(px)
COLS = 16              # 每列幾張
BG = (255, 255, 255)   # 背景色(白)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def make_thumb(path):
    """讀圖，等比放大到「填滿」THUMB x THUMB 後置中裁切，去除多餘空白。"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = THUMB / min(w, h)                       # 以較短邊為基準放大填滿方塊
    nw, nh = max(THUMB, round(w * scale)), max(THUMB, round(h * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left = (nw - THUMB) // 2
    top = (nh - THUMB) // 2
    return im.crop((left, top, left + THUMB, top + THUMB))


def build_montage(name, src_dir):
    imgs = sorted(
        f for f in os.listdir(src_dir)
        if f.lower().endswith(IMG_EXTS)
        and os.path.isfile(os.path.join(src_dir, f))
    )[:PER_SUB]
    if not imgs:
        print(f"{name}: 無照片，略過")
        return None

    n = len(imgs)
    cols = min(COLS, n)
    rows = math.ceil(n / cols)

    W = cols * THUMB + (cols + 1) * GAP
    H = rows * THUMB + (rows + 1) * GAP
    sheet = Image.new("RGB", (W, H), BG)

    for i, f in enumerate(imgs):
        r, c = divmod(i, cols)
        x = GAP + c * (THUMB + GAP)
        y = GAP + r * (THUMB + GAP)
        try:
            sheet.paste(make_thumb(os.path.join(src_dir, f)), (x, y))
        except Exception as e:
            print(f"  跳過 {f}: {e}")

    out = os.path.join(DST_ROOT, f"{name}.jpg")
    sheet.save(out, quality=90)
    print(f"{name}: {n} 張 -> {out}  ({W}x{H})")
    return out


def main():
    os.makedirs(DST_ROOT, exist_ok=True)
    count = 0
    for name in sorted(os.listdir(SRC_ROOT)):
        src_dir = os.path.join(SRC_ROOT, name)
        if not os.path.isdir(src_dir) or not name.startswith("sub_"):
            continue
        if build_montage(name, src_dir):
            count += 1
    print(f"\n完成，共生成 {count} 張拼貼圖到 {DST_ROOT}/")


if __name__ == "__main__":
    main()
