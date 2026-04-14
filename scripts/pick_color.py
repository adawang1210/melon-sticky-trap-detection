"""
pick_color.py — 點選圖片上的像素，印出該點的 RGB / HSV 值

用法：
  python pick_color.py                                    # 從 filtered_output/candidate 隨機挑一張
  python pick_color.py --image path/to/tile.jpg           # 指定圖片
  python pick_color.py --input filtered_output/candidate  # 從資料夾隨機挑一張

操作方式：
  - 左鍵點選：印出該點顏色
  - 右鍵點選：印出該點顏色並加入「排除清單」
  - 按 N：換下一張圖
  - 按 Q 或 ESC：結束，印出建議的 HSV 範圍
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
picked_colors = []  # 儲存點選的 HSV 值


def pil_to_hsv_array(img: Image.Image) -> np.ndarray:
    """回傳 HSV numpy array，shape=(H, W, 3)，值域 0-255。"""
    return np.array(img.convert("HSV"), dtype=np.uint8)


def hsv_to_color_name(h: int) -> str:
    """把 PIL HSV 的 H 值（0-255）轉成顏色名稱。"""
    h360 = h * 360 / 255
    if h360 < 15 or h360 > 345:
        return "紅色"
    elif h360 < 45:
        return "橘黃色"
    elif h360 < 75:
        return "黃色"
    elif h360 < 150:
        return "綠色"
    elif h360 < 200:
        return "青色"
    elif h360 < 260:
        return "藍色"
    elif h360 < 290:
        return "紫色"
    else:
        return "粉紅/洋紅"


def mouse_callback(event, x, y, flags, param):
    img_rgb = param["img_rgb"]
    hsv_arr = param["hsv_arr"]
    display = param["display"]
    scale   = param["scale"]

    if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        # 換算回原始座標
        ox = int(x / scale)
        oy = int(y / scale)
        ox = min(ox, img_rgb.shape[1] - 1)
        oy = min(oy, img_rgb.shape[0] - 1)

        r, g, b = img_rgb[oy, ox]
        h, s, v = hsv_arr[oy, ox]
        color_name = hsv_to_color_name(int(h))

        tag = "【加入排除清單】" if event == cv2.EVENT_RBUTTONDOWN else ""
        print(f"  座標=({ox},{oy})  RGB=({r},{g},{b})  HSV=({h},{s},{v})  → {color_name} {tag}")

        if event == cv2.EVENT_RBUTTONDOWN:
            picked_colors.append({"h": int(h), "s": int(s), "v": int(v),
                                   "r": int(r), "g": int(g), "b": int(b)})

        # 在預覽圖上畫一個十字
        cv2.drawMarker(display, (x, y), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.imshow("pick_color — 左鍵查詢 / 右鍵排除 / N換圖 / Q結束", display)


def show_image(img_path: Path, scale: float = 4.0) -> str:
    """顯示圖片並等待操作，回傳 'next' 或 'quit'。"""
    with Image.open(img_path) as pil_img:
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        hsv_arr = pil_to_hsv_array(pil_img)
        img_rgb = np.array(pil_img)

    # 放大顯示（tile 通常很小，放大才看得清楚）
    disp_w = max(int(w * scale), 400)
    disp_h = max(int(h * scale), 400)
    display = cv2.resize(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        (disp_w, disp_h), interpolation=cv2.INTER_NEAREST
    ).copy()

    win_name = "pick_color — 左鍵查詢 / 右鍵排除 / N換圖 / Q結束"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)

    param = {
        "img_rgb": img_rgb,
        "hsv_arr": hsv_arr,
        "display": display,
        "scale": disp_w / w,
    }
    cv2.setMouseCallback(win_name, mouse_callback, param)
    cv2.imshow(win_name, display)
    print(f"\n📷 {img_path.name}  ({w}×{h}px)")
    print("  左鍵=查詢顏色  右鍵=加入排除清單  N=下一張  Q/ESC=結束")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return "quit"
        elif key == ord("n"):
            cv2.destroyAllWindows()
            return "next"


def summarize(picked: list) -> None:
    """根據點選的顏色，建議 filter_yellow.py 的排除範圍。"""
    if not picked:
        print("\n（沒有右鍵點選任何顏色，無法產生建議）")
        return

    hs = [p["h"] for p in picked]
    ss = [p["s"] for p in picked]
    vs = [p["v"] for p in picked]

    print("\n" + "=" * 50)
    print("右鍵點選的顏色統計：")
    print(f"  H 範圍：{min(hs)} ~ {max(hs)}")
    print(f"  S 範圍：{min(ss)} ~ {max(ss)}")
    print(f"  V 範圍：{min(vs)} ~ {max(vs)}")
    print()
    print("建議在 filter_yellow.py 加入以下排除條件：")
    h_min = max(0,   min(hs) - 5)
    h_max = min(255, max(hs) + 5)
    s_max = min(255, max(ss) + 10)
    v_min = max(0,   min(vs) - 10)
    print(f"  # 灰白/反光區域")
    print(f"  EXCLUDE_H_MIN = {h_min}")
    print(f"  EXCLUDE_H_MAX = {h_max}")
    print(f"  EXCLUDE_S_MAX = {s_max}   # 飽和度低於此值")
    print(f"  EXCLUDE_V_MIN = {v_min}   # 亮度高於此值")
    print("=" * 50)


def collect_tiles(input_dir: Path, n: int) -> list:
    all_tiles = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )
    if not all_tiles:
        raise FileNotFoundError(f"在 '{input_dir}' 找不到圖片")
    return random.sample(all_tiles, min(n, len(all_tiles)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="點選圖片像素，取得 HSV 值，協助設定過濾參數",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=str, default=None,     help="指定單張圖片路徑")
    group.add_argument("--input", type=str,
                       default="filtered_output/candidate",   help="從資料夾隨機挑圖")
    parser.add_argument("--sample", type=int, default=20,     help="最多隨機取幾張")
    parser.add_argument("--scale",  type=float, default=6.0,  help="圖片放大倍數（預設 6x）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.image:
        tiles = [Path(args.image)]
    else:
        tiles = collect_tiles(Path(args.input), args.sample)
        random.shuffle(tiles)

    print(f"共 {len(tiles)} 張圖，右鍵點選你想排除的顏色區域")

    for tile_path in tiles:
        result = show_image(tile_path, scale=args.scale)
        if result == "quit":
            break

    summarize(picked_colors)