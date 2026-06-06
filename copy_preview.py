# -*- coding: utf-8 -*-
"""將 final_result 下每個 sub_* 資料夾的前 N 張照片複製到 final_result_preview。"""
import os
import shutil

SRC_ROOT = "final_result"
DST_ROOT = "final_result_preview"
PER_SUB = 200
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    os.makedirs(DST_ROOT, exist_ok=True)
    total = 0
    for name in sorted(os.listdir(SRC_ROOT)):
        src_dir = os.path.join(SRC_ROOT, name)
        # 只處理真正的資料夾，排除 .lnk 捷徑與檔案
        if not os.path.isdir(src_dir) or not name.startswith("sub_"):
            continue

        imgs = sorted(
            f for f in os.listdir(src_dir)
            if f.lower().endswith(IMG_EXTS)
            and os.path.isfile(os.path.join(src_dir, f))
        )
        picked = imgs[:PER_SUB]

        dst_dir = os.path.join(DST_ROOT, name)
        os.makedirs(dst_dir, exist_ok=True)
        for f in picked:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

        total += len(picked)
        print(f"{name}: 複製 {len(picked)} / {len(imgs)} 張")

    print(f"\n完成，共複製 {total} 張到 {DST_ROOT}/")


if __name__ == "__main__":
    main()
