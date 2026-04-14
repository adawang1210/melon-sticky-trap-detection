import os
from pathlib import Path
import shutil
import random
from collections import Counter
def process_npy_files(root_path, target_path, sample_size=750):
    # 建立目標資料夾（如果不存在）
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"已建立目標資料夾: {target_path}")

    all_files = []
    categories = []

    # 1. 遍歷資料夾讀取檔案
    print("正在掃描檔案...")
    for subdir, _, files in os.walk(root_path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(subdir, file)
                
                # 根據檔名 split("_")[4] 提取類別
                try:
                    category = file.split("_")[4]
                    all_files.append((file_path, file, category))
                    categories.append(category)
                except IndexError:
                    print(f"警告: 檔案 {file} 格式不符，無法提取類別。")

    # 2. 統計類別數量
    stats = Counter(categories)
    print("\n--- 類別統計結果 ---")
    for cat, count in stats.items():
        print(f"類別: {cat}, 原始數量: {count}")

    # 3. 按類別選取檔案並複製
    print(f"\n正在提取檔案（每個類別上限 {sample_size} 個）...")
    
    # 將檔案按類別分組
    cat_to_files = {}
    for path, name, cat in all_files:
        if cat not in cat_to_files:
            cat_to_files[cat] = []
        cat_to_files[cat].append((path, name))

    total_copied = 0
    for cat, files_list in cat_to_files.items():
        # 如果檔案數少於 sample_size，就全部取走；否則隨機抽樣
        num_to_sample = min(len(files_list), sample_size)
        sampled_files = random.sample(files_list, num_to_sample)

        for src_path, file_name in sampled_files:
            # 為了避免不同子資料夾有同名檔案衝突，可以在新檔名前加上類別前綴或保持原樣
            # 這裡直接使用原檔名，若有重複風險可改為 f"{cat}_{file_name}"
            dest_path = os.path.join(target_path, file_name)
            shutil.copy2(src_path, dest_path)
            total_copied += 1
            
        print(f"  - {cat}: 已複製 {num_to_sample} 個檔案")

    print(f"\n完成！總共複製了 {total_copied} 個檔案到 {target_path}")
    print("類別列表:", list(stats.keys()))

def generate_label_txt(data_root, output_txt, split_index=0):
    """
    Args:
        data_root (str): 存放 .npy 檔案的根目錄
        output_txt (str): 輸出的 .txt 檔名
        split_index (int): 檔名用 '_' 切割後，要取第幾個位置當作類別
    """
    root_path = Path(data_root)
    
    # 搜尋所有 .npy 檔案 (rglob 代表遞迴搜尋，包含子資料夾)
    npy_files = list(root_path.rglob('*.jpg'))
    
    print(f"在 {data_root} 中找到 {len(npy_files)} 個 npy 檔案。")
    print(f"正在提取檔名 split('_')[{split_index}] 作為類別...")

    success_count = 0
    fail_count = 0

    with open(output_txt, 'w', encoding='utf-8') as f:
        for file_path in npy_files:
            # 1. 取得檔名 (不含路徑也不含副檔名)
            # 例如: /data/train/corn_seedling_001.npy -> corn_seedling_001
            filename_no_ext = file_path.stem
            
            # 2. 進行切割
            parts = filename_no_ext.split('_')
            
            # 3. 取出指定位置的類別
            try:
                label_name = parts[split_index]
                
                # 4. 寫入 txt (格式: 絕對路徑 類別名稱)
                # 你可以把 file_path.resolve() 改成 str(file_path) 來保留相對路徑
                line = f"{label_name}\n"
                f.write(line)
                success_count += 1
                
            except IndexError:
                print(f"[警告] 檔案 {filename_no_ext} 無法以 '_' 切割出第 {split_index} 個位置，已跳過。")
                fail_count += 1

    print("-" * 30)
    print(f"處理完成！")
    print(f"成功寫入: {success_count} 筆")
    print(f"失敗跳過: {fail_count} 筆")
    print(f"結果已儲存至: {output_txt}")

def generate_label_pre_root_txt(data_root, output_txt):
    """
    Args:
        data_root (str): 存放圖片檔案的根目錄
        output_txt (str): 輸出的 .txt 檔名
    """
    root_path = Path(data_root)
    
    # 搜尋所有 .jpg 檔案 (rglob 代表遞迴搜尋)
    # 如果你的檔案是 .npy，請將下行改回 '*.npy'
    image_files = list(root_path.rglob('*.jpg'))
    
    print(f"在 {data_root} 中找到 {len(image_files)} 個檔案。")
    print(f"正在提取「資料夾名稱」作為類別...")

    success_count = 0

    with open(output_txt, 'w', encoding='utf-8') as f:
        for file_path in image_files:
            # --- 關鍵修改處 ---
            # file_path.parent 取得檔案所在的目錄路徑
            # .name 則取得該目錄的最末端資料夾名稱
            label_name = file_path.parent.name
            
            # 取得檔案路徑 (這裡可以根據需求存 絕對路徑 或 檔名)
            # 下方範例格式為：資料夾標籤 檔案路徑
            line = f"{label_name}\n"
            
            f.write(line)
            success_count += 1

    print("-" * 30)
    print(f"處理完成！")
    print(f"成功寫入: {success_count} 筆")
    print(f"結果已儲存至: {output_txt}")
# ==========================================
# 參數設定區
# ==========================================



if __name__ == "__main__":
    # --- 設定路徑 ---
    # 1. 資料夾路徑
    my_data_root = r"C:\Users\MOA\Desktop\Automatic Clustering tool\Secu-revised\data\train\Soyabean_Rust" 

    # 2. 輸出檔案名稱
    my_output_txt = r"C:\Users\MOA\Desktop\Automatic Clustering tool\Secu-revised\data\train\Soyabean_Rust\GT_label.txt"

    # 3. 指定要取檔名的第幾個部分 (從 0 開始算)
    # 範例檔名: "corn_healthy_001.npy"
    # index 0 -> "corn"
    # index 1 -> "healthy"
    # index 2 -> "001"
    target_index = 4  
    if os.path.exists(my_data_root):
        #process_npy_files(my_data_root,os.path.join(my_data_root,'new'))
        generate_label_pre_root_txt(my_data_root, my_output_txt)
    else:
        print(f"錯誤: 找不到路徑 {my_data_root}")