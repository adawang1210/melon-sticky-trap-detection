import os
import shutil

# 1. 設定路徑
# 這裡假設你的結果檔名是 result，請檢查實際名稱
result_file = 'result' 
src_dir = r'.\data\train\melon'
dst_dir = r'E:\洋香瓜\分群結果_Output'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

print(f"開始搬移圖片到: {dst_dir}...")

# 2. 讀取結果並搬移
try:
    with open(result_file, 'r') as f:
        for line in f:
            # 假設格式是: 檔名 群號 (例如: melon_001.jpg 4)
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            img_name = parts[0]
            cluster_id = parts[1]
            
            # 建立子資料夾 (Cluster_0, Cluster_1...)
            target_folder = os.path.join(dst_dir, f'Cluster_{cluster_id}')
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            # 執行複製 (用 copy 比較安全，原本的圖還會留在原位)
            src_path = os.path.join(src_dir, img_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, target_folder)
                
    print("✅ 搬移完成！快去 E:\\洋香瓜\\分群結果_Output 看看吧！")
except Exception as e:
    print(f"❌ 發生錯誤: {e}")
    print("請確認 'result' 檔案是否存在，或把檔案內容的前幾行貼給我看。")