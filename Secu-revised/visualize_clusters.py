import os
from PIL import Image, ImageDraw

# --- 1. 設定路徑 ---
base_dir = r'E:\洋香瓜\Secu-revised\data\test\cluster8'
output_visual_dir = r'E:\洋香瓜\Cluster_Visual大圖'

# --- 2. 視覺化參數設定 ---
IMAGES_PER_CLUSTER = 50  # 每個群組抓 50 張
GRID_COLS = 10           # 一行放 10 張
THUMB_SIZE = 200         # 小圖縮放大小
PADDING = 10             # 圖與圖之間的空白間隔 (像素)
BG_COLOR = (255, 255, 255) # 背景顏色 (白色)

os.makedirs(output_visual_dir, exist_ok=True)
print(f"🎬 開始處理留白版海報... 結果將存放在: {output_visual_dir}")

for cluster_id in range(8):
    cluster_folder_name = f'cluster_{cluster_id}'
    cluster_path = os.path.join(base_dir, cluster_folder_name)
    
    if not os.path.exists(cluster_path):
        continue
    
    print(f"🔍 正在處理 {cluster_folder_name} ...")
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_names = sorted([f for f in os.listdir(cluster_path) if f.lower().endswith(valid_extensions)])
    selected_images = image_names[:IMAGES_PER_CLUSTER]
    
    if not selected_images:
        continue

    # --- 3. 計算大圖尺寸 (包含間隔) ---
    # 寬度 = (小圖寬 + 間隔) * 列數 + 最後一個間隔
    # 高度 = (小圖高 + 間隔) * 行數 + 最後一個間隔
    grid_rows = IMAGES_PER_CLUSTER // GRID_COLS
    canvas_w = (THUMB_SIZE + PADDING) * GRID_COLS + PADDING
    canvas_h = (THUMB_SIZE + PADDING) * grid_rows + PADDING
    
    # 創建白底畫布
    montage = Image.new('RGB', (canvas_w, canvas_h), BG_COLOR)

    for idx, img_name in enumerate(selected_images):
        try:
            img_path = os.path.join(cluster_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((THUMB_SIZE, THUMB_SIZE))
            
            # 畫上小小的綠色檔名標註 (放在圖片左上角)
            draw = ImageDraw.Draw(img)
            draw.text((2, 2), img_name[-8:], fill=(0, 255, 0))
            
            # --- 4. 計算這張小圖在大圖上的精確座標 ---
            # x = (目前第幾列 * (圖寬+間隔)) + 初始邊界間隔
            x = (idx % GRID_COLS) * (THUMB_SIZE + PADDING) + PADDING
            y = (idx // GRID_COLS) * (THUMB_SIZE + PADDING) + PADDING
            
            montage.paste(img, (x, y))
        except Exception as e:
            print(f"無法讀取圖片 {img_name}: {e}")

    # --- 5. 儲存結果 ---
    output_path = os.path.join(output_visual_dir, f'Spaced_Montage_Cluster_{cluster_id}.jpg')
    montage.save(output_path, quality=95)
    print(f"   ✅ 已生成帶間隔海報: Spaced_Montage_Cluster_{cluster_id}.jpg")

print(f"\n✨ 全部完成！您可以直接去查看了：\n👉 {output_visual_dir}")