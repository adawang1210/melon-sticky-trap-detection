# 技術堆疊

## 程式語言
- Python 3.8+

## 核心 ML 函式庫
- PyTorch >= 1.6（訓練、推論、分散式）
- torchvision（影像轉換、載入）
- timm（ViT 骨幹網路，使用 `vit_base_patch16_224`）
- scikit-learn（NMI、ARI 指標、t-SNE、LabelEncoder）
- scipy（匈牙利演算法計算聚類準確度）
- munkres（替代匈牙利演算法求解器）

## 影像處理
- Pillow / PIL（影像讀寫、轉換、HSV 色彩空間）
- NumPy（陣列運算、`.npy` 檔案支援）
- OpenCV / `cv2`（Grad-CAM 熱力圖疊加）
- matplotlib（t-SNE 圖表、色彩分析圖表）

## 資料與日誌
- pandas（聚類分佈 CSV 報告）
- TensorBoard（`torch.utils.tensorboard.SummaryWriter`）

## 訓練基礎設施
- PyTorch DistributedDataParallel (DDP)，使用 `gloo` 後端
- 混合精度訓練：`torch.cuda.amp`（autocast + GradScaler）
- LARS 優化器（自訂，位於 `secu/optimizer.py`）
- ResNet 使用 SGD，ViT 使用 AdamW

## 建置 / 執行
沒有正式的建置系統、套件管理器或 `requirements.txt`。所有腳本直接用 `python` 執行。

### 常用指令

訓練（從 `Secu-revised/` 目錄執行）：
```bash
# ResNet 骨幹（CIFAR 風格）
python main.py ./data/train -j 8 -p 10 --lr 0.01 --epochs 101 \
  --secu-num-ins <N> --secu-alpha <6*N/50> --secu-k 3 4 5 \
  --clr 0.001 --min-crop 0.2 --log secu-medoid \
  --dist-url tcp://localhost:1234 --multiprocessing-distributed \
  --world-size 1 --rank 0 --secu-tx 0.07 --use-medoid 1

# ViT 骨幹
python main_org.py ./data -j 4 -p 10 --lr 0.01 --epochs 201 \
  --secu-num-ins <N> --secu-alpha <6*N/50> --secu-k 8 9 10 \
  --clr 0.001 --min-crop 0.2 --log secu \
  --dist-url tcp://localhost:1234 --multiprocessing-distributed \
  --world-size 1 --rank 0 --secu-tx 0.07 --use-medoid 1 \
  --secu-lratio 0.7 --warm-up 30 -b 64 --backbone vit --secu-cst size-mml
```

推論（從 `Secu-revised/` 目錄執行）：
```bash
python inference_new_gt_txt.py --model-path model/best_model.pth.tar \
  --secu-num-ins <N> --secu-alpha <alpha> --secu-k 8 9 10 \
  --secu-tx 0.07 --data-name custom --backbone vit
```

影像前處理（從 `scripts/` 目錄執行）：
```bash
python crop_border.py -i <input> -o cropped_border --all 100
python tile_images.py -i cropped_border -o tiles_output --preset medium
python filter_yellow.py -i tiles_output -o filtered_output -t 0.6
python adaptive_tile.py -i cropped_border -o adaptive_output
```

### 關鍵參數規則
- `--secu-num-ins` 必須等於資料集總數 N
- `--secu-alpha` 通常設為 `6 * N / 50`（不可超過每類的樣本數）
- `--secu-k` 定義多頭聚類的類別數量（例如 `8 9 10`）
- `config.py` 中的 `clusters_amount` 在推論時必須是 `--secu-k` 其中一個值
