# 🪴 Melon Sticky Trap Detection

基於無監督深度聚類的黃色黏蟲板昆蟲自動分群系統，應用於洋香瓜溫室害蟲監測。

---

## 功能特色

- **無監督聚類**：採用 SeCu（Stable Cluster Discrimination, ICCV 2023）演算法，無需人工標註即可將昆蟲影像自動分群
- **Medoid 中心重估**：改良原始 SeCu，以 top-k cosine similarity medoid 取代隨機初始化中心，提升聚類穩定性
- **Graph Modularity Loss (MML)**：額外引入圖模組化損失，強化群內相似度與群間分離度
- **雙骨幹支援**：同時支援 ResNet-18 與 ViT (vit_base_patch16_224) 作為特徵提取器
- **完整前處理流程**：從原始黏蟲板照片到可訓練的圖塊資料，提供角落遮蔽、邊框裁切、圖塊切割、自適應裁切等工具
- **聚類評估與視覺化**：輸出 t-SNE 3D 視覺化與聚類分佈 CSV 報告；若提供 Ground Truth 標籤可額外計算 ACC / NMI / ARI
- **混合精度訓練**：使用 PyTorch AMP (autocast + GradScaler) 加速訓練
- **TensorBoard 監控**：即時追蹤 loss、各群大小等訓練指標

---

## 技術架構

| 類別 | 技術 |
|------|------|
| 語言 | Python 3.8+ |
| 深度學習框架 | PyTorch >= 1.6、torchvision、timm |
| 分散式訓練 | DistributedDataParallel (DDP)、gloo backend |
| 優化器 | SGD (ResNet) / AdamW (ViT) / LARS (自訂) |
| 評估指標 | scikit-learn (NMI, ARI)、scipy (Hungarian ACC) |
| 影像處理 | Pillow、NumPy、OpenCV |
| 視覺化 | matplotlib (t-SNE)、TensorBoard |
| 資料格式 | `.jpg` / `.png` / `.npy` (NumPy array) |

---

## 操作流程

整體流程分為三個階段：**影像前處理** → **模型訓練** → **推論與分群**。

> **注意**：本系統為無監督聚類，輸入資料不需要標籤。訓練與推論使用同一批資料 — 訓練階段學習特徵表示與群中心，推論階段將每張影像分配到對應的 cluster。

### 階段一：影像前處理

所有前處理腳本位於 `scripts/` 目錄，從 `scripts/` 目錄執行：

```bash
# 1. 遮蔽角落無關區域（如相機時間戳）
python crop_corners.py -i ../Bugdatasets -o ../masked_output --size 3000

# 2. 裁切圖片四邊邊框
python crop_border.py -i ../masked_output -o ../cropped_border --all 100

# 3. 將大圖切割成小圖塊（二擇一）
# 方法 A：固定網格切割
python tile_images.py -i ../cropped_border -o ../tiles_output --preset medium

# 方法 B：自適應切割（自動偵測非黃色區域）
python adaptive_tile.py -i ../cropped_border -o ../adaptive_output --probe 32 --padding 20
```

### 階段二：模型訓練

將前處理後的影像放入 `Secu-revised/data/train/` 目錄下。由於是無監督學習，資料夾名稱不代表真實標籤，僅作為分組用途（可以只放一個子資料夾，例如 `data/train/all/`）。從 `Secu-revised/` 目錄執行：

```bash
python main.py ./data -j 4 -p 10 --lr 0.01 --epochs 201 \
  --secu-num-ins <N> \
  --secu-alpha <ALPHA> \
  --secu-k 8 9 10 \
  --clr 0.001 --min-crop 0.2 \
  --log secu \
  --dist-url tcp://localhost:1234 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --secu-tx 0.07 --use-medoid 1 \
  --secu-lratio 0.7 --warm-up 30 \
  -b 64 --backbone vit --secu-cst size-mml
```

**關鍵參數說明：**

| 參數 | 說明 | 設定規則 |
|------|------|----------|
| `--secu-num-ins` | 資料集總數 N | 必須等於訓練影像總張數 |
| `--secu-alpha` | 約束權重 | 通常設為 `6 × N / 50`（不超過每類樣本數） |
| `--secu-k` | 多頭聚類數量 | 設定 3 個值，如 `8 9 10`（類別數、+1、+2） |
| `--backbone` | 骨幹網路 | `resnet18` 或 `vit` |
| `--secu-cst` | 約束類型 | `size`、`entropy` 或 `size-mml` |
| `--use-medoid` | 啟用 medoid 中心重估 | `1` 啟用、`0` 關閉 |
| `--warm-up` | Warm-up epoch 數 | medoid 在此 epoch 後才啟動 |

模型檢查點儲存於 `model/` 目錄：
- 每 50 個 epoch 儲存一次：`model/<log>_<epoch>.pth.tar`
- 最佳模型（最低 loss）：`model/best_model.pth.tar`

### 階段三：推論與分群

1. 修改 `config.py` 中的 `clusters_amount`（必須是訓練時 `--secu-k` 的其中一個值）
2. 確認資料位於 `data/train/` 目錄（推論使用與訓練相同的資料）
3. 執行推論：

```bash
python inference.py \
  --model-path model/best_model.pth.tar \
  --secu-num-ins <N> \
  --secu-alpha <ALPHA> \
  --secu-k 8 9 10 \
  --secu-tx 0.07 \
  --data-name custom \
  --backbone vit
```

推論輸出：
- 分群後的影像（依 cluster 分資料夾存放）
- 聚類分佈 CSV 報告
- t-SNE 3D 視覺化圖表
- 若資料夾名稱對應真實類別，會額外計算 ACC / ARI / NMI 指標

---

## 輔助工具

| 腳本 | 說明 | 範例 |
|------|------|------|
| `scripts/count_image.py` | 統計各子資料夾的圖片數量 | `python count_image.py ../Secu-revised/data/train` |
| `scripts/preview.py` | 將子資料夾圖片排成網格預覽圖 | `python preview.py -i ../Secu-revised/data/train/cluster8` |
| `scripts/pick_color.py` | 互動式取色工具（需 OpenCV GUI） | `python pick_color.py --input ../tiles_output` |
| `Secu-revised/count_parcel.py` | 資料集抽樣與標籤產生 | 詳見腳本內說明 |

---

## 目錄結構

```
.
├── Secu-revised/              # 核心 ML：SeCu 深度聚類
│   ├── main.py                # 訓練入口（ResNet / ViT，含 MML 損失）
│   ├── inference.py           # 推論與分群（t-SNE 視覺化、聚類分佈報告）
│   ├── config.py              # 全域設定（clusters_amount、路徑）
│   ├── count_parcel.py        # 資料集工具（抽樣、標籤產生）
│   ├── nets/                  # 骨幹網路架構
│   │   ├── resnet_cifar.py    #   ResNet-18（CIFAR / 224×224）
│   │   ├── resnet_stl.py      #   ResNet-18（STL-10）
│   │   ├── resnet_custom.py   #   ResNet-18（自訂資料集）
│   │   └── vit.py             #   ViT 封裝（timm vit_base_patch16_224）
│   ├── secu/                  # SeCu 演算法模組
│   │   ├── builder.py         #   SeCu 模型定義（medoid + MML）
│   │   ├── folder.py          #   自訂 Dataset（ImageFolder / NPYFolder）
│   │   ├── loader.py          #   資料增強（裁切、模糊、曝光反轉）
│   │   └── optimizer.py       #   LARS 優化器
│   ├── data/                  # 影像資料（gitignored）
│   ├── model/                 # 模型檢查點（gitignored）
│   └── result/                # 推論文字結果
│
└── scripts/                   # 影像前處理工具
    ├── crop_corners.py        #   批次遮蔽圖片角落
    ├── crop_border.py         #   裁切圖片四邊邊框
    ├── tile_images.py         #   固定網格切割圖塊
    ├── adaptive_tile.py       #   自適應偵測非黃色區域並裁切
    ├── count_image.py         #   統計各子資料夾圖片數量
    ├── pick_color.py          #   互動式 HSV 取色工具
    └── preview.py             #   子資料夾圖片網格預覽
```

### 資料目錄格式

```
data/
└── train/
    ├── all/              # 無標籤時，所有影像放在同一個子資料夾即可
    │   ├── img_001.jpg
    │   ├── img_002.npy
    │   └── ...
    │
    # ── 或者，若有多個來源 / 已知類別 ──
    ├── source_A/
    │   └── ...
    └── source_B/
        └── ...
```

> 資料夾名稱在訓練時不影響聚類結果（無監督），但在推論時會作為 Ground Truth 用於計算評估指標。若無真實標籤，放在單一子資料夾即可。

支援格式：`.jpg`、`.jpeg`、`.png`、`.npy`（NumPy array, shape: H×W×C, uint8）

---

## 引用

本專案的聚類演算法基於以下論文：

```bibtex
@inproceedings{qian2023secu,
  author    = {Qi Qian},
  title     = {Stable Cluster Discrimination for Deep Clustering},
  booktitle = {{IEEE/CVF} International Conference on Computer Vision, {ICCV} 2023},
  year      = {2023}
}
```
