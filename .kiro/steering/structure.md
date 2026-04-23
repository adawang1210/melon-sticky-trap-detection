# 專案結構

```
.
├── Secu-revised/              # 核心 ML：SeCu 深度聚類
│   ├── config.py              # 全域設定（輸出路徑、clusters_amount）
│   ├── main.py                # 訓練入口（ResNet/ViT/DINOv2、MML 損失）
│   ├── inference.py           # 推論（自訂資料、GT 來自資料夾名稱）
│   ├── count_parcel.py        # 資料集工具（抽樣、標籤產生）
│   ├── nets/                  # 骨幹網路架構
│   │   ├── resnet_cifar.py    # ResNet-18（CIFAR 用，32x32 / 224x224）
│   │   ├── resnet_stl.py      # ResNet-18（STL-10 用）
│   │   ├── resnet_custom.py   # ResNet-18（自訂資料集用）
│   │   └── vit.py             # ViT / DINOv2 封裝
│   ├── secu/                  # SeCu 演算法模組
│   │   ├── builder.py         # SeCu 模型（medoid + MML 損失）
│   │   ├── folder.py          # 自訂 Dataset 類別（ImageFolder、NPYFolder）
│   │   ├── loader.py          # 資料增強轉換（裁切、模糊、曝光反轉）
│   │   └── optimizer.py       # LARS 優化器
│   ├── data/                  # 影像資料（已加入 gitignore）
│   ├── model/                 # 儲存的模型檢查點（已加入 gitignore）
│   ├── output/                # 分群結果輸出
│   └── result/                # 推論文字結果
│
├── scripts/                   # 影像前處理與工具
│   ├── crop_corners.py        # 批次遮蔽圖片角落（遮蔽無關區域）
│   ├── crop_border.py         # 裁切大圖的固定邊框
│   ├── adaptive_tile.py       # 自動偵測非黃色區域並裁切
│   ├── count_image.py         # 統計各子資料夾的圖片數量
│   └── preview.py             # 將子資料夾圖片排成網格預覽圖
│
├── requirements.txt           # Python 套件依賴
└── .kiro/steering/            # AI 助手引導規則
```

## 資料格式慣例
- 訓練資料：`Secu-revised/data/<子資料夾>/*.npy` 或 `*.jpg`（無標籤，子資料夾名稱不影響訓練）
- 推論使用與訓練相同的資料目錄
- 分群結果輸出到 `Secu-revised/output/cluster<N>/`
- NPY 檔案儲存原始影像陣列 (H, W, C)，通常為 uint8 RGB 或 RGBA
- Ground Truth 來自資料夾名稱（僅用於推論時的評估指標計算，非必要）

## 關鍵慣例
- `secu/` 子目錄作為 Python 套件匯入（`import secu.builder`）
- `config.py` 是共用設定檔，被 builder 和推論腳本共同匯入
- `config.py` 的 `folder_path` 控制分群結果輸出位置（預設 `.\output`）
- 模型檢查點每 50 個 epoch 儲存至 `model/<log>_<epoch>.pth.tar`
- 最佳模型儲存為 `model/best_model.pth.tar`
- `scripts/` 中的腳本是獨立的 CLI 工具（使用 argparse），不屬於 secu 套件
- 支援三種 backbone：`resnet18`、`vit`、`dinov2`（透過 `--backbone` 參數切換）
