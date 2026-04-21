# vit.py
import torch
import torch.nn as nn
import timm
from timm.models.layers import trunc_normal_

class ViT(nn.Module):
    def __init__(self, num_classes=512):  # SeCu 的 builder 會傳進來
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0)
        #self.backbone = timm.create_model('vit_tiny_patch4_32', pretrained=False, num_classes=0)
        in_features = self.backbone.num_features  # ViT 輸出的特徵維度是 768
        self.fc = nn.Linear(in_features, num_classes)  # 對齊 SeCu 的需求（如 512 維）

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class DINOv2(nn.Module):
    """DINOv2 封裝，透過 timm 載入，相容 Python 3.8+。

    介面與 ViT 完全相同，可直接替換使用。

    用法（在 main.py 中）：
        --backbone dinov2

    使用 timm 的 vit_base_patch14_dinov2 模型（86M params, embed_dim=768）。
    """

    def __init__(self, num_classes=512):
        super().__init__()
        # 透過 timm 載入 DINOv2（相容 Python 3.8，不需要 torch.hub）
        # 使用 reg4 版本（有 register tokens），輸入 224x224 與現有 pipeline 相容
        self.backbone = timm.create_model(
            'vit_base_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,  # 移除分類頭，只輸出特徵
            img_size=224,   # 強制 224x224 輸入
        )
        embed_dim = self.backbone.num_features  # 768

        # 凍結 backbone（DINOv2 已經預訓練好，只訓練 projection head）
        for param in self.backbone.parameters():
            param.requires_grad = False

        # fc：與 ViT 相同介面，builder.py 會讀取 fc.weight.shape[1] 並替換
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # (B, 768)
        x = self.fc(features)
        return x

class ViT_CIFAR(nn.Module):
    def __init__(self, num_classes=512):  # SeCu 的 builder 會傳 dim 進來
        super().__init__()
        # 建立 ViT-tiny (224 預設)，去掉原本 classifier
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=0
        )

        # 修改 patch_embed，支援 CIFAR-10 (32x32)
        self.backbone.patch_embed.img_size = (32, 32)
        self.backbone.patch_embed.grid_size = (2, 2)
        self.backbone.patch_embed.num_patches = 4  # 2*2 patches

        # 重新初始化位置編碼 (CLS token + 4 patches = 5 tokens)
        embed_dim = self.backbone.embed_dim  # 192
        num_prefix = getattr(self.backbone, 'num_prefix_tokens', 1)
        new_pos_embed = torch.zeros(1, num_prefix + self.backbone.patch_embed.num_patches, embed_dim)
        nn.init.trunc_normal_(new_pos_embed, std=0.02)
        self.backbone.pos_embed = nn.Parameter(new_pos_embed)

        # fc：同時當 projection head，輸出 SeCu 需要的 dim (通常是 512)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)   # (B, 192)
        x = self.fc(x)         # (B, 512)
        return x