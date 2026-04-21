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
    """DINOv2 封裝，介面與 ViT 完全相同，可直接替換使用。

    用法（在 main.py 中）：
        --backbone dinov2

    支援的模型大小（透過 model_size 參數控制）：
        's' → dinov2_vits14  (21M params, embed_dim=384)   ← 記憶體較小時使用
        'b' → dinov2_vitb14  (86M params, embed_dim=768)   ← 預設，推薦
        'l' → dinov2_vitl14  (300M params, embed_dim=1024) ← 效果最好但吃記憶體
    """

    # 模型名稱 → 特徵維度 對照表
    _MODELS = {
        's': ('dinov2_vits14', 384),
        'b': ('dinov2_vitb14', 768),
        'l': ('dinov2_vitl14', 1024),
    }

    def __init__(self, num_classes=512, model_size='b'):
        super().__init__()
        model_name, embed_dim = self._MODELS[model_size]
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', model_name, pretrained=True
        )
        # 凍結 backbone（DINOv2 已經預訓練好，只訓練 projection head）
        for param in self.backbone.parameters():
            param.requires_grad = False

        # fc：與 ViT 相同介面，builder.py 會讀取 fc.weight.shape[1] 並替換
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # DINOv2 輸入建議 224x224（會自動 interpolate position embedding）
        with torch.no_grad():
            features = self.backbone(x)  # (B, embed_dim)
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