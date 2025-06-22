import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dpt_models import OutputHead
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


class LRASPP_MobileNet_V3(nn.Module):
    def __init__(self, num_outputs=1):
        super(LRASPP_MobileNet_V3, self).__init__()
        self.model = lraspp_mobilenet_v3_large(weights="COCO_WITH_VOC_LABELS_V1")  # pretrained=False
        self.depth_head = OutputHead(embed_size=960, num_outputs=num_outputs)

    def forward(self, x):
        features = self.model.backbone(x)
        high_res_features = features["high"]
        upsampled_features = F.interpolate(high_res_features, size=(192, 192), mode="bilinear", align_corners=False)
        depth_pred = self.depth_head(upsampled_features)
        return depth_pred.squeeze(1)


class ViTSegmentationModel(nn.Module):
    def __init__(self, img_size=[128, 128, 3], patch_size=16, dim=256, depth=6, heads=8, mlp_dim=512, num_classes=3):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        assert self.img_size[0] == self.img_size[1]
        self.h = self.w = img_size[0] // patch_size
        num_patches = self.h * self.w

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 2, stride=2), nn.ReLU(), nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x)  # [B, D, H', W']
        _, D, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.encoder(x)

        x = x[:, 1:].transpose(1, 2).reshape(B, D, H_p, W_p)
        x = self.decoder(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x
