import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


class PatchImage(nn.Module):
    def __init__(self, patch_kernel_size=16):
        super(PatchImage, self).__init__()
        """
        https://github.com/aldipiroli/mae_from_scratch/blob/main/python/model/mae.py
        """
        self.patch_kernel_size = patch_kernel_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_kernel_size, stride=patch_kernel_size)

    def forward(self, x):
        x = self.unfold(x)  # B x (C*P*P) x N
        x = x.permute(0, 2, 1)  # (B, N, C*P*P)
        return x

    def fold(self, x, output_size):
        x = x.permute(0, 2, 1)  # (B, C*P*P, N)
        fold = torch.nn.Fold(
            output_size=output_size,
            kernel_size=self.patch_kernel_size,
            stride=self.patch_kernel_size,
        )
        x = fold(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_size=128, out_size=128):
        super(SelfAttention, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.K = nn.Linear(self.in_size, self.out_size)
        self.Q = nn.Linear(self.in_size, self.out_size)
        self.V = nn.Linear(self.in_size, self.out_size)

    def forward(self, x):
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)

        qk = torch.matmul(q, k.transpose(-2, -1))
        attention = nn.functional.softmax(qk / math.sqrt(self.out_size), -1)
        self_attention = torch.matmul(attention, v)
        return self_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_size=128, out_size=128, num_heads=3):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.attention_heads = torch.nn.ModuleList(
            [SelfAttention(self.in_size, int(self.out_size / self.num_heads)) for i in range(self.num_heads)]
        )
        self.projection_matrix = nn.Linear((self.out_size // self.num_heads) * self.num_heads, self.out_size)

    def forward(self, x):
        all_self_attentions = []
        for attention_head in self.attention_heads:
            z = attention_head(x)
            all_self_attentions.append(z)
        all_self_attentions = torch.cat(all_self_attentions, -1)
        z = self.projection_matrix(all_self_attentions)
        return z


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size=128, num_patches=16):
        super(TransformerEncoder, self).__init__()
        # From: https://github.com/aldipiroli/ViT_from_scratch/blob/main/python/models/simple_vit.py
        self.embed_size = embed_size
        self.num_patches = num_patches

        self.ln = nn.LayerNorm(embed_size)
        self.multi_head_self_attention = MultiHeadSelfAttention(
            in_size=self.embed_size, out_size=self.embed_size, num_heads=3
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size * 2),
            nn.GELU(),
            nn.Linear(self.embed_size * 2, self.embed_size),
        )

    def forward(self, z):
        z_norm = self.ln(z)
        z0 = self.multi_head_self_attention(z_norm)
        z1 = z + z0
        z = self.mlp(self.ln(z1))
        z = z + z1
        return z


class ReadIgnore(nn.Module):
    def __init__(self):
        super(ReadIgnore, self).__init__()

    def forward(self, x):
        return x[:, 1:, :]


class ReadModule(nn.Module):
    def __init__(self, read_type="ignore"):
        super(ReadModule, self).__init__()
        """
        Projects token from (Np+1,D)->(Np,D) i.e., handles cls token
        """
        self.read_type = read_type
        if read_type == "ignore":
            self.read = ReadIgnore()

    def forward(self, x):
        return self.read(x)


class ConcatenateModule(nn.Module):
    def __init__(self, img_size, patch_size):
        super(ConcatenateModule, self).__init__()
        """
        Projects token from (Np,D)->(H/p,W/p,D)
        """
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        h, w, c = self.img_size
        b = x.shape[0]
        embed_size = x.shape[-1]
        x_reshape = x.reshape(b, h // self.patch_size, w // self.patch_size, embed_size)
        return x_reshape


class ResampleModule(nn.Module):
    def __init__(self, patch_size, scale_size, embed_size, new_embed_size):
        super(ResampleModule, self).__init__()
        """
        Projects token from (H/p,W/p,D)->(H/s,W/s,D')
        Note: paper uses ConvTranspose2d but official code uses nn.functional.interpolate
        """
        self.patch_size = patch_size
        self.scale_size = scale_size

        self.embed_projection = nn.Conv2d(embed_size, new_embed_size, kernel_size=1, stride=1)

    def forward(self, x, permute=True):
        if permute:
            x = x.contiguous().permute(0, 3, 1, 2)  # (b,H/s,W/s,D') -> (b,D',H/s,W/s)
        x_embed = self.embed_projection(x)
        y = nn.functional.interpolate(
            x_embed, scale_factor=self.patch_size / self.scale_size, mode="bilinear", align_corners=True
        )
        return y


class ReassambleModule(nn.Module):
    def __init__(self, img_size, patch_size, scale_size, embed_size, new_embed_size, read_type="ignore"):
        super(ReassambleModule, self).__init__()
        self.read_module = ReadModule(read_type=read_type)
        self.concat_module = ConcatenateModule(img_size, patch_size)
        self.reassemble_module = ResampleModule(
            patch_size=patch_size, scale_size=scale_size, embed_size=embed_size, new_embed_size=new_embed_size
        )

    def forward(self, x):
        x_read = self.read_module(x)
        x_concat = self.concat_module(x_read)
        x_reassemble = self.reassemble_module(x_concat)
        return x_reassemble


class ResidualConvUnit(nn.Module):
    def __init__(self, embed_size):
        super(ResidualConvUnit, self).__init__()
        # From: https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(embed_size, embed_size, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(embed_size, embed_size, kernel_size=3, stride=1, padding=1)
        self.conv_module = nn.Sequential(self.relu, self.conv_1, self.relu, self.conv_2)

    def forward(self, x):
        x_ = self.conv_module(x)
        x = x_ + x
        return x


class FusionModule(nn.Module):
    def __init__(self, new_embed_size):
        super(FusionModule, self).__init__()
        self.rcu_1 = ResidualConvUnit(new_embed_size)
        self.rcu_2 = ResidualConvUnit(new_embed_size)
        self.project = nn.Conv2d(new_embed_size, new_embed_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2=None):
        x1 = self.rcu_1(x1)
        if x2 is not None:
            x = x1 + x2
        else:
            x = x1
        x = self.rcu_2(x)
        x_reassamble = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x_project = self.project(x_reassamble)
        return x_project


class DepthEstimationHead(nn.Module):
    def __init__(self, embed_size):
        super(DepthEstimationHead, self).__init__()
        # Note: in supplemenatry material of "Vision Transformers for Dense Prediction"
        self.conv_1 = nn.Conv2d(embed_size, embed_size // 2, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Sequential(nn.Conv2d(embed_size // 2, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1), nn.ReLU()
        )  # Note: final ReLU sometimes leads to gradinet saturation

    def forward(self, x):
        x = self.conv_1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class DPT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_size=128,
        num_encoder_blocks=4,
        scales=[4, 8, 16, 32],
        reassamble_embed_size=256,
    ):
        super(DPT, self).__init__()
        """
        Implementation of the paper: "Vision Transformers for Dense Prediction" (https://arxiv.org/pdf/2103.13413)
        """
        height, width, channels = img_size
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.patch_dim = patch_size**2 * channels
        self.patch_size = patch_size

        self.embed_size = embed_size
        self.num_encoder_blocks = num_encoder_blocks
        assert num_encoder_blocks >= len(scales)

        self.patcher = PatchImage()
        self.patch_embed_transform = nn.Linear(self.patch_dim, self.embed_size)
        self.positional_embeddings = nn.Parameter(
            data=torch.randn(self.num_patches + 1, embed_size), requires_grad=True
        )
        self.class_token = nn.Parameter(data=torch.randn(1, 1, embed_size), requires_grad=True)

        self.encoders = torch.nn.ModuleList(
            [TransformerEncoder(embed_size=embed_size, num_patches=self.num_patches) for i in range(num_encoder_blocks)]
        )
        self.reassamble_modules = torch.nn.ModuleList(
            [
                ReassambleModule(
                    img_size=img_size,
                    patch_size=patch_size,
                    scale_size=scale_size,
                    embed_size=embed_size,
                    new_embed_size=reassamble_embed_size,
                    read_type="ignore",
                )
                for scale_size in scales
            ]
        )

        self.fusion_modules = torch.nn.ModuleList([FusionModule(reassamble_embed_size) for scale_size in scales])
        self.depth_pred_head = DepthEstimationHead(embed_size=reassamble_embed_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patcher(x)
        class_token = self.class_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((class_token, self.patch_embed_transform(x)), 1) + self.positional_embeddings
        z = embeddings
        all_reassamble_outputs = []
        for layer_id, encoder in enumerate(self.encoders):
            z = encoder(z)
            curr_reassable_output = self.reassamble_modules[layer_id](z)  # TODO: use reassamble based on fixed layer_id
            all_reassamble_outputs.append(curr_reassable_output)

        all_reassamble_outputs = all_reassamble_outputs[::-1]
        r2 = None
        for r_id in range(len(all_reassamble_outputs)):
            r1 = all_reassamble_outputs[r_id]
            r2 = self.fusion_modules[r_id](r1, r2)

        depth_pred = self.depth_pred_head(r2)
        depth_pred = depth_pred.squeeze(1)
        return depth_pred


class LRASPP_MobileNet_V3(nn.Module):
    def __init__(self):
        super(LRASPP_MobileNet_V3, self).__init__()
        self.model = lraspp_mobilenet_v3_large(weights="COCO_WITH_VOC_LABELS_V1")  # pretrained=False
        self.depth_head = DepthEstimationHead(960)

    def forward(self, x):
        features = self.model.backbone(x)
        high_res_features = features["high"]
        upsampled_features = F.interpolate(high_res_features, size=(192, 192), mode="bilinear", align_corners=False)
        depth_pred = self.depth_head(upsampled_features)
        return depth_pred.squeeze(1)


if __name__ == "__main__":
    h, w = 384, 384
    x = torch.rand(2, 3, h, w)

    model = LRASPP_MobileNet_V3()
    y = model(x)
