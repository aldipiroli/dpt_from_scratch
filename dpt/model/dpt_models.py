import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            x_embed, scale_factor=self.patch_size / self.scale_size, mode="bilinear", align_corners=False
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
    def __init__(self, new_embed_size, use_rcu=False):
        super(FusionModule, self).__init__()
        self.rcu_1 = ResidualConvUnit(new_embed_size) if use_rcu else nn.Identity()
        self.rcu_2 = ResidualConvUnit(new_embed_size) if use_rcu else nn.Identity()
        self.project = nn.Conv2d(new_embed_size, new_embed_size, kernel_size=1, stride=1, padding=0)
        self.upsample_2x = nn.ConvTranspose2d(
            new_embed_size,
            new_embed_size,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x1, x2=None):
        x1 = self.rcu_1(x1)
        if x2 is not None:
            x = x1 + x2
        else:
            x = x1
        x = self.rcu_2(x)
        x_reassamble = self.upsample_2x(x)
        return x_reassamble


class OutputHead(nn.Module):
    def __init__(self, embed_size, num_outputs=1, img_size=(128, 128)):
        super(OutputHead, self).__init__()
        # Note: in supplemenatry material of "Vision Transformers for Dense Prediction"
        self.img_size = img_size
        self.conv_1 = nn.Conv2d(embed_size, embed_size // 2, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Sequential(nn.Conv2d(embed_size // 2, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, num_outputs, kernel_size=1, stride=1)
        )  # Note: final ReLU sometimes leads to gradinet saturation

    def forward(self, x):
        x = self.conv_1(x)
        x = F.interpolate(x, size=(self.img_size[0], self.img_size[1]), mode="bilinear", align_corners=False)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class DPT_standard(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_size=128,
        num_encoder_blocks=12,
        scales=[4, 8, 16, 32],
        blocks_ids=[2, 5, 8, 11],
        reassamble_embed_size=256,
        num_heads=8,
        dropout=0.1,
        num_outputs=1,
    ):
        super(DPT_standard, self).__init__()
        height, width, channels = img_size
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.patch_dim = patch_size**2 * channels
        self.patch_size = patch_size

        self.embed_size = embed_size
        self.num_encoder_blocks = num_encoder_blocks
        self.blocks_ids = blocks_ids
        assert num_encoder_blocks >= len(scales)
        assert max(blocks_ids) <= num_encoder_blocks

        self.patcher = PatchImage()
        self.patch_embed_transform = nn.Linear(self.patch_dim, self.embed_size)
        self.positional_embeddings = nn.Parameter(
            data=torch.randn(self.num_patches + 1, embed_size), requires_grad=True
        )
        self.class_token = nn.Parameter(data=torch.randn(1, 1, embed_size), requires_grad=True)

        self.encoders = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=embed_size,
                        nhead=num_heads,
                        dim_feedforward=embed_size * 4,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    ),
                    num_layers=1,
                )
                for _ in range(num_encoder_blocks)
            ]
        )

        self.reassamble_modules = nn.ModuleList(
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

        self.fusion_modules = nn.ModuleList([FusionModule(reassamble_embed_size) for _ in scales])
        self.depth_pred_head = OutputHead(
            embed_size=reassamble_embed_size, num_outputs=num_outputs, img_size=(img_size[0], img_size[0])
        )
        self.upsample2x = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patcher(x)
        class_token = self.class_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((class_token, self.patch_embed_transform(x)), 1) + self.positional_embeddings
        z = embeddings
        all_reassamble_outputs = []
        blocks_ids = copy.deepcopy(self.blocks_ids)
        curr_block_id = 0
        for layer_id, encoder in enumerate(self.encoders):
            z = encoder(z)
            if blocks_ids[0] == layer_id:
                curr_reassable_output = self.reassamble_modules[curr_block_id](z)
                all_reassamble_outputs.append(curr_reassable_output)
                blocks_ids.pop(0)
                curr_block_id += 1

        assert len(all_reassamble_outputs) == len(self.blocks_ids), len(all_reassamble_outputs)
        all_reassamble_outputs = all_reassamble_outputs[::-1]
        r2 = None
        for r_id in range(len(all_reassamble_outputs)):
            r1 = all_reassamble_outputs[r_id]
            r2 = self.fusion_modules[r_id](r1, r2)

        r2 = self.upsample2x(r2)
        depth_pred = self.depth_pred_head(r2)
        return depth_pred


if __name__ == "__main__":
    h, w = 384, 384
    x = torch.rand(2, 3, h, w)
