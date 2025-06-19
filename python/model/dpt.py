import math

import torch
import torch.nn as nn


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
        self.projection_matrix = nn.Linear(int(self.out_size / 3) * num_heads, self.out_size)

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
        """
        self.patch_size = patch_size
        self.scale_size = scale_size

        self.embed_projection = nn.Conv2d(embed_size, new_embed_size, kernel_size=1, stride=1)
        if scale_size >= patch_size:
            self.resample = nn.Conv2d(
                new_embed_size, new_embed_size, kernel_size=3, stride=scale_size // patch_size, padding=1
            )
        else:
            padding = 3 if scale_size == 4 else 1
            self.resample = nn.ConvTranspose2d(
                new_embed_size,
                new_embed_size,
                kernel_size=3,
                stride=patch_size // scale_size,
                padding=1,
                output_padding=padding,
            )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (b,H/s,W/s,D') -> (b,D',H/s,W/s)
        print(x.shape)
        x_embed = self.embed_projection(x)
        y = self.resample(x_embed)
        return y


class ReassambleModule(nn.Module):
    def __init__(self, img_size, patch_size, scale_size, embed_size, new_embed_size, read_type="ignore"):
        super(ReassambleModule, self).__init__()
        self.read_module = ReadModule(read_type=read_type)
        self.concat_module = ConcatenateModule(img_size, patch_size)
        self.resample_module = ResampleModule(
            patch_size=patch_size, scale_size=scale_size, embed_size=embed_size, new_embed_size=new_embed_size
        )

    def forward(self, x):
        x_read = self.read_module(x)
        x_concat = self.concat_module(x_read)
        x_resample = self.resample_module(x_concat)
        return x_resample


class DPT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_size=128,
        num_encoder_blocks=3,
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

        self.patcher = PatchImage(patch_size)
        self.patch_embed_transform = nn.Linear(self.patch_dim, self.embed_size)
        self.positional_embeddings = nn.Parameter(
            data=torch.randn(self.num_patches + 1, embed_size), requires_grad=True
        )
        self.class_token = nn.Parameter(data=torch.randn(1, 1, embed_size), requires_grad=True)

        self.encoders = torch.nn.ModuleList(
            [TransformerEncoder(embed_size=embed_size, num_patches=self.num_patches) for i in range(num_encoder_blocks)]
        )
        self.reassamble_module_4 = ReassambleModule(
            img_size=img_size,
            patch_size=patch_size,
            scale_size=4,
            embed_size=embed_size,
            new_embed_size=256,
            read_type="ignore",
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patcher(x)
        class_token = self.class_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((class_token, self.patch_embed_transform(x)), 1) + self.positional_embeddings
        z = embeddings
        for encoder in self.encoders:
            z = encoder(z)


if __name__ == "__main__":
    h, w = 384, 384
    x = torch.rand(2, 3, h, w)
    model = DPT(
        img_size=(h, w, 3),
        patch_size=16,
        embed_size=128,
        num_encoder_blocks=3,
        num_classes=10,
    )
    y = model(x)
    print(y.shape)
