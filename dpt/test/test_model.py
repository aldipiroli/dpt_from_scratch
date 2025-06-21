import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from model.dpt_models import DPT_standard, FusionModule, OutputHead, PatchImage, ResampleModule, ResidualConvUnit


@pytest.mark.parametrize(
    "b, c, w, h, ps",
    [
        (2, 3, 480, 640, 16),
        (2, 3, 480, 480, 16),
        (2, 3, 480, 480, 8),
    ],
)
def test_image_patcher(b, c, w, h, ps):
    patcher = PatchImage(ps)
    x = torch.rand(b, c, w, h)
    fold = patcher(x)
    assert fold.shape[1] == (w // ps) * (h // ps)
    assert fold.shape[2] == ps**2 * c

    unfold = patcher.fold(fold, output_size=(w, h))
    assert unfold.shape == (b, c, w, h)


@pytest.mark.parametrize(
    "b, w, h, ps, ss",
    [
        (2, 384, 384, 16, 4),
        (2, 384, 384, 16, 8),
        (2, 384, 384, 16, 16),
        (2, 384, 384, 16, 32),
        (2, 384, 640, 16, 4),
        (2, 384, 640, 16, 32),
    ],
)
def test_resample_module(b, w, h, ps, ss):
    patch_size = ps
    scale_size = ss
    embed_size = 128
    new_embed_size = 256
    x = torch.rand(b, h // patch_size, w // patch_size, embed_size)
    reassemble = ResampleModule(
        patch_size=patch_size, scale_size=scale_size, embed_size=embed_size, new_embed_size=new_embed_size
    )
    y = reassemble(x)
    assert y.shape == (b, new_embed_size, h // scale_size, w // scale_size)


@pytest.mark.parametrize(
    "b, w, h, ss",
    [
        (2, 384, 384, 4),
        (2, 384, 384, 8),
        (2, 384, 384, 16),
        (2, 384, 384, 32),
        (2, 384, 640, 32),
    ],
)
def test_residual_conv_unit(b, w, h, ss):
    scale_size = ss
    embed_size = 128
    x = torch.rand(b, embed_size, h // scale_size, w // scale_size)

    rcu = ResidualConvUnit(embed_size)
    y = rcu(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "h, w, ss",
    [
        (384, 384, 32),
        (384, 384, 16),
        (384, 384, 8),
        (384, 384, 4),
    ],
)
def test_fusion_module(h, w, ss):
    embed_size = 128
    scale_size = ss
    bs = 2
    x1 = torch.rand(bs, embed_size, h // scale_size, w // scale_size)
    x2 = torch.rand(bs, embed_size, h // scale_size, w // scale_size)
    rcu = FusionModule(embed_size)
    y = rcu(x1, x2)
    assert y.shape == (bs, embed_size, (h // scale_size) * 2, (w // scale_size) * 2)


def test_depth_estimation_head():
    embed_size = 128
    h, w = 192, 192
    num_outputs = 3
    head = OutputHead(embed_size=embed_size, num_outputs=num_outputs)
    x = torch.rand(2, embed_size, h, w)
    y = head(x)
    assert y.shape[1] == num_outputs


@pytest.mark.parametrize(
    "h, w",
    [
        (384, 384),
        (480, 640),
    ],
)
def test_dpt_model(h, w):
    model = DPT_standard(
        img_size=(h, w, 3),
        patch_size=16,
        embed_size=128,
        num_encoder_blocks=12,
        scales=[4, 8, 16, 32],
        blocks_ids=[2, 5, 8, 11],
        reassamble_embed_size=256,
        num_heads=8,
        num_outputs=1,
    )
    x = torch.rand(2, 3, h, w)
    depth_pred = model(x)
    assert depth_pred.shape == (2, 1, h, w)


if __name__ == "__main__":
    print("All test passed!")
