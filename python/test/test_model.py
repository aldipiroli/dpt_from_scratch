import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from model.dpt import PatchImage, ResampleModule


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
    ],
)
def test_resample_module(b, w, h, ps, ss):
    patch_size = ps
    scale_size = ss
    embed_size = 128
    new_embed_size = 256
    x = torch.rand(2, h // patch_size, w // patch_size, embed_size)
    resample = ResampleModule(
        patch_size=patch_size, scale_size=scale_size, embed_size=embed_size, new_embed_size=new_embed_size
    )
    y = resample(x)
    assert y.shape == (b, new_embed_size, h // scale_size, w // scale_size)


if __name__ == "__main__":
    print("All test passed!")
