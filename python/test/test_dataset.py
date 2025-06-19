import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dataset.download_dataset import download_nyu_depth_dataset
from dataset.nyu_depth_dataset import NYUDepthDataset
from torch.utils.data import DataLoader


def test_dummy():
    v1 = torch.tensor([1, 0, 0]).float()
    v2 = torch.tensor([1, 0, 0]).float()
    assert torch.allclose(v1, v2)


def test_download_nyu_depth_dataset(skip=True):
    if not skip:
        download_nyu_depth_dataset(data_dir="../data")


def test_nyu_dataset():
    dataset = NYUDepthDataset(root_dir="../data")
    bs = 2
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
    )
    itr = iter(dataloader)
    img, depth = next(itr)
    assert img.shape == (bs, 3, 480, 640)
    assert depth.shape == (bs, 480, 640)


if __name__ == "__main__":
    print("All test passed!")
