import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, mode="train", target_size=(128, 128)):
        if mode == "train":
            mode = "trainval"
        self.dataset = OxfordIIITPet(
            root=root_dir,
            split=mode,
            target_types="segmentation",
            download=True,
            transform=v2.Compose(
                [
                    v2.Resize(target_size),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            ),
            target_transform=v2.Compose(
                [
                    v2.Resize(target_size, interpolation=InterpolationMode.NEAREST),
                    v2.ToImage(),
                    v2.ToDtype(torch.int64, scale=False),
                ]
            ),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        mask = mask - 1  # Adjust label range from 1-3 to 0-2
        return img, mask.squeeze(0)
