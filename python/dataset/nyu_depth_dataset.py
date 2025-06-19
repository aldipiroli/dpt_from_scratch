import glob
import os
from pathlib import Path

import numpy as np
from dataset.download_dataset import download_nyu_depth_dataset
from torch.utils.data import Dataset


class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.root_dir = Path(root_dir)
        self.mode = mode
        download_nyu_depth_dataset(data_dir=self.root_dir)
        self.root_dir = self.root_dir / "nyu"
        all_files = sorted(glob.glob(os.path.join(self.root_dir, "*.npz")))
        n_train = int(len(all_files) * 0.8)
        train_files = all_files[:n_train]
        val_files = all_files[n_train:]
        self.files = train_files if self.mode == "train" else val_files
        print(f"Loaded split {self.mode} with size {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        img = data["rgb"].astype(np.float32)
        img /= 255
        img = np.transpose(img, (2, 1, 0))  # (3, 480, 640)

        depth = data["depth"].astype(np.float32)  # (480, 640)
        depth = np.transpose(depth, (1, 0))

        return img, depth
