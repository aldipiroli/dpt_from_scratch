import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
from dataset.download_dataset import download_nyu_depth_dataset
from torch.utils.data import Dataset


class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, mode="train", target_shape=[384, 384], reshape_input=True, use_augmentation=True):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.target_shape = target_shape
        self.reshape_input = reshape_input
        self.use_augmentation = use_augmentation
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
        img = data["rgb"].astype(np.float32)  # (640, 480, 3)
        img /= 255
        depth = data["depth"].astype(np.float32)  # (640, 480)

        if self.reshape_input:
            img = cv2.resize(img, (self.target_shape[0], self.target_shape[1]), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (self.target_shape[0], self.target_shape[1]), interpolation=cv2.INTER_NEAREST)

        if self.mode == "train" and self.use_augmentation:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                depth = np.fliplr(depth).copy()

            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, k=rot_k).copy()
                depth = np.rot90(depth, k=rot_k).copy()

        img = np.transpose(img, (2, 1, 0))
        depth = np.transpose(depth, (1, 0))
        depth = np.where(depth < 0.2, 0.2, depth)
        disp = 1.0 / depth
        return img, disp
