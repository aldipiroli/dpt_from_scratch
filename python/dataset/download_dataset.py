import os
import urllib.request

import h5py
import numpy as np


def download_nyu_depth_dataset(data_dir="data"):
    url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    filename = "nyu_depth_v2_labeled.mat"

    save_dir = os.path.join(data_dir, "nyu")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if not os.path.exists(save_path):
        print(f"Downloading NYU Depth v2 labeled subset to {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        print("Download complete.")
    else:
        print("File already exists. Skipping download.")
        return

    with h5py.File(save_path, "r") as f:
        images = f["images"][:]  # (3, 640, 480, N)
        depths = f["depths"][:]  # (640, 480, N)

        n_samples = images.shape[0]
        print(f"Found {n_samples} samples")

        for i in range(n_samples):
            rgb = images[i].transpose(1, 2, 0)  # (640, 480, 3)
            depth = depths[i]  # (640, 480)

            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            depth = depth.astype(np.float32)

            filename = os.path.join(save_dir, f"{i:06d}.npz")
            np.savez_compressed(filename, rgb=rgb, depth=depth)

            if i % 100 == 0 or i == n_samples - 1:
                print(f"Saved: {filename}")
