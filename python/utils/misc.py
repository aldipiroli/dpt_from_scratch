import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import yaml


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device


def to_device(x):
    x = x.to(get_device())
    return x


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{now}.log")
    logger = logging.getLogger(f"logger_{now}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def plot_images(images_list, filename="tmp.png", curr_iter=0):
    num_images = len(images_list)
    fig, axs = plt.subplots(num_images, 1, figsize=(5, num_images * 3))
    for i, tensor in enumerate(images_list):
        axs[i].imshow(tensor.detach().cpu(), cmap="magma")
        axs[i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig.suptitle(f"iter {curr_iter} - {current_datetime}")
    plt.savefig(filename, format="png", bbox_inches="tight")
    print(f"Saved figure {filename}")
    plt.close()
