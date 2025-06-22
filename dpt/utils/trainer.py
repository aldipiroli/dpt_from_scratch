import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import get_device, plot_images


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.logger.info(f"config: {config}")
        self.epoch = 0
        self.total_iters = 0

        self.ckpt_dir = Path(config["CKPT_DIR"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")
        self.artifacts_img_dir = Path(config["IMG_OUT_DIR"])
        self.artifacts_img_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = config["OPTIM"]["eval_every"]
        self.task = config["MODEL"]["task"]
        self.plot_cfg = (
            {"cmap": "magma", "vmax": None, "vmin": None}
            if self.task == "depth_est"
            else {"cmap": "tab10", "vmax": 2, "vmin": 0}
        )

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.logger.info("Model:")
        self.logger.info(self.model)

    def save_checkpoint(self):
        model_path = Path(self.ckpt_dir) / f"ckpt_{str(self.epoch).zfill(4)}.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_path,
        )
        self.logger.info(f"Saved checkpoint in: {model_path}")

    def load_latest_checkpoint(self):
        if not self.ckpt_dir.exists():
            self.logger.info("No checkpoint directory found.")
            return None

        ckpt_files = sorted(self.ckpt_dir.glob("ckpt_*.pt"))
        if not ckpt_files:
            self.logger.info("No checkpoints found.")
            return None

        latest_ckpt = max(ckpt_files, key=lambda x: int(x.stem.split("_")[1]))
        self.logger.info(f"Loading checkpoint: {latest_ckpt}")

        checkpoint = torch.load(latest_ckpt, weights_only=False, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        return latest_ckpt

    def set_dataset(self, train_dataset, val_dataset, data_config):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.data_config = data_config

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=data_config["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
        )
        self.logger.info(f"Train Dataset: {self.train_dataset}")
        self.logger.info(f"Val Dataset: {self.val_dataset}")

    def set_optimizer(self, optim_config):
        self.optim_config = optim_config
        if self.optim_config["optimizer"] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optim_config["lr"])
        elif self.optim_config["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optim_config["lr"], weight_decay=0)
        else:
            raise ValueError("Unknown optimizer")
        if self.optim_config["scheduler"] == "cosine":
            T_max = self.optim_config["T_max"]
            eta_min = self.optim_config["eta_min"]
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
            self.logger.info(f"Scheduler: CosineAnnealingLR(T_max={T_max}, eta_min={eta_min})")
        else:
            self.scheduler = None

        self.use_gradient_clip = optim_config["gradient_clip"]
        self.logger.info(f"Optimizer: {self.optimizer}")

    def scheaduler_step(self):
        if self.scheduler:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_loss_function(self, loss_fn):
        self.loss_fn = loss_fn.to(self.device)
        self.logger.info(f"Loss function {self.loss_fn}")

    def train(self):
        for curr_epoch in range(self.optim_config["num_epochs"]):
            self.epoch = curr_epoch
            self.train_one_epoch()
            if (curr_epoch + 1) % self.eval_every == 0:
                self.evaluate_model()
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        with tqdm(enumerate(self.train_loader), desc=f"Epoch {self.epoch}") as pbar:
            for n_iter, (imgs, labels) in pbar:
                self.optimizer.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)
                preds = self.output_postprocess(preds)

                loss.backward()
                self.gradient_clip()
                self.optimizer.step()
                self.total_iters += 1
                pbar.set_postfix({"total_iters": self.total_iters, "loss": loss.item(), "lr": self.get_lr()})
                if n_iter % 10 == 0:
                    plot_images(
                        [imgs[0].permute(1, 2, 0), preds[0], labels[0]],
                        curr_iter=self.epoch,
                        filename=os.path.join(self.artifacts_img_dir, f"train_img.png"),
                        plot_cfg=self.plot_cfg,
                    )
                self.scheaduler_step()

    def output_postprocess(self, output):
        if output.shape[1] == 1:
            output = output.squeeze(1)  # depth case
        else:
            output = torch.softmax(output, dim=1)  # semseg case
            output = torch.argmax(output, dim=1)
        return output

    def overfit_one_batch(self):
        self.model.train()
        itr = iter(self.train_loader)
        imgs, depths = next(itr)
        for n_iter in range(1000000):
            self.optimizer.zero_grad()
            imgs = imgs.to(self.device)
            depths = depths.to(self.device)

            preds = self.model(imgs)
            loss = self.loss_fn(preds, depths)
            loss.backward()
            self.gradient_clip()
            self.optimizer.step()
            self.total_iters += 1
            print(f"loss {loss}")
            if n_iter % 10 == 0:
                plot_images(
                    [imgs[0].permute(1, 2, 0), preds[0], depths[0]],
                    curr_iter=self.epoch,
                    filename=os.path.join(self.artifacts_img_dir, f"overfit_img.png"),
                    plot_cfg=self.plot_cfg,
                )

    def evaluate_model(self, max_num_samples=3):
        self.logger.info("Running Evaluation...")
        self.model.eval()
        for n_iter, (imgs, labels) in enumerate(self.val_loader):
            if n_iter > max_num_samples:
                break
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(imgs)
            preds = self.output_postprocess(preds)

            if n_iter % self.eval_every == 0:
                plot_images(
                    [imgs[0].permute(1, 2, 0), preds[0], labels[0]],
                    curr_iter=self.epoch,
                    filename=os.path.join(self.artifacts_img_dir, f"val_img_{str(n_iter).zfill(3)}.png"),
                    plot_cfg=self.plot_cfg,
                )

    def gradient_sanity_check(self):
        total_gradient = 0
        no_grad_name = []
        grad_name = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                no_grad_name.append(name)
                self.logger.info(f"None grad: {name}")
            else:
                grad_name.append(name)
                total_gradient += torch.sum(torch.abs(param.grad))
        assert total_gradient == total_gradient
        if len(no_grad_name) > 0:
            self.logger.info(f"no_grad_name {no_grad_name}")
            raise ValueError("layers without gradient are present")
        assert len(no_grad_name) == 0

    def gradient_clip(self):
        if self.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
