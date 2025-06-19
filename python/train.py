from dataset.nyu_depth_dataset import NYUDepthDataset
from model.depth_anything_loss import AffineInvariantDepthLoss
from model.depth_anything_model import DepthAnythingModel
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


def train():
    config = load_config("config/depth_anything_config.yaml")
    print(config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model_cfg = config["MODEL"]
    model = DepthAnythingModel(model_cfg)
    trainer.set_model(model)

    data_config = config["DATA"]
    train_dataset = NYUDepthDataset(root_dir=data_config["root"], mode="train")
    val_dataset = NYUDepthDataset(root_dir=data_config["root"], mode="val")

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss_fn=AffineInvariantDepthLoss())
    trainer.load_latest_checkpoint()
    # trainer.overfit_one_batch()
    trainer.train()


if __name__ == "__main__":
    train()
