from dataset.nyu_depth_dataset import NYUDepthDataset
from model.dense_prediction_transformer import DPT
from model.depth_loss import AffineInvariantDepthLoss
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


def train():
    config = load_config("config/depth_anything_config.yaml")
    print(config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model_cfg = config["MODEL"]
    model = DPT(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        embed_size=model_cfg["embed_size"],
        reassamble_embed_size=model_cfg["reassamble_embed_size"],
        scales=model_cfg["scales"],
        num_encoder_blocks=model_cfg["num_encoder_blocks"],
    )
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
