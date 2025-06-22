import argparse

from dataset.nyu_depth_dataset import NYUDepthDataset
from dataset.OxfordIIITPet_dataset import OxfordIIITPetDataset
from model.dpt_models import DPT, DPT_pretrained
from model.loss_functions import AffineInvariantDepthLoss, SegmentationLoss
from utils.misc import get_logger, load_config
from utils.trainer import Trainer

__all_datasets__ = {"NYUDepthDataset": NYUDepthDataset, "OxfordIIITPetDataset": OxfordIIITPetDataset}
__all_losses__ = {"AffineInvariantDepthLoss": AffineInvariantDepthLoss, "SegmentationLoss": SegmentationLoss}
__all_models__ = {"DPT": DPT, "DPT_pretrained": DPT_pretrained}


def train(args):
    config = load_config(args.config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model_cfg = config["MODEL"]
    model = __all_models__[model_cfg["model_name"]](model_cfg)
    trainer.set_model(model)

    data_config = config["DATA"]
    train_dataset = __all_datasets__[data_config["dataset"]](
        root_dir=data_config["root"], mode="train", target_shape=(model_cfg["img_size"][0], model_cfg["img_size"][1])
    )
    val_dataset = __all_datasets__[data_config["dataset"]](
        root_dir=data_config["root"], mode="val", target_shape=(model_cfg["img_size"][0], model_cfg["img_size"][1])
    )

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss_fn=__all_losses__[config["OPTIM"]["loss"]]())

    # trainer.overfit_one_batch()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/dpt_semseg_config.yaml", help="Config path")
    args = parser.parse_args()
    train(args)
