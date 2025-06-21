import argparse

from dataset.nyu_depth_dataset import NYUDepthDataset
from dataset.OxfordIIITPet_dataset import OxfordIIITPetDataset
from model.dpt_models import DPT_standard
from model.loss_functions import AffineInvariantDepthLoss, SegmentationLoss
from model.simple_semseg_models import ViTSegmentationModel
from utils.misc import get_logger, load_config
from utils.trainer import Trainer

__all_datasets__ = {"NYUDepthDataset": NYUDepthDataset, "OxfordIIITPetDataset": OxfordIIITPetDataset}
__all_losses__ = {"AffineInvariantDepthLoss": AffineInvariantDepthLoss, "SegmentationLoss": SegmentationLoss}


def train(args):
    config = load_config(args.config)
    print(config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model_cfg = config["MODEL"]
    model_dpt = DPT_standard(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        embed_size=model_cfg["embed_size"],
        reassamble_embed_size=model_cfg["reassamble_embed_size"],
        scales=model_cfg["scales"],
        blocks_ids=model_cfg["blocks_ids"],
        num_encoder_blocks=model_cfg["num_encoder_blocks"],
        num_outputs=model_cfg["num_outputs"],
    )
    model_vit = ViTSegmentationModel(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        num_classes=model_cfg["num_outputs"],
    )
    model = model_vit
    model = model_dpt
    trainer.set_model(model)

    data_config = config["DATA"]
    train_dataset = __all_datasets__[data_config["dataset"]](root_dir=data_config["root"], mode="train")
    val_dataset = __all_datasets__[data_config["dataset"]](root_dir=data_config["root"], mode="train")

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
