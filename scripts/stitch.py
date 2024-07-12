from pathlib import Path

import pytorch_lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from CIFAR10Module import CIFAR10Module
from StitchingModule import LightningStitchingModel
from stitching_layer import StitchingModel
from utils import find_checkpoint_for_model

results = {
    "init": {},
    "after_regression": {},
    "before_training": {},
    "after_training": {},
}


def save_results_metadata(results, model1, model2):
    results["init"]["model1_state_dict"] = model1.state_dict()
    results["init"]["model2_state_dict"] = model2.state_dict()

    results["after_regression"]["model1_state_dict"] = model1.state_dict()
    results["after_regression"]["model2_state_dict"] = model2.state_dict()

    results["before_training"]["model1_state_dict"] = model1.state_dict()
    results["before_training"]["model2_state_dict"] = model2.state_dict()

    results["after_training"]["model1_state_dict"] = model1.state_dict()
    results["after_training"]["model2_state_dict"] = model2.state_dict()


def snapshot(model, description):
    return {
        "description": description,
        "state_dict": model.state_dict(),
    }


def load_models(model1_name, model2_name, log_dir):
    # Load the pre-existing model from log_dir
    checkpoint_path = find_checkpoint_for_model(log_dir, model1_name)
    print("[INFO]: loading model1 from", checkpoint_path)
    model1 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    checkpoint_path = find_checkpoint_for_model(log_dir, model2_name)
    print("[INFO]: loading model2 from", checkpoint_path)
    model2 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    return model1, model2


def do_linear_regression(stitching_model, device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    batch_im, _ = next(iter(train_loader))
    batch_im = batch_im.to(device)
    stitching_model.initialize_stitching_layer(batch_im)


def main(model1_name, model2_name, split1, split2, log_dir, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1, model2 = load_models(model1_name, model2_name, log_dir)
    stitching_lightning_model = LightningStitchingModel(
        model1, model2, split1, split2
    ).to(device)

    log_dir = Path(log_dir) / "stitching_logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=log_dir,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    logger = TensorBoardLogger(save_dir=log_dir, name="lightning_logs")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        progress_bar_refresh_rate=20,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    trainer.fit(stitching_lightning_model, train_loader, val_loader)
    torch.save(stitching_lightning_model.state_dict(), log_dir / "stitched_model.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stitching Model Creation Script")
    parser.add_argument(
        "--model1_name", type=str, required=True, help="Name of the first model"
    )
    parser.add_argument(
        "--model2_name", type=str, required=True, help="Name of the second model"
    )
    parser.add_argument(
        "--index1",
        type=int,
        required=True,
        help="Split Index of the layer in the first model",
    )
    parser.add_argument(
        "--index2",
        type=int,
        required=True,
        help="Split Index of the layer in the second model",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Directory to store logs and checkpoints",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train the model"
    )
    args = parser.parse_args()

    main(
        args.model1_name,
        args.model2_name,
        args.index1,
        args.index2,
        args.log_dir,
        args.num_epochs,
    )
