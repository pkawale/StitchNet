import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
from datetime import datetime

from helper_scripts.CIFAR10Data import CIFAR10Data
from helper_scripts.CIFAR10Module import CIFAR10Module


def main(
    model_name,
    max_epochs,
    batch_size,
    data_dir,
    log_dir,
    learning_rate,
    weight_decay,
    num_workers,
    seed: int = 24682479,
):
    run_logs = Path(log_dir) / f"{model_name}_{datetime.now()}_logs"
    run_logs.mkdir(parents=True, exist_ok=True)
    # TODO - break early if model with same hyperparams exists

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_last=False,
        dirpath=run_logs / "checkpoints",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    logger = TensorBoardLogger(Path(log_dir) / "lightning_logs", name=model_name)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    seed_everything(seed)

    logger.log_hyperparams(
        {
            "model_name": model_name,
            "seed": seed,
            "run_logs": str(run_logs),
        }
    )

    trainer = Trainer(
        devices="auto",
        accelerator="gpu",
        strategy="auto",
        log_every_n_steps=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint, lr_monitor, early_stop],
        logger=logger,
    )

    data_module = CIFAR10Data(
        data_dir, batch_size, num_workers=num_workers, pin_memory=num_workers > 1
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

    model = CIFAR10Module(
        model_name=model_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    # Save the model parameters
    torch.save({
        'model_name': model_name,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'state_dict': model.state_dict()
    }, run_logs / "model_final.pth")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model (Pre)training Script")

    # PROGRAM level args
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory to store CIFAR-10 data"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory to store Lightning logs and model checkpoints",
    )

    # TRAINER args
    parser.add_argument("--model_name", type=str, required=True, help="TIMM model name")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Max number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--seed", type=int, default=24682479, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )

    args = parser.parse_args()

    main(
        args.model_name,
        args.max_epochs,
        args.batch_size,
        args.data_dir,
        args.log_dir,
        args.learning_rate,
        args.weight_decay,
        args.num_workers,
        args.seed,
    )
