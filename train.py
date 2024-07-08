from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
from datetime import datetime
from CIFAR10Data import CIFAR10Data
from CIFAR10Module import CIFAR10Module


def main(
    model_name,
    num_epochs,
    batch_size,
    data_dir,
    log_dir,
    learning_rate,
    weight_decay,
    seed: int = 24682479,
):
    log_dir = Path(log_dir) / f"{model_name}_{datetime.now()}_logs"

    # TODO - break early if model with same hyperparams exists

    checkpoint = ModelCheckpoint(
        monitor="val_loss", mode="min", save_last=False, dirpath=log_dir / "checkpoints"
    )
    logger = TensorBoardLogger(Path(log_dir) / "lightning_logs", name=model_name)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    seed_everything(seed)

    logger.log_hyperparams(
        {
            "model_name": model_name,
            "seed": seed,
            "log_dir": log_dir,
        }
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        strategy="auto",
        log_every_n_steps=1,
        max_epochs=num_epochs,
        callbacks=[checkpoint, lr_monitor],
        logger=logger,
    )

    data_module = CIFAR10Data(data_dir, batch_size)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    model = CIFAR10Module(
        model_name=model_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=4,
    )

    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())


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
        "--num_epochs", type=int, default=10, help="Number of training epochs"
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

    args = parser.parse_args()

    main(
        args.model_name,
        args.num_epochs,
        args.batch_size,
        args.data_dir,
        args.log_dir,
        args.learning_rate,
        args.weight_decay,
        args.seed,
    )
