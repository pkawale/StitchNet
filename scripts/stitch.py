import argparse
from pathlib import Path
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger

from helper_scripts.CIFAR10Module import CIFAR10Module
from helper_scripts.stitching_layer import StitchingModel
from helper_scripts.CIFAR10Data import CIFAR10Data
from helper_scripts.utils import find_checkpoint_for_model


def save_loss(stitching_model, data_module, trainer):
    return {
        "model1_loss": test_model(stitching_model.model1, data_module, trainer),
        "model2_loss": test_model(stitching_model.model2, data_module, trainer),
        "stitching_model_loss": test_model(stitching_model, data_module, trainer),
    }


def snapshot(stitching_model, data_module, trainer) -> dict:
    return {
        "state_dict": stitching_model.state_dict(),
        "losses": save_loss(stitching_model, data_module, trainer),
    }


def load_model(model_name, log_dir):
    checkpoint_path = find_checkpoint_for_model(log_dir, model_name)
    print(f"[INFO]: loading {model_name} from", checkpoint_path)
    model = CIFAR10Module.load_from_checkpoint(checkpoint_path, map_location="cpu")
    return model


def do_linear_regression(stitching_model, datamodule, device="cpu"):
    train_loader = datamodule.train_dataloader()
    batch_im, _ = next(iter(train_loader))
    stitching_model.initialize_stitching_layer(batch_im.to(device))


def test_model(model, datamodule, trainer):
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)
    print(f"Test results for {model.__class__.__name__}: {test_results}")
    # Assuming the default key for loss in the test results is 'test_loss'
    # Find the key containing the loss
    possible_keys = ["test_loss_epoch", "loss", "test_loss"]
    for key in possible_keys:
        if key in test_results[0]:
            return test_results[0][key]
    raise KeyError("No recognized loss key found in test results")


def train_stitching_layer_and_model2_part2(
    stitching_model, datamodule, trainer, num_epochs
):
    # Freeze model1 parameters
    for param in stitching_model.part1_model1.parameters():
        param.requires_grad = False

    # Train only the stitching layer and the second part of model2
    trainer.fit(stitching_model, datamodule=datamodule, max_epochs=num_epochs)



def initialize_trainer(logger, num_epochs, devices="auto"):
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
    )

    return pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=devices,
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if torch.cuda.device_count() > 1
            else None
        ),
        callbacks=[lr_monitor, early_stopping],
    )


def main(
    model1_name,
    model2_name,
    split1,
    split2,
    log_dir,
    data_dir,
    num_epochs,
    enable_learning,
    lambda_model2: list[float],
    devices,
):
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = TensorBoardLogger("lightning_logs")
    trainer = initialize_trainer(logger, num_epochs, devices=devices)

    cifar10_data = CIFAR10Data(
        data_dir=data_dir, batch_size=32, num_workers=4, pin_memory=True
    )
    cifar10_data.prepare_data()
    cifar10_data.setup(stage="fit")

    model1 = load_model(model1_name, log_dir)
    model2 = load_model(model2_name, log_dir)
    stitching_model = StitchingModel(
        model1.model,
        model2.model,
        split1,
        split2,
        learning_rate=1e-3,
        enable_learning=enable_learning,
    )

    analysis_dir = Path(log_dir) / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    results_file = (
        analysis_dir / f"results_{model1_name}_{model2_name}_{split1}_{split2}.pth"
    )

    try:
        results = torch.load(results_file, map_location="cpu")
        stitching_model.load_state_dict(results["init"]["state_dict"])
    except (FileNotFoundError, KeyError):
        results = {}

    if "init" not in results:
        results["init"] = snapshot(stitching_model, cifar10_data, trainer)
        torch.save(results, results_file)

    if "after_regression" not in results:
        stitching_model.load_state_dict(results["init"]["state_dict"])
        do_linear_regression(stitching_model, cifar10_data)
        results["after_regression"] = snapshot(stitching_model, cifar10_data, trainer)
        torch.save(results, results_file)

    if "after_training" not in results:
        stitching_model.load_state_dict(results["after_regression"]["state_dict"])
        trainer.fit(stitching_model, datamodule=cifar10_data)
        results["after_training"] = snapshot(stitching_model, cifar10_data, trainer)
        torch.save(results, results_file)

    if enable_learning:
        for lam in lambda_model2:
            key = f"after_training_model2_part2_stitching_{lam:.3f}"
            if key not in results:
                # Train the stitching layer and the second part of model2
                stitching_model.load_state_dict(results["after_training"]["state_dict"])
                stitching_model.l2_lambda = lam
                train_stitching_layer_and_model2_part2(
                    stitching_model, cifar10_data, trainer, num_epochs
                )
                results[key] = snapshot(stitching_model, cifar10_data, trainer)
                torch.save(results, results_file)

    # Log model architecture
    batch = next(iter(cifar10_data.train_dataloader()))[0].to(
        next(stitching_model.parameters()).device
    )
    logger.experiment.add_graph(stitching_model, batch)

    # Reshape weights for embedding
    conv_weights = stitching_model.stitching_layer.conv.weight.data
    conv_weights_reshaped = conv_weights.view(conv_weights.size(0), -1)
    logger.experiment.add_embedding(
        conv_weights_reshaped, metadata=None, label_img=None
    )


if __name__ == "__main__":
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
        "--data_dir",
        type=Path,
        required=True,
        help="Directory to store data",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--enable_learning",
        type=bool,
        default=False,
        help="Enable learning of the stitching layer and model 2 part 2",
    )
    parser.add_argument(
        "--lambda_model2",
        type=float,
        nargs="+",
        required=True,
        help="List of lambda values for regularization",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Device to train the model on. Use 'auto' for auto detection",
    )
    args = parser.parse_args()

    main(
        args.model1_name,
        args.model2_name,
        args.index1,
        args.index2,
        args.log_dir,
        args.data_dir,
        args.num_epochs,
        args.enable_learning,
        args.lambda_model2,
        args.devices,
    )
