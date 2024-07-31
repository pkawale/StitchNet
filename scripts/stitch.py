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


def save_loss(
    model1,
    model2,
    model1_name,
    model2_name,
    stitching_model,
    cifar10_data,
    trainer,
):
    return {
        f"{model1_name}_loss": test_model(model1, cifar10_data, trainer),
        f"{model2_name}_loss": test_model(model2, cifar10_data, trainer),
        "stitching_model_loss": test_model(stitching_model, cifar10_data, trainer),
    }


def save_model_states(
    model1,
    model2,
    model1_name,
    model2_name,
    stitching_model,
    cifar10_data,
    trainer,
    stage,
) -> dict:
    return {
        "state_dict": stitching_model.state_dict(),
        "losses": save_loss(
            model1,
            model2,
            model1_name,
            model2_name,
            stitching_model,
            cifar10_data,
            trainer,
        ),
    }


def load_model(model_name, log_dir):
    checkpoint_path = find_checkpoint_for_model(log_dir, model_name)
    print(f"[INFO]: loading {model_name} from", checkpoint_path)
    model = CIFAR10Module.load_from_checkpoint(checkpoint_path)
    return model


def do_linear_regression(stitching_model, datamodule):
    train_loader = datamodule.train_dataloader()
    batch_im, _ = next(iter(train_loader))
    batch_im = batch_im.to(
        next(stitching_model.parameters()).device
    )  # Ensure batch_im is on the same device
    stitching_model.initialize_stitching_layer(batch_im)


def test_model(model, datamodule, trainer):
    # test_module = StitchingModel(model, learning_rate=1e-3)
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

    # # Create a module for the stitching model
    # stitching_module = StitchingModel(
    #     stitching_model, learning_rate=1e-3, enable_learning=True
    # )

    # Train only the stitching layer and the second part of model2
    trainer.fit(stitching_model, datamodule=datamodule)


def save_results(results, log_dir, stage, model1, model2, split1, split2):
    torch.save(
        results, log_dir / f"results_{stage}_{model1}_{model2}_{split1}_{split2}.pth"
    )


def load_results(log_dir, stage, model1, model2, split1, split2):
    results_path = log_dir / f"results_{stage}_{model1}_{model2}_{split1}_{split2}.pth"
    if results_path.exists():
        return torch.load(results_path)
    return None


def initialize_trainer(logger, num_epochs):
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
    )

    return pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=1,
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
):
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = TensorBoardLogger("lightning_logs")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=1,
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=[lr_monitor, early_stopping],
    )

    cifar10_data = CIFAR10Data(
        data_dir=data_dir, batch_size=32, num_workers=4, pin_memory=True
    )
    cifar10_data.prepare_data()
    cifar10_data.setup(stage="fit")

    results = (
        load_results(log_dir, "init", model1_name, model2_name, split1, split2) or {}
    )

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

    log_dir = Path(log_dir) / "analysis"
    log_dir.mkdir(exist_ok=True)

    if "init" not in results:
        results["init"] = save_model_states(
            model1,
            model2,
            model1_name,
            model2_name,
            stitching_model,
            cifar10_data,
            trainer,
            "init",
        )
        save_results(results, log_dir, "init", model1_name, model2_name, split1, split2)

    do_linear_regression(stitching_model, cifar10_data)

    if "after_regression" not in results:
        results["after_regression"] = save_model_states(
            model1,
            model2,
            model1_name,
            model2_name,
            stitching_model,
            cifar10_data,
            trainer,
            "after_regression",
        )
        save_results(
            results,
            log_dir,
            "after_regression",
            model1_name,
            model2_name,
            split1,
            split2,
        )

    if "before_training" not in results:
        results["before_training"] = save_model_states(
            model1,
            model2,
            model1_name,
            model2_name,
            stitching_model,
            cifar10_data,
            trainer,
            "before_training",
        )
        save_results(
            results,
            log_dir,
            "before_training",
            model1_name,
            model2_name,
            split1,
            split2,
        )

    trainer.fit(stitching_model, datamodule=cifar10_data)

    if "after_training" not in results:
        results["after_training"] = save_model_states(
            model1,
            model2,
            model1_name,
            model2_name,
            stitching_model,
            cifar10_data,
            trainer,
            "after_training",
        )
        save_results(
            results, log_dir, "after_training", model1_name, model2_name, split1, split2
        )

    if enable_learning:
        for lam in lambda_model2:
            if "after_training_model2_part2_stitching_{lam:.3f}" not in results:
                # Train the stitching layer and the second part of model2
                stitching_model.load_state_dict(results["after_training"]["state_dict"])
                train_stitching_layer_and_model2_part2(
                    stitching_model, cifar10_data, trainer, num_epochs
                )

                results[f"after_training_model2_part2_stitching_{lam:.3f}"] = (
                    save_model_states(
                        model1,
                        model2,
                        model1_name,
                        model2_name,
                        stitching_model,
                        cifar10_data,
                        trainer,
                        f"after_training_model2_part2_stitching_{lam:.3f}",
                    )
                )
                results[f"after_training_model2_part2_stitching_{lam:.3f}"][
                    "lambda"
                ] = lam
                save_results(
                    results,
                    log_dir,
                    f"after_training_model2_part2_stitching_{lam:.3f}",
                    model1_name,
                    model2_name,
                    split1,
                    split2,
                )

    if "losses" not in results:
        # Test models and capture losses
        results["losses"] = save_loss(
            model1,
            model2,
            model1_name,
            model2_name,
            stitching_model,
            cifar10_data,
            trainer,
        )
        save_results(
            results, log_dir, "losses", model1_name, model2_name, split1, split2
        )

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

    print(
        "Resulats path:",
        log_dir / f"results_{model1_name}_{model2_name}_{split1}_{split2}.pth",
    )
    # Save results
    torch.save(
        results, log_dir / f"results_{model1_name}_{model2_name}_{split1}_{split2}.pth"
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
    )
