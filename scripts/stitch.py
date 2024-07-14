import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from helper_scripts.CIFAR10Module import CIFAR10Module
from helper_scripts.StitchingModelModule import StitchingModelTrainer
from stitching_layer import StitchingModel
from helper_scripts.utils import find_checkpoint_for_model, load_dataset

results = {
    "init": {},
    "after_regression": {},
    "before_training": {},
    "after_training": {},
}


def snapshot(model, description):
    return {
        "description": description,
        "state_dict": model.state_dict(),
    }


def load_models(model1_name, model2_name, log_dir):
    checkpoint_path = find_checkpoint_for_model(log_dir, model1_name)
    print("[INFO]: loading model1 from", checkpoint_path)
    model1 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    checkpoint_path = find_checkpoint_for_model(log_dir, model2_name)
    print("[INFO]: loading model2 from", checkpoint_path)
    model2 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    return model1, model2


def do_linear_regression(stitching_model):
    train_loader = load_dataset(
        batch_size=32, num_workers=4, pin_memory=True, train=True
    )
    batch_im, _ = next(iter(train_loader))
    stitching_model.initialize_stitching_layer(batch_im)

def main(model1_name, model2_name, split1, split2, log_dir, num_epochs):

    model1, model2 = load_models(model1_name, model2_name, log_dir)
    stitching_model = StitchingModel(model1, model2, split1, split2)

    log_dir = Path(log_dir) / "checkpoints"
    log_dir.mkdir(exist_ok=True, parents=True)

    cifar10_module = CIFAR10Module(
        model_name="resnet18", learning_rate=1e-3, weight_decay=1e-4
    )
    train_loader = cifar10_module.train_dataloader()
    val_loader = cifar10_module.val_dataloader()
    test_loader = cifar10_module.test_dataloader()

    do_linear_regression(stitching_model)
    logger = TensorBoardLogger(save_dir=log_dir, name="stitching_model")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if torch.cuda.device_count() > 1
            else None
        ),
    )

    stitching_trainer = StitchingModelTrainer(stitching_model, learning_rate=1e-3)

    trainer.fit(stitching_trainer, train_loader, val_loader)
    trainer.test(stitching_trainer, test_loader)
    # Log model architecture and embeddings
    logger.experiment.add_graph(stitching_model, next(iter(train_loader))[0])
    logger.experiment.add_embedding(
        stitching_model.stitching_layer.weight.data, metadata=None, label_img=None
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
