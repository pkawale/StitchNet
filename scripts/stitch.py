import argparse
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pytorch_lightning.loggers import TensorBoardLogger

from helper_scripts.CIFAR10Module import CIFAR10Module
from helper_scripts.StitchingModelTrainer import StitchingModelTrainer
from stitching_layer import StitchingModel
from helper_scripts.utils import find_checkpoint_for_model


# BEST-PRACTICES ISSUE: this is a global variable, which should be avoided.
results = {
    "init": {},
    "after_regression": {},
    "before_training": {},
    "after_training": {},
}


def snapshot(model, description):
    # BUG: this function is never called??
    # SUGGESTION: use this function to save not just the state dict, but also the loss/accuracy
    # metrics. This will make it much easier to make the plots later. We'll just load the snapshots
    # to compute ∆L.
    return {
        "description": description,
        "state_dict": model.state_dict(),
    }


def load_models(model1_name, model2_name, log_dir):
    # BEST-PRACTICES ISSUE: should probably be a 'load_model' function and load one at a time, and
    # put this somewhere that other files can access it. Loading a model will be a common operation
    # going forward. Loading two models is a special case.
    checkpoint_path = find_checkpoint_for_model(log_dir, model1_name)
    print("[INFO]: loading model1 from", checkpoint_path)
    model1 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    checkpoint_path = find_checkpoint_for_model(log_dir, model2_name)
    print("[INFO]: loading model2 from", checkpoint_path)
    model2 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    return model1, model2


def do_linear_regression(stitching_model, device):
    # BUG: using a different transform here vs in the train function. Use DRY principle and
    # refactor to re-use the same datamodule everywhere! Maybe the data module should be an
    # argument to this function?
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # BUG: (same as above, essentially): don't hard-code the data path.
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    batch_im, _ = next(iter(train_loader))
    batch_im = batch_im.to(device)
    stitching_model.initialize_stitching_layer(batch_im)

def main(model1_name, model2_name, split1, split2, log_dir, num_epochs):
    # BEST-PRACTICES ISSUE: let lightning handle devices for you.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1, model2 = load_models(model1_name, model2_name, log_dir)
    stitching_model = StitchingModel(model1, model2, split1, split2).to(device)

    log_dir = Path(log_dir) / "checkpoints"
    log_dir.mkdir(exist_ok=True, parents=True)

    # BEST-PRACTICES ISSUE: you already have a DataModule in CIFAR10Data.py; You're repeating
    # yourself here by defining the dataloaders again inside the LightningModule. It's good practice
    # to separate the data and model wherever possible.
    cifar10_module = CIFAR10Module(model_name="resnet18", learning_rate=1e-3, weight_decay=1e-4)
    train_loader = cifar10_module.train_dataloader()
    val_loader = cifar10_module.val_dataloader()
    test_loader = cifar10_module.test_dataloader()

    # QUESTION: will this create a different log directory for each combination of model1, model2,
    # split1, split2? If not, how will you keep track of the different experiments?
    logger = TensorBoardLogger(save_dir=log_dir, name='stitching_model')

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else None
    )

    # BEST-PRACTICES ISSUE: 'StitchingModelTrainer' is of type 'LightningModule'. I find this naming
    # confusing because lightning already has a 'Trainer' class.
    stitching_trainer = StitchingModelTrainer(stitching_model, learning_rate=1e-3)

    trainer.fit(stitching_trainer, train_loader, val_loader)
    trainer.test(stitching_trainer, test_loader)

    # BUG: you're not taking snapshot()s, and you're not calling do_linear_regression() anywhere.


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
        type=int, required=True, help="Split Index of the layer in the first model",
    )
    parser.add_argument(
        "--index2",
        type=int, required=True, help="Split Index of the layer in the second model",
    )
    parser.add_argument(
        "--log_dir",
        type=Path, required=True, help="Directory to store logs and checkpoints",
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
