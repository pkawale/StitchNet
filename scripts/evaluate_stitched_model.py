from pathlib import Path

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from helper_scripts.CIFAR10Module import CIFAR10Module
from stitching_layer import StitchingModel
from helper_scripts.utils import find_checkpoint_for_model


def load_models(model1_name, model2_name, log_dir):
    checkpoint_path = find_checkpoint_for_model(log_dir, model1_name)
    print("[INFO]: loading model1 from", checkpoint_path)
    model1 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    checkpoint_path = find_checkpoint_for_model(log_dir, model2_name)
    print("[INFO]: loading model2 from", checkpoint_path)
    model2 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    return model1, model2


def load_stitched_model(stitched_model_path, model1, model2, split1, split2, device):
    stitching_model = StitchingModel(model1, model2, split1, split2)
    state_dict = torch.load(stitched_model_path, map_location=device)
    print(f"Keys in the loaded state dict: {state_dict.keys()}")

    # Adjust the key based on the actual state dict structure
    if 'model' in state_dict:
        stitching_model.load_state_dict(state_dict['model'])
    else:
        stitching_model.load_state_dict(state_dict)
    return stitching_model.to(device)


def calculate_loss(model1, stitched_model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model1.eval()
    stitched_model.eval()

    loss_model1 = 0.0
    loss_stitched = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs_model1 = model1(images)
            loss_model1 += criterion(outputs_model1, labels).item()

            outputs_stitched = stitched_model(images)
            loss_stitched += criterion(outputs_stitched, labels).item()

    avg_loss_model1 = loss_model1 / len(loader)
    avg_loss_stitched = loss_stitched / len(loader)

    return avg_loss_model1, avg_loss_stitched


def visualize_losses(loss_model1, loss_stitched):
    labels = ['Model1', 'Stitched Model']
    losses = [loss_model1, loss_stitched]

    plt.bar(labels, losses, color=['blue', 'orange'])
    plt.xlabel('Models')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Comparison')
    plt.show()


def main(model1_name, model2_name, split1, split2, log_dir, stitched_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1, model2 = load_models(model1_name, model2_name, log_dir)
    stitched_model = load_stitched_model(stitched_model_path, model1, model2, split1, split2, device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    loss_model1, loss_stitched = calculate_loss(model1, stitched_model, val_loader, device)
    visualize_losses(loss_model1, loss_stitched)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate and visualize loss for stitched model")
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
        "--stitched_model_path",
        type=Path,
        required=True,
        help="Path to the saved stitched model",
    )
    args = parser.parse_args()

    main(
        args.model1_name,
        args.model2_name,
        args.index1,
        args.index2,
        args.log_dir,
        args.stitched_model_path,
    )