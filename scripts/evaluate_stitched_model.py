import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from helper_scripts.CIFAR10Module import CIFAR10Module
from stitching_layer import StitchingModel
from helper_scripts.utils import find_checkpoint_for_model
from helper_scripts.CIFAR10Data import CIFAR10Data


def load_model(model_name, log_dir, device):
    checkpoint_path = find_checkpoint_for_model(log_dir, model_name)
    print(f"[INFO]: loading {model_name} from", checkpoint_path)
    model = CIFAR10Module.load_from_checkpoint(checkpoint_path)
    model.to(device)  # Move the model to the specified device
    return model


def load_stitched_model(model1, model2, split1, split2, log_dir, device):
    stitched_model = StitchingModel(model1, model2, split1, split2)
    checkpoint_path = log_dir / "checkpoints" / "results.pth"
    results = torch.load(
        checkpoint_path, map_location=device
    )  # Ensure the loaded state_dict is on the correct device
    if "after_training" not in results:
        raise KeyError("The key 'after_training' is not found in the results file.")

    stitching_layer_state_dict = results["after_training"]["stitching_model_state_dict"]
    stitched_model.stitching_layer.load_state_dict(stitching_layer_state_dict)
    stitched_model.to(device)  # Move the stitched model to the specified device
    return stitched_model, results


def calculate_metrics(model, dataloader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_losses = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            batch_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, batch_losses


def visualize_results(results,model_name, comparison_type="model1_vs_stitched"):
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))

    # Loss Comparison
    ax[0, 0].bar(results["losses"].keys(), results["losses"].values())
    ax[0, 0].set_title(f"Loss Comparison: {comparison_type}")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].set_xlabel("Model")

    # Accuracy Comparison
    ax[0, 1].bar(
        results["accuracies"].keys(), results["accuracies"].values(), color="orange"
    )
    ax[0, 1].set_title(f"Accuracy Comparison: {comparison_type}")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].set_xlabel("Model")

    # Heatmap of Stitching Layer Weights
    conv_weights = (
        results["after_training"]["stitching_model_state_dict"]["conv.weight"]
        .cpu()
        .numpy()
    )
    sns.heatmap(
        conv_weights.reshape(-1, conv_weights.shape[-1]), cmap="viridis", ax=ax[1, 0]
    )
    ax[1, 0].set_title("Stitching Layer Weights")

    # Batch-wise Loss Comparison
    ax[1, 1].plot(
        results["batch_losses"]["model1"], label=f"{model_name} Loss", linestyle="--"
    )
    ax[1, 1].plot(
        results["batch_losses"]["stitched_model"],
        label="Stitched Model Loss",
        linestyle="-",
    )
    ax[1, 1].set_title("Batch-wise Loss Comparison")
    ax[1, 1].set_ylabel("Loss")
    ax[1, 1].set_xlabel("Batch")
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()


def main(model1_name, model2_name, split1, split2, log_dir):
    results = {"losses": {}, "accuracies": {}, "batch_losses": {}}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = load_model(model1_name, log_dir, device)
    model2 = load_model(model2_name, log_dir, device)
    stitched_model, results_data = load_stitched_model(
        model1, model2, split1, split2, log_dir, device
    )

    cifar10_data = CIFAR10Data(
        data_dir="./data", batch_size=32, num_workers=4, pin_memory=True
    )
    cifar10_data.prepare_data()
    cifar10_data.setup(stage="test")
    test_loader = cifar10_data.test_dataloader()

    # Model 1 vs Stitched Model
    model1_loss, model1_accuracy, model1_batch_losses = calculate_metrics(
        model1, test_loader, device
    )
    stitched_model_loss, stitched_model_accuracy, stitched_model_batch_losses = (
        calculate_metrics(stitched_model, test_loader, device)
    )

    results["losses"]["model1_loss"] = model1_loss
    results["losses"]["stitched_model_loss"] = stitched_model_loss
    results["accuracies"]["model1_accuracy"] = model1_accuracy
    results["accuracies"]["stitched_model_accuracy"] = stitched_model_accuracy
    results["batch_losses"]["model1"] = model1_batch_losses
    results["batch_losses"]["stitched_model"] = stitched_model_batch_losses
    results["after_training"] = results_data["after_training"]

    visualize_results(results, model1_name, comparison_type=f"{model1_name}_vs_stitched")

    # Model 2 vs Stitched Model
    model2_loss, model2_accuracy, model2_batch_losses = calculate_metrics(
        model2, test_loader, device
    )

    results["losses"]["model2_loss"] = model2_loss
    results["accuracies"]["model2_accuracy"] = model2_accuracy
    results["batch_losses"]["model2"] = model2_batch_losses

    # temporary update to add logs in stitch.py first
    # visualize_results(results, model2_name, comparison_type=f"{model2_name}_vs_stitched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stitched Model")
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
        help="Directory to load logs and checkpoints",
    )
    args = parser.parse_args()

    main(
        args.model1_name,
        args.model2_name,
        args.index1,
        args.index2,
        args.log_dir,
    )
