import os
import certifi
import timm
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from stitching_layer import StitchingModel
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import logging
import argparse


def setup_logging(model1, model2, index1, index2):
    logging.basicConfig(level=logging.INFO)
    training_logger = logging.getLogger("training")
    testing_logger = logging.getLogger("testing")
    comparison_logger = logging.getLogger("comparison")

    # Create handlers
    training_handler = logging.FileHandler(
        f"training_{model1}_{model2}_{index1}_{index2}.log"
    )
    testing_handler = logging.FileHandler(
        f"testing_{model1}_{model2}_{index1}_{index2}.log"
    )
    comparison_handler = logging.FileHandler(
        f"comparison_{model1}_{model2}_{index1}_{index2}.log"
    )

    # Set level for handlers
    training_handler.setLevel(logging.INFO)
    testing_handler.setLevel(logging.INFO)
    comparison_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    training_handler.setFormatter(formatter)
    testing_handler.setFormatter(formatter)
    comparison_handler.setFormatter(formatter)

    # Add handlers to loggers
    training_logger.addHandler(training_handler)
    testing_logger.addHandler(testing_handler)
    comparison_logger.addHandler(comparison_handler)

    return training_logger, testing_logger, comparison_logger


os.environ["SSL_CERT_FILE"] = (
    certifi.where()
    if "SSL_CERT_FILE" not in os.environ
    else os.environ["SSL_CERT_FILE"]
)


def load_dataset(batch_size=64, num_workers=4, pin_memory=True):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cifar10_train = datasets.CIFAR10(
        root=os.path.join(os.getenv("DATA_DIR", "data"), "cifar10"),
        train=True,
        download=True,
        transform=transform,
    )
    cifar10_test = datasets.CIFAR10(
        root=os.path.join(os.getenv("DATA_DIR", "data"), "cifar10"),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        cifar10_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train(
    model, train_loader, criterion, optimizer, device, num_epochs=10, logger=None
):
    model.train()
    epoch_losses = []
    for epoch in trange(num_epochs, desc="Training Epochs"):
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc="training", total=len(train_loader)
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        if logger:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
    return epoch_losses


def test(model, test_loader, criterion, device, num_epochs=10, logger=None):
    epoch_losses = []
    for epoch in trange(num_epochs, desc="Testing Epochs"):
        model.eval()
        total = 0
        correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(
                test_loader, desc="Testing", total=len(test_loader)
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(test_loader)
        epoch_losses.append(avg_loss)
        if logger:
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}] Test Loss: {avg_loss}, Accuracy: {accuracy}%"
            )
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Test Loss: {avg_loss}, Accuracy: {accuracy}%"
        )
    return avg_loss, accuracy, epoch_losses


def main(model1_name, model2_name, index1, index2, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_logger, testing_logger, comparison_logger = setup_logging(
        model1_name, model2_name, index1, index2
    )
    train_loader, test_loader = load_dataset(batch_size=batch_size)

    stitching_model = StitchingModel(model1_name, model2_name, index1, index2).to(
        device
    )

    # Take only one batch of data for initializing the stitching layer
    images, _ = next(iter(train_loader))
    images = images.to(device)

    # Initialize the stitching layer
    stitching_model.initialize_stitching_layer(images)

    # Move the model to the device again to ensure everything is on the correct device
    stitching_model.to(device)

    # Refine the stitching layer by gradient descent
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model.parameters(), lr=0.001)

    # Train the stitched model
    stitched_train_losses = train(
        stitching_model,
        train_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        logger=training_logger,
    )

    # Test the stitched model
    stitched_test_loss, stitched_accuracy, stitched_test_losses = test(
        stitching_model,
        test_loader,
        criterion,
        device,
        num_epochs=num_epochs,
        logger=testing_logger,
    )

    # Load Model 1 for training and testing (without stitching)
    model1 = timm.create_model(model1_name, pretrained=True).to(device)
    optimizer_model1 = optim.Adam(model1.parameters(), lr=0.001)

    # Train the original Model 1
    model1_train_losses = train(
        model1,
        train_loader,
        criterion,
        optimizer_model1,
        device,
        num_epochs=num_epochs,
        logger=training_logger,
    )

    # Test the original Model 1
    model1_test_loss, model1_accuracy, model1_test_losses = test(
        model1,
        test_loader,
        criterion,
        device,
        num_epochs=num_epochs,
        logger=testing_logger,
    )

    # Plot comparison results
    plot_comparison(
        stitched_test_loss, stitched_accuracy, model1_test_loss, model1_accuracy
    )

    # Calculate and plot loss difference
    plot_loss_difference(stitched_test_loss, model1_test_loss)

    # Plot the training and testing losses
    plot_losses(
        stitched_train_losses,
        stitched_test_losses,
        model1_train_losses,
        model1_test_losses,
        model1_name,
        model2_name,
        index1,
        index2,
    )


def plot_losses(
    stitched_train_losses,
    stitched_test_losses,
    model1_train_losses,
    model1_test_losses,
    model1_name,
    model2_name,
    index1,
    index2,
):
    epochs = range(1, len(stitched_train_losses) + 1)

    plt.figure(figsize=(12, 8))

    # Plot training losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, stitched_train_losses, "bo-", label="Stitched Model Training Loss")
    plt.plot(epochs, model1_train_losses, "ro-", label="Model 1 Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True)

    # Plot testing losses
    plt.subplot(2, 1, 2)
    plt.plot(epochs, stitched_test_losses, "bo-", label="Stitched Model Testing Loss")
    plt.plot(epochs, model1_test_losses, "ro-", label="Model 1 Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Testing Loss")
    plt.legend()
    plt.grid(True)

    plt.suptitle(
        f"Training and Testing Losses for Stitched Model and Model 1 ({model1_name}, {model2_name})"
    )
    plt.tight_layout()
    plt.savefig(
        f"training_testing_losses_{model1_name}_{model2_name}_{index1}_{index2}.png"
    )
    plt.close()


def plot_comparison(stitched_loss, stitched_accuracy, model1_loss, model1_accuracy):
    labels = ["Stitched Model", "Model 1"]
    losses = [stitched_loss, model1_loss]
    accuracies = [stitched_accuracy, model1_accuracy]

    x = range(len(labels))

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Loss", color=color)
    bars = ax1.bar(x, losses, color=color, alpha=0.7)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, max(losses) * 1.1)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Accuracy (%)", color=color)
    lines = ax2.plot(x, accuracies, color=color, marker="o", linestyle="-")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 100)

    plt.xticks(x, labels)
    plt.title("Comparison of Stitched Model and Model 1")
    fig.tight_layout()

    # Annotate bars with loss values
    for bar, loss in zip(bars, losses):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            round(loss, 4),
            va="bottom",
            ha="center",
        )

    # Annotate lines with accuracy values
    for i, acc in enumerate(accuracies):
        ax2.text(i, acc, f"{acc:.2f}%", color=color, va="bottom", ha="center")

    plt.savefig("compare_stitched_and_model_1.png")
    plt.close()


def plot_loss_difference(stitched_loss, model1_loss):
    abs_difference = abs(stitched_loss - model1_loss)

    plt.figure(figsize=(8, 6))
    bar = plt.bar(["Loss Difference"], [abs_difference], color="orange")
    plt.ylabel("Absolute Loss Difference")
    plt.title("Absolute Loss Difference between Stitched Model and Model 1")

    # Annotate bar with loss difference value
    yval = bar[0].get_height()
    plt.text(
        bar[0].get_x() + bar[0].get_width() / 2.0,
        yval,
        round(abs_difference, 4),
        va="bottom",
        ha="center",
    )

    plt.savefig("absolute_loss_comparison.png")
    plt.close()


def compare_outputs(output1, output2, logger=None):
    difference = torch.abs(output1 - output2)
    if logger:
        logger.info(
            f"Difference between stitched model output and Model 1 output: {difference}"
        )
    print("\nDifference between stitched model output and Model 1 output:")
    print(difference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitching Model Training Script")
    parser.add_argument(
        "--model1_name", type=str, required=True, help="Name of the first model"
    )
    parser.add_argument(
        "--model2_name", type=str, required=True, help="Name of the second model"
    )
    parser.add_argument(
        "--index1", type=int, required=True, help="Split index for the first model"
    )
    parser.add_argument(
        "--index2", type=int, required=True, help="Split index for the second model"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--pin_memory", action="store_true", help="Use pinned memory for data loading"
    )

    args = parser.parse_args()

    main(
        args.model1_name,
        args.model2_name,
        args.index1,
        args.index2,
        args.num_epochs,
        args.batch_size,
    )
