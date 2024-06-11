import os
import certifi
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from stitching_layer import StitchingModel  # Assuming this is your custom module
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import logging
import argparse


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    training_logger = logging.getLogger("training")
    testing_logger = logging.getLogger("testing")
    comparison_logger = logging.getLogger("comparison")

    # Create handlers
    training_handler = logging.FileHandler("training.log")
    testing_handler = logging.FileHandler("testing.log")
    comparison_handler = logging.FileHandler("comparison.log")

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
    for epoch in trange(num_epochs, desc="Training Epochs"):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        if logger:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")


def test(model, test_loader, criterion, device, logger=None):
    model.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    if logger:
        logger.info(f"Test Loss: {avg_loss}, Accuracy: {accuracy}%")
    print(f"Test Loss: {avg_loss}, Accuracy: {accuracy}%")
    return avg_loss, accuracy


def main(model1_name, model2_name, index1, index2, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_logger, testing_logger, comparison_logger = setup_logging()
    print("Checkpoint 1")
    train_loader, test_loader = load_dataset(batch_size=batch_size)
    print("Checkpoint 2")
    stitching_model = StitchingModel(model1_name, model2_name, index1, index2).to(
        device
    )

    if torch.cuda.device_count() > 1:
        stitching_model = nn.DataParallel(stitching_model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    print("Checkpoint 3")
    # Take only one batch of data for initializing the stitching layer
    images, _ = next(iter(train_loader))
    images = images.to(device)
    with torch.no_grad():
        part1_output = stitching_model.module.part1_model1(images)
        part2_input = stitching_model.module.part2_model2(part1_output)
        stitching_model.module.stitching_layer.initialize_weights_with_regression(
            part1_output.cpu(), part2_input.cpu()
        )
    del part1_output, part2_input, images
    torch.cuda.empty_cache()

    print("Checkpoint 4")
    # Move the model to the device again to ensure everything is on the correct device
    stitching_model.to(device)

    # Refine the stitching layer by gradient descent
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model.parameters(), lr=0.001)

    train(
        stitching_model,
        train_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        logger=training_logger,
    )
    print("Checkpoint 5")
    # Test the stitched model
    stitched_loss, stitched_accuracy = test(
        stitching_model, test_loader, criterion, device, logger=testing_logger
    )
    print("Checkpoint 6")
    # Test Model 1 (Part 1 of model 1 followed by Part 2 of model 2)
    model1 = nn.Sequential(
        *list(stitching_model.module.part1_model1.children())
        + [stitching_model.module.part2_model2]
    ).to(device)
    model1 = nn.DataParallel(model1)
    model1_loss, model1_accuracy = test(
        model1, test_loader, criterion, device, logger=testing_logger
    )
    print("Checkpoint 7")
    # Plot comparison results
    plot_comparison(stitched_loss, stitched_accuracy, model1_loss, model1_accuracy)


def plot_comparison(stitched_loss, stitched_accuracy, model1_loss, model1_accuracy):
    labels = ["Stitched Model", "Model 1"]
    losses = [stitched_loss, model1_loss]
    accuracies = [stitched_accuracy, model1_accuracy]

    x = range(len(labels))

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Loss", color=color)
    ax1.bar(x, losses, color=color, alpha=0.7)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Accuracy (%)", color=color)
    ax2.plot(x, accuracies, color=color, marker="o", linestyle="-")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.xticks(x, labels)
    plt.title("Comparison of Stitched Model and Model 1")
    fig.tight_layout()
    plt.show()


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
