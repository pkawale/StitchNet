import argparse
import torch
import timm
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import trange, tqdm

from stitching_layer import StitchingModel
from utils import setup_logging, load_dataset
from plotter import plot_stitching_penalty


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    logger=None,
):
    model.train()
    scaler = GradScaler()
    epoch_losses = []

    for epoch in trange(num_epochs, desc="Training Epochs"):
        running_loss = torch.zeros(len(train_loader), device=device)
        optimizer.zero_grad()  # Reset gradients
        for i, (images, labels) in enumerate(
            tqdm(train_loader, desc="Training", total=len(train_loader))
        ):
            images, labels = images.to(device), labels.to(device)

            with autocast():
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward()

            running_loss[i] = loss.detach()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss = torch.mean(running_loss).item()
        epoch_losses.append(epoch_loss)
        if logger:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
    return epoch_losses


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


def measure_stitching_penalty(
    model1_name, model2_name, device, train_loader, test_loader, criterion, num_epochs=5
):
    num_layers_model1 = len(
        list(timm.create_model(model1_name, pretrained=True).children())
    )
    num_layers_model2 = len(
        list(timm.create_model(model2_name, pretrained=True).children())
    )
    penalties = []

    for split_fraction in range(1, num_layers_model1):
        split1 = max(1, int(split_fraction * num_layers_model1 / num_layers_model1))
        split2 = max(1, int(split_fraction * num_layers_model2 / num_layers_model2))

        try:
            stitching_model = StitchingModel(
                model1_name, model2_name, split1, split2
            ).to(device)
        except ValueError as e:
            print(f"Skipping invalid split configuration: {e}")
            continue

        images, _ = next(iter(train_loader))
        images = images.to(device)
        stitching_model.initialize_stitching_layer(images)
        stitching_model = stitching_model.to(device)  # Ensure model is on device

        optimizer = optim.Adam(stitching_model.parameters(), lr=0.001)

        train(stitching_model, train_loader, criterion, optimizer, device, num_epochs)
        test_loss, _ = test(stitching_model, test_loader, criterion, device)
        penalties.append(test_loss)

    return penalties


def main(
    model1_name,
    model2_name,
    split1,
    split2,
    num_epochs,
    batch_size,
    num_workers,
    pin_memory,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_dataset(
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )

    training_logger, testing_logger, comparison_logger = setup_logging(
        model1_name, model2_name
    )

    stitching_model = StitchingModel(model1_name, model2_name, split1, split2).to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model.parameters(), lr=0.001)

    stitched_train_losses = train(
        stitching_model,
        train_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        logger=training_logger,
    )
    stitched_test_loss, stitched_accuracy = test(
        stitching_model, test_loader, criterion, device, logger=testing_logger
    )
    penalties = measure_stitching_penalty(
        model1_name,
        model2_name,
        device,
        train_loader,
        test_loader,
        criterion,
        num_epochs,
    )
    plot_stitching_penalty(penalties, model1_name, model2_name)


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
        default=2,  # Reduced number of workers
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
        args.num_workers,
        args.pin_memory,
    )
