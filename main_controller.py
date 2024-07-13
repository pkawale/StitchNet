import argparse
import torch
from torch import nn, optim
from torchviz import make_dot
from tqdm import tqdm
from glob import glob
from pathlib import Path
from helper_scripts.CIFAR10Module import CIFAR10Module
from stitching_layer import StitchingModel


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    logger=None,
    scheduler=None,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        if scheduler:
            scheduler.step(epoch_loss)
        if logger:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return epoch_losses


def test(model, test_loader, criterion, device, logger=None):
    model.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
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
        logger.info(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def measure_stitching_penalty(
    model1,
    model2,
    model1_name,
    model2_name,
    device,
    train_loader,
    test_loader,
    criterion,
    num_epochs=3,
):
    num_layers_model1 = len(list(model1.children()))
    num_layers_model2 = len(list(model2.children()))
    penalties = []

    for split_fraction in range(
        1, 9
    ):  # split based on number of residual blocks for resnet18
        split1 = max(1, int(split_fraction * num_layers_model1 / num_layers_model1))
        split2 = max(1, int(split_fraction * num_layers_model2 / num_layers_model2))

        print(f"Split point for model1: {split1}, Split point for model2: {split2}")
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5, verbose=True
        )

        train(
            stitching_model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_epochs,
            scheduler=scheduler,
        )
        test_loss, _ = test(stitching_model, test_loader, criterion, device)
        penalties.append(test_loss)

    return penalties


def visualize_model(model, inputs):
    y = model(inputs)
    make_dot(y, params=dict(list(model.named_parameters()))).render(
        "model_graph", format="png"
    )


def stitching_test_same_model(
    model1_name, num_epochs, device, data_module, training_logger, testing_logger
):
    # Assumed values of i and j for the test case
    i = 3  # Example layer index for model1
    j = 3  # Example layer index for model1
    print(f"Stitching layer {i} of {model1_name} to layer {j} of the same model")

    stitching_model = StitchingModel(model1_name, model1_name, i, j).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model.parameters(), lr=0.001)

    images, _ = next(iter(data_module.train_dataloader()))
    images = images.to(device)
    stitching_model.initialize_stitching_layer(images)
    stitching_model = stitching_model.to(device)

    stitched_train_losses = train(
        stitching_model,
        data_module.train_dataloader(),
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        logger=training_logger,
    )
    stitched_test_loss, stitched_accuracy = test(
        stitching_model,
        data_module.val_dataloader(),
        criterion,
        device,
        logger=testing_logger,
    )
    print(f"Expected Loss: 0, Actual Test Loss: {stitched_test_loss}")


def find_checkpoint_for_model(log_dir: Path, model_name: str) -> Path:
    checkpoint_dir = list(glob(str(log_dir / f"{model_name}_*" / "checkpoints")))
    if len(checkpoint_dir) == 0:
        raise ValueError(f"Model {model_name} not found in {log_dir}")
    elif len(checkpoint_dir) > 1:
        raise ValueError(f"Multiple models found in {log_dir}")
    checkpoint_dir = Path(checkpoint_dir[0])
    list_of_files = list(checkpoint_dir.glob("*.ckpt"))
    if len(list_of_files) == 0:
        raise ValueError(f"No checkpoint found for {model_name}")
    elif len(list_of_files) > 1:
        # TODO - return 'best' or 'last' or let user pick
        raise ValueError(f"Multiple checkpoints found for {model_name}")
    return list_of_files[0]


def main(
    model1_name,
    model2_name,
    index1,
    index2,
    log_dir,
):
    # we are pretraining model1 and model2 and saving them in the checkpoints in train.py
    # we are loading the pre-trained model1 and model2 here
    # Load the pre-existing model from log_dir

    checkpoint_path = find_checkpoint_for_model(log_dir, model1_name)
    print("[INFO]: loading model1 from", checkpoint_path)
    model1 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    checkpoint_path = find_checkpoint_for_model(log_dir, model2_name)
    print("[INFO]: loading model2 from", checkpoint_path)
    model2 = CIFAR10Module.load_from_checkpoint(checkpoint_path)

    stitching_model = StitchingModel(model1_name, model2_name, index1, index2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitching Model Training Script")

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

    args = parser.parse_args()

    main(
        args.model1_name,
        args.model2_name,
        args.index1,
        args.index2,
        args.log_dir,
    )
