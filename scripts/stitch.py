from safetensors import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

from CIFAR10Module import CIFAR10Module
from glob import glob
from pathlib import Path

from stitching_layer import StitchingModel


def train(
    model,
    train_loader,
    criterion,
    optimizer,
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
            images, labels = images, labels
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
            images, labels = images, labels
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
        # M3: TODO - return 'best' or 'last' or let user pick
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

    results = {}
    stitching_model = StitchingModel(model1, model2, index1, index2)

    # results before regression
    results["part1_model1_state_dict"] = stitching_model.part1_model1.state_dict()
    results["part2_model2_state_dict"] = stitching_model.part2_model2.state_dict()

    sample_input, _ = next(iter(CIFAR10Module.train_dataloader()))
    sample_input = sample_input[0]

    stitching_model.initialize_stitching_layer(sample_input)

    # results after regression and initialization in results
    results["part1_model1_state_dict_after_regression"] = (
        stitching_model.part1_model1.state_dict()
    )
    results["part2_model2_state_dict_after_regression"] = (
        stitching_model.part2_model2.state_dict()
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model.parameters(), lr=0.001)

    # TensorBoard logger
    log_folder = Path(log_dir) / "logs" / "stitched_model_{model1_name}_{model2_name}"
    log_folder.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_folder)

    # Training and testing the stitching model
    num_epochs = 10  # Set the number of epochs
    for epoch in range(num_epochs):
        train_loss = train(
            stitching_model,
            CIFAR10Module.train_dataloader(),
            criterion,
            optimizer,
            num_epochs=num_epochs,
            logger=writer,
        )
        test_loss, test_accuracy = test(
            stitching_model, CIFAR10Module.test_dataloader(), criterion, logger=writer
        )

        results[f"epoch_{epoch + 1}"] = {
            "train_loss": train_loss[-1],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

    # Save the final model
    torch.save(stitching_model.state_dict(), log_folder / "final_stitched_model.pt")
    writer.close()


if __name__ == "__main__":
    import argparse

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
