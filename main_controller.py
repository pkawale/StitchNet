import argparse
import os
import torch
import timm
from torch import nn, optim
from torchviz import make_dot
from tqdm import tqdm
from stitching_layer import StitchingModel
from utils import setup_logging
from plotter import plot_stitching_penalty


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


def main(
    model1_name,
    model2_name,
    index1,
    index2,
    num_epochs,
    batch_size,
    num_workers,
    pin_memory,
    data_dir,
    pretrained,
    test_phase,
    dev,
    precision,
    learning_rate,
    weight_decay,
):

    # TODO - load pretrained models

    # Original stitching code starts here

    training_logger, testing_logger, comparison_logger = setup_logging(
        model1_name, model2_name
    )

    # Use the already trained model1
    model1 = model.model
    model1 = model1.to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"Using trained {model1_name}")
    _, accuracy1 = test(
        model1,
        data_module.val_dataloader(),
        criterion,
        device,
        logger=testing_logger,
    )
    assert accuracy1 > 80, "Model 1 is not learning; check the model and data"

    # Train and test the initial model2
    model2 = timm.create_model(model2_name, pretrained=False, num_classes=10).to(device)
    optimizer = optim.AdamW(
        model2.parameters(), lr=learning_rate, weight_decay=weight_decay
    )  # Use same optimizer and parameters
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    print(f"Training {model2_name}")
    train(
        model2,
        data_module.train_dataloader(),
        criterion,
        optimizer,
        device,
        num_epochs,
        logger=training_logger,
        scheduler=scheduler2,
    )
    _, accuracy2 = test(
        model2,
        data_module.val_dataloader(),
        criterion,
        device,
        logger=testing_logger,
    )
    assert accuracy2 > 80, "Model 2 is not learning; check the model and data"

    # Save initial parameters
    initial_params_model1 = [p.clone() for p in model1.parameters()]
    initial_params_model2 = [p.clone() for p in model2.parameters()]

    images = next(iter(data_module.train_dataloader()))[0].to(device)

    stitching_model = StitchingModel(model1_name, model2_name, index1, index2).to(
        device
    )

    # Visualize the stitching model
    visualize_model(stitching_model, images)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model.parameters(), lr=0.001)

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
    penalties = measure_stitching_penalty(
        model1,
        model2,
        model1_name,
        model2_name,
        device,
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        criterion,
        num_epochs,
    )
    plot_stitching_penalty(penalties, model1_name, model2_name)

    stitching_test_same_model(model1_name, num_epochs, device, data_module, training_logger, testing_logger)

    # Assert that parameters of model1 and model2 haven't changed
    for original, new in zip(initial_params_model1, model1.parameters()):
        assert torch.equal(original, new), "Model 1 parameters changed after stitching"
    for original, new in zip(initial_params_model2, model2.parameters()):
        assert torch.equal(original, new), "Model 2 parameters changed after stitching"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitching Model Training Script")

    # PROGRAM level args
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory to store CIFAR-10 data"
    )
    parser.add_argument(
        "--download_weights",
        type=int,
        default=0,
        choices=[0, 1],
        help="Download pretrained weights",
    )
    parser.add_argument(
        "--test_phase", type=int, default=0, choices=[0, 1], help="Test phase flag"
    )
    parser.add_argument(
        "--dev", type=int, default=0, choices=[0, 1], help="Development mode flag"
    )

    # TRAINER args
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
        "--pretrained", type=int, default=0, choices=[0, 1], help="Use pretrained model"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Precision for training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
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
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
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
        args.data_dir,
        args.pretrained,
        args.test_phase,
        args.dev,
        args.precision,
        args.learning_rate,
        args.weight_decay,
    )
