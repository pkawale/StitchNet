import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from stitching_layer import StitchingModel
from tqdm.auto import trange, tqdm
from config import DATA_DIR, DEVICE


def load_dataset(batch_size=64):
    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cifar10_train = datasets.CIFAR10(
        root=os.path.join(DATA_DIR, "cifar10"),
        train=True,
        download=True,
        transform=transform,
    )
    cifar10_test = datasets.CIFAR10(
        root=os.path.join(DATA_DIR, "cifar10"),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    test_loader = DataLoader(
        cifar10_test,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )
    return train_loader, test_loader


def train(model, train_loader, criterion, optimizer, num_epochs=10, device=DEVICE):
    model.train()
    model.to(device)
    for epoch in trange(num_epochs, desc="Training Epochs", position=0):
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader,
            desc="Training Batches",
            total=len(train_loader),
            leave=False,
            position=1,
        ):
            optimizer.zero_grad()
            outputs = model.forward(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}"
        )


def test(model, test_loader, criterion, device=DEVICE):
    model.eval()
    model.to(device)
    total = 0
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(
            test_loader, desc="Testing", total=len(test_loader), leave=False, position=1
        ):
            outputs = model.forward(images.to(device))
            loss = criterion(outputs, labels.to(device))
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f"Test Loss: {test_loss / len(test_loader)}, Accuracy: {100 * correct / total}%"
    )


def main():
    train_loader, test_loader = load_dataset()

    stitching_model = StitchingModel("resnet18", "resnet34", 5, 5)

    # Do a first pass over the data to initialize the stitching layer
    ...  # TODO

    # Refine the stitching layer by gradient descent
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model.parameters_stitching(), lr=0.001)

    train(stitching_model, train_loader, criterion, optimizer, num_epochs=10)
    test(stitching_model, test_loader, criterion)

    # Additional forward pass and comparison logic for analysis
    images, labels = next(iter(train_loader))
    outputs1, outputs2 = stitching_model.create_stitching_layer(images)

    print("\nOutputs of Model 1 layers:")
    for idx, output in enumerate(outputs1):
        print(f"Output of layer {idx}: {output.shape}")

    print("\nOutputs of Model 2 layers:")
    for idx, output in enumerate(outputs2):
        print(f"Output of layer {idx}: {output.shape}")

    # Forward pass through the stitching model
    final_output = stitching_model.forward(images)

    print("\nFinal output after stitching:")
    print(final_output.shape)

    # Compute outputs for comparison
    part1_output = stitching_model.part1_model1(images)
    stitched_output = stitching_model.stitching_layer(part1_output)
    final_output_model1 = stitching_model.part2_model2(stitched_output)

    print("\nFinal output of Model 1:")
    print(final_output_model1.shape)

    # Add comparison logic
    compare_outputs(final_output, final_output_model1)


def compare_outputs(output1, output2):
    difference = torch.abs(output1 - output2)
    print("\nDifference between stitched model output and Model 1 output:")
    print(difference)


if __name__ == "__main__":
    main()
