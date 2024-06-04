import os
import certifi
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader

from StitchNet.temp_stitchingLayer import StitchingModel

# Use specific SSL-updated certificates
os.environ['SSL_CERT_FILE'] = certifi.where()


def load_dataset(batch_size=64):
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")


def test(model, test_loader, criterion):
    model.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Loss: {test_loss / len(test_loader)}, Accuracy: {100 * correct / total}%")


def main():
    train_loader, test_loader = load_dataset()

    stitching_model = StitchingModel('resnet18', 'resnet34', 5, 5)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model.parameters(), lr=0.001)

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
