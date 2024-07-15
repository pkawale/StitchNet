import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader


# Define the Stitching Layer
class StitchingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StitchingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x


# Function to get the number of output channels from a layer
def get_out_channels(layer):
    if isinstance(layer, nn.Conv2d):
        return layer.out_channels
    elif isinstance(layer, nn.BatchNorm2d):
        return layer.num_features
    elif isinstance(layer, nn.Linear):
        return layer.out_features
    else:
        raise TypeError(
            f"Layer type {type(layer)} is not supported to get out_channels"
        )


# Function to split the model
def split_model(model, split_layer):
    layers = list(model.children())
    bottom_layers = nn.Sequential(*layers[:split_layer])
    top_layers = nn.Sequential(*layers[split_layer:])
    return bottom_layers, top_layers


# Main Controller
class StitchedModel(nn.Module):
    def __init__(self, bottom_model, top_model, split_layer):
        super(StitchedModel, self).__init__()
        self.bottom_layers, _ = split_model(bottom_model, split_layer)
        _, self.top_layers = split_model(top_model, split_layer)

        # Get the output channels of the last layer of bottom_model and input channels of the first layer of top_model
        bottom_out_channels = get_out_channels(list(self.bottom_layers.children())[-1])
        top_in_channels = get_out_channels(list(self.top_layers.children())[0])

        self.stitching_layer = StitchingLayer(
            in_channels=bottom_out_channels, out_channels=top_in_channels
        )

    def forward(self, x):
        x = self.bottom_layers(x)
        x = self.stitching_layer(x)
        x = self.top_layers(x)
        return x


# Example usage
def train_stitched_model():
    # Load pre-trained models
    bottom_model = models.resnet18(pretrained=True)
    top_model = models.resnet34(pretrained=True)

    # Create the stitched model
    split_layer = 7  # Example split layer, split after layer3 of ResNet
    stitched_model = StitchedModel(bottom_model, top_model, split_layer)

    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitched_model.parameters(), lr=0.001)

    # Training loop
    stitched_model.train()
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = stitched_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


if __name__ == "__main__":
    train_stitched_model()
