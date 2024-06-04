import os
import certifi
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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

    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    return train_loader

def main():
    train_loader = load_dataset()

    images, labels = next(iter(train_loader))

    stitching_model = StitchingModel('resnet18', 'resnet34', 5,5)

    outputs1, outputs2 = stitching_model.create_stitching_layer(images)

    print("\n Outputs of Model 1 layers:")
    for idx, output in enumerate(outputs1):
        print(f"Output of layer {idx}: {output.shape}")

    print("\nOutputs of Model 2 layers:")
    for idx, output in enumerate(outputs2):
        print(f"Output of layer {idx} : {output.shape}")

    # Forward pass through the stitching model
    final_output = stitching_model.forward(images)

    print("\n Final output after stitching:")
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