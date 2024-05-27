import timm
import torch
from model_splitter import ModelSplitter, split_model_by_index

def main():
    model1 = timm.create_model('resnet18', pretrained=True)
    model2 = timm.create_model('resnet34', pretrained=True)

    model_splitter1 = ModelSplitter(model1)
    model_splitter2 = ModelSplitter(model2)

    split_index1 = 5
    part1_model1, part2_model1 = split_model_by_index(model1, split_index1)

    split_index2 = 10
    part1_model2, part2_model2 = split_model_by_index(model2, split_index2)

    print("Model 1 - Part 1:")
    print(part1_model1)
    print("\nModel 1 - Part 2:")
    print(part2_model1)

    print("\nModel 2 - Part 1:")
    print(part1_model2)
    print("\nModel 2 - Part 2:")
    print(part2_model2)

    # Example input tensor
    input_tensor = torch.randn(1, 3, 224, 224)

    # Get the outputs of each layer for model1
    outputs1 = model_splitter1(input_tensor)

    # Get the outputs of each layer for model2
    outputs2 = model_splitter2(input_tensor)

    # Print the outputs
    print("\nOutputs of Model 1 layers:")
    for idx, output in enumerate(outputs1):
        print(f"Output of layer {idx}: {output.shape}")

    print("\nOutputs of Model 2 layers:")
    for idx, output in enumerate(outputs2):
        print(f"Output of layer {idx}: {output.shape}")

if __name__ == "__main__":
    main()
