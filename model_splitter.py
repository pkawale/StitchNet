import torch
from torch import nn


def split_model(model, split_index):
    layers = list(model.children())
    part1 = nn.Sequential(*layers[:split_index])
    part2 = nn.Sequential(*layers[split_index:])
    return part1, part2


def get_output_dim(part):
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = part(dummy_input)
    if len(output.shape) == 4:  # Expecting (batch_size, channels, height, width)
        print(f"Output dimension from get_output_dim: {output.shape[1]}")  # Debug print
        return output.shape[1]
    else:
        raise ValueError("Output tensor does not have 4 dimensions as expected.")


def get_input_dim(part):
    for layer in part.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            dim = (
                layer.in_channels if isinstance(layer, nn.Conv2d) else layer.in_features
            )
            return dim
    raise AttributeError("No layer with 'in_channels' or 'in_features' found")
