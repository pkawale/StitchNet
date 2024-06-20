import torch
from torch import nn


def split_model(model, split_index):
    layers = list(model.children())
    part1 = nn.Sequential(*layers[:split_index])
    part2 = nn.Sequential(*layers[split_index:])
    return part1, part2


def get_output_dim(part):
    last_conv_or_linear = None
    for layer in part.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            last_conv_or_linear = layer

    if last_conv_or_linear is None:
        raise AttributeError(
            "No convolutional or linear layer found in the given model part."
        )

    if isinstance(last_conv_or_linear, nn.Conv2d):
        print(
            f"Output dimension from get_output_dim (Conv2d): {last_conv_or_linear.out_channels}"
        )  # Debug print
        return last_conv_or_linear.out_channels
    elif isinstance(last_conv_or_linear, nn.Linear):
        print(
            f"Output dimension from get_output_dim (Linear): {last_conv_or_linear.out_features}"
        )  # Debug print
        return last_conv_or_linear.out_features
    else:
        raise ValueError("The last layer is neither Conv2d nor Linear.")


def get_input_dim(part):
    for layer in part.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            dim = (
                layer.in_channels if isinstance(layer, nn.Conv2d) else layer.in_features
            )
            return dim
    raise AttributeError("No layer with 'in_channels' or 'in_features' found")
