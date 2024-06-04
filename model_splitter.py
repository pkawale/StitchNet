from torch import nn


def split_model(model, split_index):
    layers = list(model.children())
    part1 = nn.Sequential(*layers[:split_index])
    part2 = nn.Sequential(*layers[split_index:])
    return part1, part2


def get_output_dim(part):
    for layer in reversed(list(part.modules())):
        if hasattr(layer, "out_channels"):
            return layer.out_channels
    raise AttributeError("No layer with 'out_channels' found")


def get_input_dim(part):
    for layer in part.modules():
        if hasattr(layer, "in_channels"):
            return layer.in_channels
    raise AttributeError("No layer with 'in_channels' found")
