import torch
import torch.nn as nn


class ModelSplitter(nn.Module):
    def __init__(self, model):
        super(ModelSplitter, self).__init__()
        self.model = model
        self.layers = []
        self._register_layers()

    def _register_layers(self):
        for name, layer in self.model.named_children():
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


def split_model_by_index(model, split_index):
    """
    Split a PyTorch model into two parts at the specified index.
    """
    part1 = nn.Sequential()
    part2 = nn.Sequential()

    current_index = 0
    split_point_reached = False

    for name, layer in model.named_children():
        if current_index == split_index:
            split_point_reached = True

        if not split_point_reached:
            part1.add_module(name, layer)
        else:
            part2.add_module(name, layer)

        current_index += 1

    return part1, part2
