import timm
import torch
import torch.nn as nn
from model_splitter import get_output_dim, get_input_dim, split_model


class StitchingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StitchingLayer, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class StitchingModel:
    def __init__(self, model_name1, model_name2, split_index1, split_index2):
        super(StitchingModel, self).__init__()
        self.model1 = timm.create_model(model_name1, pretrained=True)
        self.model2 = timm.create_model(model_name2, pretrained=True)

        self.split_index1 = split_index1
        self.split_index2 = split_index2

        self.part1_model1, self.part2_model1 = split_model(self.model1, split_index1)
        self.part1_model2, self.part2_model2 = split_model(self.model2, split_index2)

        input_dim = get_output_dim(self.part1_model1)
        output_dim = get_input_dim(self.part2_model2)

        self.stitching_layer = StitchingLayer(input_dim, output_dim)

    def create_stitching_layer(self, input_tensor):
        outputs1 = [input_tensor]
        x = input_tensor
        for layer in self.part1_model1:
            x = layer(x)
            outputs1.append(x)

        stitched_output = self.stitching_layer(outputs1[self.split_index1])

        outputs2 = [stitched_output]
        x = stitched_output
        for layer in self.part2_model2:
            x = layer(x)
            outputs2.append(x)

        return outputs1, outputs2

    def forward(self, x):
        part1_output = self.part1_model1(x)
        stitched_output = self.stitching_layer(part1_output)
        final_output = self.part2_model2(stitched_output)
        return final_output
