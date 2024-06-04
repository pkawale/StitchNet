import timm
import torch
import torch.nn as nn

from StitchNet.model_splitter import split_model_by_index, ModelSplitter


class StitchingLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(StitchingLayer, self).__init__()
        self.stitching_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.stitching_layer(x)


class StitchingModel:
    def __init__(self, model_name1, model_name2, split_index1, split_index2):
        self.model1 = timm.create_model(model_name1, pretrained=True)
        self.model2 = timm.create_model(model_name2, pretrained=True)
        self.split_index1 = split_index1
        self.split_index2 = split_index2

        # Split Models
        self.part1_model1, self.part2_model1 = split_model_by_index(self.model1, self.split_index1)
        self.part1_model2, self.part2_model2 = split_model_by_index(self.model2, self.split_index2)

        self.model_splitter1 = ModelSplitter(self.part1_model1)
        self.model_splitter2 = ModelSplitter(self.part2_model2)

        # Create stitching layer
        self.stitch_input_channels = None
        self.stitch_output_channels = None
        self.stitching_layer = None

    def create_stitching_layer(self, input_tensor):
        outputs1 = self.model_splitter1(input_tensor)
        outputs2 = self.model_splitter2(input_tensor)

        print(f"Length of outputs1: {len(outputs1)}")
        print(f"Length of outputs2: {len(outputs2)}")

        if self.split_index1 >= len(outputs1):
            raise IndexError(
                f"split_index1 ({self.split_index1}) is out of range for outputs1 with length {len(outputs1)}")

        if self.split_index2 >= len(outputs2):
            raise IndexError(
                f"split_index2 ({self.split_index2}) is out of range for outputs2 with length {len(outputs2)}")

        self.stitch_input_channels = outputs1[self.split_index1].size(1)
        self.stitch_output_channels = self.part2_model2[0].in_channels

        self.stitching_layer = StitchingLayer(self.stitch_input_channels, self.stitch_output_channels)

        return outputs1, outputs2

    def forward(self, input_tensor):
        outputs1 = self.model_splitter1(input_tensor)
        stitched_output = self.stitching_layer(outputs1[self.split_index1].unsqueeze(2).unsqueeze(3))
        final_output = self.part2_model2(stitched_output)
        return final_output
