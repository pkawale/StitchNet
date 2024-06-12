import timm
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from model_splitter import split_model, get_output_dim, get_input_dim


class StitchingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StitchingLayer, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def initialize_weights_with_regression(self, input_tensor, output_tensor):
        """
        Given input_tensor and output_tensor, which are example inputs and outputs of the stitching
        layer, this function initializes self.conv.weight and self.conv.bias with a linear regression fit to the data.
        :param input_tensor: torch.Tensor
        :param output_tensor: torch.Tensor
        :return: None
        """
        # Ensure input tensor has 4 dimensions
        if len(input_tensor.shape) != 4:
            raise ValueError(
                "Input tensor must have 4 dimensions (batch, features, height, width)."
            )

        # Reshape output tensor to have 4 dimensions if it has 2
        if len(output_tensor.shape) == 2:
            output_tensor = output_tensor.view(
                output_tensor.size(0), output_tensor.size(1), 1, 1
            )

        # Check output tensor again
        if len(output_tensor.shape) != 4:
            raise ValueError(
                "Output tensor must have 4 dimensions (batch, features, height, width)."
            )

        batch_size, input_dim, height, width = input_tensor.shape
        _, output_dim, out_height, out_width = output_tensor.shape

        # Upscale or downscale input_tensor to match output_tensor dimensions
        if height != out_height or width != out_width:
            input_tensor = nn.functional.interpolate(
                input_tensor,
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False,
            )

        # Move tensors to CPU
        input_tensor = input_tensor.cpu()
        output_tensor = output_tensor.cpu()

        # Flatten the tensors while keeping the channel dimension
        X = input_tensor.permute(0, 2, 3, 1).reshape(-1, input_dim)
        y = output_tensor.permute(0, 2, 3, 1).reshape(-1, output_dim)

        reg = LinearRegression().fit(X.numpy(), y.numpy())

        # Initialize convolutional layer weights and bias
        self.conv.weight.data = (
            torch.tensor(reg.coef_, dtype=torch.float32)
            .view(output_dim, input_dim, 1, 1)
            .to(input_tensor.device)
        )
        self.conv.bias.data = torch.tensor(reg.intercept_, dtype=torch.float32).to(
            input_tensor.device
        )

        # Update BatchNorm layer to match the output dimension of the conv layer
        self.bn = nn.BatchNorm2d(output_dim)


class StitchingModel(nn.Module):
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

    def parameter_part1(self):
        yield from self.model1.parameters()

    def parameters_part2(self):
        yield from self.model2.parameters()

    def parameters_stitching(self):
        yield from self.stitching_layer.parameters()

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

    def initialize_stitching_layer(self, sample_input):
        with torch.no_grad():
            part1_output = self.part1_model1(sample_input)
            part2_input_dim = get_input_dim(self.part2_model2)
            upscaled_output = nn.functional.interpolate(
                part1_output,
                size=(part1_output.size(2), part1_output.size(3)),
                mode="bilinear",
                align_corners=False,
            )
            self.stitching_layer.initialize_weights_with_regression(
                part1_output, upscaled_output
            )
