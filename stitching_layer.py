import torch
from timm import create_model
from torch import nn
from sklearn.linear_model import LinearRegression

from model_splitter import get_input_dim


class StitchingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, output_size):
        super(StitchingLayer, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.upsample = (
            nn.Upsample(size=output_size, mode="bilinear", align_corners=False)
            if input_size != output_size
            else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
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
        final_device = input_tensor.device
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
            .to(final_device)
        )
        self.conv.bias.data = torch.tensor(reg.intercept_, dtype=torch.float32).to(
            final_device
        )


class StitchingModel(nn.Module):
    def __init__(self, model1, model2, split1, split2):
        super(StitchingModel, self).__init__()

        # Split the models into two parts
        self.part1_model1 = nn.Sequential(*list(self.model1.children())[:split1])
        self.part2_model2 = nn.Sequential(*list(self.model2.children())[split2:])
        self.part1_model2 = nn.Sequential(*list(self.model2.children())[:split2])

        if len(list(self.part1_model1.children())) == 0:
            raise ValueError(f"Model1 part1 is empty with split index {split1}")

        if len(list(self.part2_model2.children())) == 0:
            raise ValueError(f"Model2 part2 is empty with split index {split2}")

        # Get number of channels and dimensions
        self.num_channels_model1, shape_model1 = self._get_num_channels(
            self.part1_model1
        )
        self.num_channels_model2, shape_model2 = self._get_num_channels(
            self.part1_model2
        )

        if len(shape_model1) == 2 or len(shape_model1) == 3:
            shape_model1 = (shape_model1[0], shape_model1[1], 1, 1)
        if len(shape_model2) == 2 or len(shape_model1) == 3:
            shape_model2 = (shape_model2[0], shape_model2[1], 1, 1)
        # Initialize the stitching layer to adjust channels and dimensions if needed
        self.stitching_layer = StitchingLayer(
            self.num_channels_model1,
            self.num_channels_model2,
            (shape_model1[2], shape_model1[3]),
            (shape_model2[2], shape_model2[3]),
        )

    def _get_num_channels(self, mdl):
        if len(list(mdl.children())) == 0:
            raise ValueError("One of the model parts is empty.")

        with torch.no_grad():
            # Forward pass through part1 to get the output shape
            x = torch.randn(1, 3, 32, 32).to(next(mdl.parameters()).device)
            for layer in mdl:
                x = layer(x)
            num_output_channels = x.shape[1]
            output_shape = x.shape

        return num_output_channels, output_shape

    def forward(self, x):
        x = self.part1_model1(x)
        x = self.stitching_layer(x)
        x = self.part2_model2(x)
        return x

    def parameter_part1(self):
        yield from self.part1_model1.parameters()

    def parameters_part2(self):
        yield from self.part2_model2.parameters()

    def parameters_stitching(self):
        if isinstance(self.stitching_layer, StitchingLayer):
            yield from self.stitching_layer.parameters()

    def initialize_stitching_layer(self, sample_input):
        with torch.no_grad():
            part1_output = self.part1_model1(sample_input)
            if part1_output.dim() < 4:
                part1_output = part1_output.view(
                    part1_output.size(0), part1_output.size(1), 1, 1
                )
            if part1_output.dim() < 4:
                raise ValueError("part1_output has less than 4 dimensions.")
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


# Example usage
if __name__ == "__main__":
    model1_name = "resnet18"
    model2_name = "resnet34"
    split1 = 6
    split2 = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stitching_model = StitchingModel(model1_name, model2_name, split1, split2).to(
        device
    )
    # print(stitching_model)

    # Debugging with a dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    stitching_model(dummy_input)
