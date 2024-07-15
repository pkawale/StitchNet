import torch
from torch import nn
from sklearn.linear_model import LinearRegression
from pytorch_lightning import LightningModule
from helper_scripts.timm_surgery import split_model


class StitchingModel(LightningModule):
    def __init__(self, model1, model2, split1, split2, learning_rate=1e-3):
        super(StitchingModel, self).__init__()

        # Split the models into two parts
        self.part1_model1, self.part2_model1 = split_model(model1, split1)
        self.part1_model2, self.part2_model2 = split_model(model2, split2)

        # Sanity-check the model parts equal the model whole after splitting
        dummy_data = torch.randn(4, 3, 32, 32).to(next(self.part1_model1.parameters()).device)
        assert torch.all(model1(dummy_data) == self.part2_model1(self.part1_model1(dummy_data)))
        dummy_data = torch.randn(4, 3, 32, 32).to(next(self.part1_model2.parameters()).device)
        assert torch.all(model2(dummy_data) == self.part2_model2(self.part1_model2(dummy_data)))

        # Get number of channels and dimensions
        self.num_channels_model1, shape_model1 = self._get_num_channels(
            self.part1_model1, (4, 3, 32, 32)
        )
        self.num_channels_model2, shape_model2 = self._get_num_channels(
            self.part1_model2, (4, 3, 32, 32)
        )

        # Initialize the stitching layer to adjust channels and dimensions if needed
        self.stitching_layer = StitchingLayer(
            self.num_channels_model1,
            self.num_channels_model2,
            shape_model1,
            shape_model2,
        )
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def _get_num_channels(self, mdl, input_shape=(4, 3, 32, 32)):
        if len(list(mdl.children())) == 0:
            raise ValueError("One of the model parts is empty.")

        with torch.no_grad():
            # Forward pass through mdl to get the output shape
            x = torch.randn(*input_shape).to(next(mdl.parameters()).device)

            for layer in mdl.children():
                if isinstance(layer, nn.Module) and not isinstance(
                    layer, (nn.CrossEntropyLoss, nn.MSELoss, nn.L1Loss)
                ):
                    x = layer(x)
            num_output_channels = x.shape[1]
            output_shape = x.shape

        return num_output_channels, output_shape

    def forward(self, x):
        x = self.part1_model1(x)
        if x.dim() != 4:
            raise ValueError("Output of part1_model1 must have 4 dimensions.")
        x = self.stitching_layer(x)
        if x.dim() != 4:
            raise ValueError("Output of stitching_layer must have 4 dimensions.")
        x = self.part2_model2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log(
            "train_loss", loss,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log(
            "val_loss", loss,
        )
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log(
            "test_loss", loss,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.stitching_model.stitching_layer.parameters(), lr=self.learning_rate
        )

    def parameter_part1(self):
        yield from self.part1_model1.parameters()

    def parameters_part2(self):
        yield from self.part2_model2.parameters()

    def parameters_stitching(self):
        if isinstance(self.stitching_layer, StitchingLayer):
            yield from self.stitching_layer.parameters()

    def initialize_stitching_layer(self, sample_input):
        sample_input = sample_input.to(next(self.parameters()).device)
        with torch.no_grad():
            part1_output = self.part1_model1(sample_input)
            if part1_output.dim() < 4:
                part1_output = part1_output.view(
                    part1_output.size(0), part1_output.size(1), 1, 1
                )
            if part1_output.dim() < 4:
                raise ValueError("part1_output has less than 4 dimensions.")
            upscaled_output = nn.functional.interpolate(
                part1_output,
                size=(part1_output.size(2), part1_output.size(3)),
                mode="bilinear",
                align_corners=False,
            )
            self.stitching_layer.initialize_weights_with_regression(
                part1_output, upscaled_output
            )


class StitchingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, in_shape, out_shape):
        super(StitchingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Store the target spatial dimensions
        self.target_size = out_shape[2:]

    def forward(self, x):
        x = nn.functional.interpolate(
            x, size=self.target_size, mode="bilinear", align_corners=False
        )
        x = self.conv(x)
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
        final_device = self.conv.weight.device  # Use the device of the model weights
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
