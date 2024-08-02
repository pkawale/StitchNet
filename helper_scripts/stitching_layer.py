import torch
import numpy as np
from torch import nn
from sklearn.linear_model import LinearRegression
from lightning.pytorch import LightningModule
import torch.nn.functional as F
from typing import Self
from itertools import chain
from helper_scripts.timm_surgery import split_model


def _get_num_channels(mdl, input_shape=(4, 3, 32, 32), device="cpu"):
    if len(list(mdl.children())) == 0:
        raise ValueError("One of the model parts is empty.")

    with torch.no_grad():
        # Forward pass through mdl to get the output shape
        x = torch.randn(*input_shape).to(device)

        for layer in mdl.children():
            if isinstance(layer, nn.Module) and not isinstance(
                layer, (nn.CrossEntropyLoss, nn.MSELoss, nn.L1Loss)
            ):
                x = layer(x)
        num_output_channels = x.shape[1]
        output_shape = x.shape

    return num_output_channels, output_shape


class StitchingModel(LightningModule):
    def __init__(
        self,
        model1,
        model2=None,
        split1=None,
        split2=None,
        learning_rate=1e-3,
        enable_learning=False,
        l2_lambda=np.inf,
    ):
        super(StitchingModel, self).__init__()

        # split the models into two parts based on index
        self.part1_model1, self.part2_model1 = split_model(model1, split1)
        self.part1_model2, self.part2_model2 = split_model(model2, split2)

        self.l2_lambda = l2_lambda

        # # Sanity-check the model parts equal the model whole after splitting
        dummy_data = torch.randn(4, 3, 32, 32)
        assert torch.all(model1(dummy_data) == self.model1(dummy_data))
        assert torch.all(model2(dummy_data) == self.model2(dummy_data))

        if len(list(self.part1_model1.children())) == 0:
            raise ValueError(f"Model1 part1 is empty with split index {split1}")

        if len(list(self.part2_model2.children())) == 0:
            raise ValueError(f"Model2 part2 is empty with split index {split2}")

        # Get number of channels and dimensions
        self.num_channels_model1, shape_model1 = _get_num_channels(
            self.part1_model1, (4, 3, 32, 32)
        )
        self.num_channels_model2, shape_model2 = _get_num_channels(
            self.part1_model2, (4, 3, 32, 32)
        )

        # Initialize the stitching layer to adjust channels and dimensions if needed
        self.stitching_layer = StitchingLayer(
            self.num_channels_model1,
            self.num_channels_model2,
            shape_model1,
            shape_model2,
        )
        # Debug: Check convolutional layer weights
        # print(f"Conv layer weights after init: {self.stitching_layer.conv.weight.shape}")
        self.learning_rate = learning_rate
        self.enable_learning = enable_learning
        self.criterion = nn.CrossEntropyLoss()

        # Save original parameters for regularization
        self.original_param_values = {}
        self.store_model2_parameters()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["enable_learning"] = self.enable_learning
        state["l2_lambda"] = self.l2_lambda
        state["learning_rate"] = self.learning_rate
        return state

    def load_state_dict(self, dict, *args, **kwargs):
        super().load_state_dict(dict, *args, **kwargs, strict=False)
        self.enable_learning = dict["enable_learning"]
        self.l2_lambda = dict["l2_lambda"]
        self.learning_rate = dict["learning_rate"]
        self.store_model2_parameters()

    def store_model2_parameters(self):
        self.original_param_values = {
            name: param.clone().detach()
            for name, param in self.part2_model2.named_parameters()
        }

    @property
    def model1(self):
        return LitSequential(self.part1_model1, self.part2_model1)

    @property
    def model2(self):
        return LitSequential(self.part1_model2, self.part2_model2)

    def to(self, *args, **kwargs) -> Self:
        super().to(*args, **kwargs)
        for key, value in self.original_param_values.items():
            self.original_param_values[key] = value.to(*args, **kwargs)
        return self

    def forward(self, x):
        x = self.part1_model1(x)
        if x.dim() != 4:
            raise ValueError("Output of part1_model1 must have 4 dimensions.")
        x = self.stitching_layer(x)
        if x.dim() != 4:
            raise ValueError("Output of stitching_layer must have 4 dimensions.")
        x = self.part2_model2(x)
        return x

    def losses(self, batch, pre: str = ""):
        x, y = batch
        outputs = self(x)
        loss_terms = {
            f"{pre}cross_entropy": F.cross_entropy(outputs, y, reduction="mean"),
            f"{pre}delta_weights": self.regularization(),
        }
        if self.enable_learning:
            loss_terms[f"{pre}loss"] = (
                loss_terms[f"{pre}cross_entropy"]
                + loss_terms[f"{pre}delta_weights"] * self.l2_lambda
            )
        else:
            loss_terms[f"{pre}loss"] = loss_terms[f"{pre}cross_entropy"]
        return loss_terms

    def training_step(self, batch, batch_idx):
        terms = self.losses(batch, "train_")
        self.log_dict(terms)
        return terms["train_loss"]

    def validation_step(self, batch, batch_idx):
        terms = self.losses(batch, "val_")
        self.log_dict(terms)
        return terms["val_loss"]

    def test_step(self, batch, batch_idx):
        terms = self.losses(batch, "test_")
        self.log_dict(terms)
        return terms["test_loss"]

    def regularization(self):
        l2_reg = torch.tensor(0.0).to(self.device)
        for name, param in self.part2_model2.named_parameters():
            # Skip all batchnorm parameters; they change during training but not by gradient descent
            # so we don't want to 'penalize' those changes in the logged loss values
            if "bn" in name:
                continue
            original_param_value = self.original_param_values[name]
            # Move original_param_value to the same device as param
            original_param_value = original_param_value.to(param.device)
            l2_reg += torch.sum((param - original_param_value) ** 2)
        return self.l2_lambda * l2_reg

    def configure_optimizers(self):
        if self.enable_learning:
            return torch.optim.Adam(
                chain(
                    self.stitching_layer.parameters(), self.part2_model2.parameters()
                ),
                lr=self.learning_rate,
            )
        else:
            return torch.optim.Adam(
                self.stitching_layer.parameters(), lr=self.learning_rate
            )

    def initialize_stitching_layer(self, sample_input):
        sample_input = sample_input.to(next(self.parameters()).device)
        with torch.no_grad():
            part1_output = self.part1_model1(sample_input)
            # print(f"part1_output shape: {part1_output.shape}")
            if part1_output.dim() < 4:
                part1_output = part1_output.view(
                    part1_output.size(0), part1_output.size(1), 1, 1
                )
            if part1_output.dim() < 4:
                raise ValueError("part1_output has less than 4 dimensions.")
            part2_output = self.part1_model2(sample_input)
            # upscaled_output = nn.functional.interpolate(
            #     part1_output,
            #     size=(part1_output.size(2), part1_output.size(3)),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            self.stitching_layer.initialize_weights_with_regression(
                part1_output, part2_output
            )
            # print(
            #     f"Stitching layer initialized with shapes: input {part1_output.shape}, output {part2_output.shape}"
            # )


class LitSequential(LightningModule):
    def __init__(self, *models):
        super(LitSequential, self).__init__()
        self.model = nn.Sequential(*models)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        raise RuntimeError(
            "We don't expect to ever be trining the LitSequential wrapper module. "
            "Something must have gone wrong."
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        stats = {"val_cross_entropy": F.cross_entropy(self(x), y)}
        stats["val_loss"] = stats["val_cross_entropy"]
        self.log_dict(stats)
        return stats["val_cross_entropy"]

    def test_step(self, batch, batch_idx):
        x, y = batch
        stats = {"test_cross_entropy": F.cross_entropy(self(x), y)}
        stats["test_loss"] = stats["test_cross_entropy"]
        self.log_dict(stats)
        return stats["test_cross_entropy"]


class StitchingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, in_shape, out_shape):
        super(StitchingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Store the target spatial dimensions
        self.target_size = out_shape[2:]
        # Debug: Check convolutional layer weights
        # print(f"Conv layer weights after init: {self.conv.weight.shape}")

    def forward(self, x):
        x = nn.functional.interpolate(
            x, size=self.target_size, mode="bilinear", align_corners=False
        )
        # print(f"Shape after interpolation: {x.shape}")
        # print(f"Applying conv layer with weights shape: {self.conv.weight.shape}")
        x = self.conv(x)
        # print(f"Shape after conv: {x.shape}")
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

        # print(
        #     f"Resized input tensor shape: {input_tensor.shape}, resized output tensor shape: {output_tensor.shape}"
        # )  # Debugging output shape

        # Upscale or downscale input_tensor to match output_tensor dimensions
        if height != out_height or width != out_width:
            input_tensor = nn.functional.interpolate(
                input_tensor,
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False,
            )

            # print(
            #     f"Final input tensor shape for regression: {input_tensor.shape}, output tensor shape: {output_tensor.shape}"
            # )  # Debugging output shape

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


if __name__ == "__main__":
    x = torch.randn(
        32, 256, 2, 2
    )  # Assuming batch size 32, 256 channels from part1_model1
    stitching_layer = StitchingLayer(
        in_channels=256,
        out_channels=128,
        in_shape=(32, 256, 2, 2),
        out_shape=(32, 128, 4, 4),
    )
    output = stitching_layer(x)
    print(f"Final output shape: {output.shape}")  # Should be (32, 128, 4, 4)
