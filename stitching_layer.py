import torch
from timm import create_model
from torch import nn


class StitchingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, output_size):
        super(StitchingLayer, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=False) if input_size != output_size else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.upsample(x)
        return x


class StitchingModel(nn.Module):
    def __init__(self, model1_name, model2_name, split1, split2):
        super(StitchingModel, self).__init__()

        # Load the pre-trained models
        self.model1 = self.create_cifar10_resnet(model1_name)
        self.model2 = self.create_cifar10_resnet(model2_name)

        # Split the models into two parts
        self.part1_model1 = nn.Sequential(*list(self.model1.children())[:split1])
        self.part2_model2 = nn.Sequential(*list(self.model2.children())[split2:])
        self.part1_model2 = nn.Sequential(*list(self.model2.children())[:split2])

        if len(list(self.part1_model1.children())) == 0:
            raise ValueError(f"Model1 part1 is empty with split index {split1}")

        if len(list(self.part2_model2.children())) == 0:
            raise ValueError(f"Model2 part2 is empty with split index {split2}")

        # Get number of channels and dimensions
        self.num_channels_model1, shape_model1 = self._get_num_channels(self.part1_model1)
        self.num_channels_model2, shape_model2 = self._get_num_channels(self.part1_model2)

        print(f"num_output_channels: {self.num_channels_model1}, num_input_channels: {self.num_channels_model2}")

        # Initialize the stitching layer to adjust channels and dimensions if needed
        self.stitching_layer = StitchingLayer(
            self.num_channels_model1,
            self.num_channels_model2,
            (shape_model1[2], shape_model1[3]),
            (shape_model2[2], shape_model2[3])
        )

    def create_cifar10_resnet(self, model_name):
        model = create_model(model_name, pretrained=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model

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
        print(f"Input shape: {x.shape}")
        x = self.part1_model1(x)
        print(f"After part1_model1: {x.shape}")
        x = self.stitching_layer(x)
        print(f"After stitching_layer: {x.shape}")
        x = self.part2_model2(x)
        print(f"After part2_model2: {x.shape}")
        return x

    def parameter_part1(self):
        yield from self.part1_model1.parameters()

    def parameters_part2(self):
        yield from self.part2_model2.parameters()

    def parameters_stitching(self):
        if isinstance(self.stitching_layer, StitchingLayer):
            yield from self.stitching_layer.parameters()


# Example usage
if __name__ == "__main__":
    model1_name = "resnet18"
    model2_name = "resnet34"
    split1 = 6
    split2 = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stitching_model = StitchingModel(model1_name, model2_name, split1, split2).to(device)
    print(stitching_model)

    # Debugging with a dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    stitching_model(dummy_input)
