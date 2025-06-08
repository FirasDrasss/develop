from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Double 3D Convolution block with batch normalization and ReLU activation."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling block with maxpool and double convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling block with transposed convolution and double convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # in_channels is the number of channels from the previous layer
        # We need to halve it for the upsampling path
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        # After concatenation, the number of channels will be (in_channels // 2) + out_channels
        self.conv = DoubleConv3D((in_channels // 2) + out_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle potential size mismatch
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FlowFieldUNet3D(nn.Module):
    """3D U-Net model for predicting flow fields from body shape masks."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 5,
        features: List[int] = [32, 64, 128, 256],
    ):
        """
        Initialize the 3D U-Net model.

        Args:
            in_channels: Number of input channels (default: 1 for binary mask)
            out_channels: Number of output channels (default: 5 for flow field)
            features: List of feature dimensions for each level of the U-Net
        """
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Initial convolution
        self.inc = DoubleConv3D(in_channels, features[0])

        # Downsampling path
        for i in range(len(features) - 1):
            self.downs.append(Down3D(features[i], features[i + 1]))

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1])

        # Upsampling path
        for i in reversed(range(len(features))):
            if i == 0:
                # For the first upsampling block, we need to handle the bottleneck output
                self.ups.append(Up3D(features[-1], features[0]))
            else:
                # For other upsampling blocks, we use the current and previous feature dimensions
                self.ups.append(Up3D(features[i], features[i - 1]))

        # Final convolution
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, in_channels, x, y, z)

        Returns:
            Output tensor of shape (batch_size, out_channels, x, y, z)
        """
        # Store intermediate outputs for skip connections
        skip_connections = []

        # Initial convolution
        x = self.inc(x)
        skip_connections.append(x)

        # Downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        # Remove last skip connection as it's not needed
        skip_connections = skip_connections[:-1]

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling path with skip connections
        for up, skip in zip(self.ups, reversed(skip_connections)):
            x = up(x, skip)

        # Final convolution
        return self.outc(x)


def create_flow_field_model(
    input_shape: Tuple[int, int, int] = (16, 16, 16),
    in_channels: int = 1,
    out_channels: int = 5,
    features: List[int] = [32, 64, 128, 256],
) -> FlowFieldUNet3D:
    """
    Create a flow field prediction model with specified parameters.

    Args:
        input_shape: Expected input shape (x, y, z)
        in_channels: Number of input channels
        out_channels: Number of output channels
        features: List of feature dimensions for each level of the U-Net

    Returns:
        Initialized FlowFieldUNet3D model
    """
    model = FlowFieldUNet3D(
        in_channels=in_channels, out_channels=out_channels, features=features
    )

    # Initialize weights using Kaiming initialization
    def init_weights(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    return model


# Example usage:
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 2
    x = torch.randn(batch_size, 1, 16, 16, 16)  # (batch_size, channels, x, y, z)

    # Create and test the model
    model = create_flow_field_model()
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Print shapes at each layer for debugging
    def print_shapes(model, x):
        print("\nLayer shapes:")
        print(f"Input: {x.shape}")

        # Initial convolution
        x = model.inc(x)
        print(f"After initial conv: {x.shape}")

        # Downsampling path
        for i, down in enumerate(model.downs):
            x = down(x)
            print(f"After down {i}: {x.shape}")

        # Bottleneck
        x = model.bottleneck(x)
        print(f"After bottleneck: {x.shape}")

        # Upsampling path (just for demonstration, won't match actual forward pass)
        for i, up in enumerate(model.ups):
            print(
                f"Up {i} input channels: {up.up.in_channels}, output channels: {up.up.out_channels}"
            )
            print(f"Up {i} conv input channels: {up.conv.double_conv[0].in_channels}")

    print_shapes(model, torch.randn(1, 1, 16, 16, 16))
