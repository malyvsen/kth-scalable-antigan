from .config import noise_dimensionality
import torch
from typing import List
from .residual_block import ResidualBlock


class Reconstructor(torch.nn.Module):
    def __init__(
        self,
        num_conv_layers: int,
        num_conv_channels: int,
        kernel_size: int,
        downsampling: int,
        num_adapter_units: int,
    ):
        super().__init__()
        self.convolutional = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=num_conv_channels, kernel_size=1
            ),
            *[
                torch.nn.Sequential(
                    ResidualBlock(
                        channels=num_conv_channels,
                        kernel_size=kernel_size,
                        activation=torch.nn.LeakyReLU(),
                    ),
                    torch.nn.MaxPool2d(downsampling),
                )
                for _ in range(num_conv_layers)
            ]
        )
        final_image_size = 512 // downsampling ** num_conv_layers
        self.adapters = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(final_image_size ** 2, num_adapter_units),
                torch.nn.LeakyReLU(),
            )
            for channel in range(num_conv_channels)
        )
        self.head = torch.nn.Linear(
            num_conv_channels * num_adapter_units, noise_dimensionality
        )

    def forward(self, image):
        convolved: torch.Tensor = self.convolutional(
            image
        )  # batch, channels, height, width
        to_adapt = convolved.flatten(start_dim=2).permute(
            1, 0, 2
        )  # channels, batch, height * width
        adapted = torch.stack(
            [
                adapter(channel_slice)
                for channel_slice, adapter in zip(to_adapt, self.adapters)
            ],
            dim=1,
        )  # batch, channels, num_adapter_units
        return self.head(adapted.flatten(start_dim=1))
