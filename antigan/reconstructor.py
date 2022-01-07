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
        self.head = torch.nn.Linear(
            final_image_size ** 2 * num_conv_channels, noise_dimensionality
        )

    def forward(self, image):
        convolved = self.convolutional(image)
        return self.head(convolved.flatten(start_dim=1))
