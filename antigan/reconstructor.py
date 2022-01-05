from functools import reduce
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
        dense_widths: List[int],
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
        first_dense_width = num_conv_channels * reduce(
            (lambda size, _: (size - kernel_size + 1) // downsampling),
            range(num_conv_layers),
            initial=512,
        )
        self.dense = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(prev_width, width), torch.nn.LeakyReLU()
                )
                for prev_width, width in zip(
                    [first_dense_width] + dense_widths, dense_widths + [128]
                )
            ]
        )

    def forward(self, image):
        convolved: torch.Tensor = self.convolutional(image)
        return self.dense(convolved.flatten(start_dim=1))
