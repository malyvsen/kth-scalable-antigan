from .config import noise_dimensionality
import torch


class Reconstructor(torch.nn.Module):
    def __init__(
        self,
        downsampling: int,
        num_hidden_layers: int,
        hidden_width: int,
    ):
        super().__init__()
        self.downsampler = torch.nn.AvgPool2d(downsampling)
        downsampled_size = 512 // downsampling
        self.head = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(prev_width, width), torch.nn.LeakyReLU()
                )
                for prev_width, width in zip(
                    [downsampled_size ** 2 * 3] + [hidden_width] * num_hidden_layers,
                    [hidden_width] * num_hidden_layers,
                )
            ],
            torch.torch.nn.Linear(hidden_width, noise_dimensionality),
        )

    def forward(self, image):
        downsampled = self.downsampler(image)
        return self.head(downsampled.flatten(start_dim=1))
