import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=kernel_size
        )
        self.activation = activation

    def forward(self, image):
        return image + self.activation(self.convolutional(image))
