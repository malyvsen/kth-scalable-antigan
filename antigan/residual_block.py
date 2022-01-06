import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation):
        super().__init__()
        self.padding = kernel_size // 2
        self.convolutional = torch.nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=kernel_size
        )
        self.activation = activation

    def forward(self, image):
        padded_image = torch.nn.functional.pad(
            image, (self.padding, self.padding, self.padding, self.padding)
        )
        return image + self.activation(self.convolutional(padded_image))
