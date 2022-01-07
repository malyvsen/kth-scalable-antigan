from antigan import Reconstructor
from antigan import dataset
from antigan import train
import click
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


@click.command()
@click.option("--num_batches", type=int, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--num_conv_layers", type=int, default=3)
@click.option("--num_conv_channels", type=int, default=16)
@click.option("--kernel_size", type=int, default=3)
@click.option("--downsampling", type=int, default=4)
@click.option("--num_adapter_units", type=int, default=16)
@click.option("--device", type=str, default="cpu")
def main(
    num_batches: int,
    batch_size: int,
    learning_rate: float,
    num_conv_layers: int,
    num_conv_channels: int,
    kernel_size: int,
    downsampling: int,
    num_adapter_units: int,
    device: str,
):
    device = torch.device(device)
    reconstructor = Reconstructor(
        num_conv_layers=num_conv_layers,
        num_conv_channels=num_conv_channels,
        kernel_size=kernel_size,
        downsampling=downsampling,
        num_adapter_units=num_adapter_units,
    ).to(device)
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_labels = []
    train_images = []

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    examples_path = Path.cwd() / "examples"
    with open(examples_path / "noise.npy", "rb") as noise_file:
        noise_vectors = torch.from_numpy(np.load(noise_file)).to(device)

    for counter in range(5):
        train_labels.append(noise_vectors[counter])

        img = Image.open(examples_path / f"image_" + str(counter) + ".png")

        imarray = np.array(img, dtype=np.double) / 255
        train_images.append(imarray)

    train_dataset = dataset.AntiganDataset(train_images, train_labels, transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    num_epochs = 5
    train_loss = []

    for epoch in range(num_epochs):
        train_loss.append(
            train.model_train(
                epoch, train_dataloader, device, reconstructor, criterion, optimizer
            )
        )
        torch.save(reconstructor, "reconstructor_" + str(epoch) + ".pth")


main()
