from antigan import Reconstructor
import click
import numpy as np
from pathlib import Path
import torch


@click.command()
@click.option("--num_batches", type=int, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--num_conv_layers", type=int, default=3)
@click.option("--num_conv_channels", type=int, default=16)
@click.option("--downsampling", type=int, default=4)
@click.option("--dense_width", type=int, default=256)
@click.option("--device", type=str, default="cpu")
def main(
    num_batches: int,
    batch_size: int,
    learning_rate: float,
    num_conv_layers: int,
    num_conv_channels: int,
    kernel_size: int,
    downsampling: int,
    dense_width: int,
    device: str,
):
    device = torch.device(device)
    reconstructor = Reconstructor(
        num_conv_layers=num_conv_layers,
        num_conv_channels=num_conv_channels,
        kernel_size=kernel_size,
        downsampling=downsampling,
        dense_widths=[dense_width],
    ).to(device)
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=learning_rate)

    examples_path = Path.cwd() / "examples"
    with open(examples_path / "noise.npy", "rb") as noise_file:
        noise_vectors = torch.from_numpy(np.load(noise_file)).to(device)
    # TODO: iterate over the data and train the reconstructor


main()
