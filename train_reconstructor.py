from antigan import Reconstructor
import click
import numpy as np
from pathlib import Path
import torch
from PIL import Image

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

def parse_data():
    labels= []
    images= []
    examples_path = Path.cwd() / "examples"

    with open(examples_path / "noise.npy", "rb") as noise_file:
        noise_vectors = torch.from_numpy(np.load(noise_file)).to('cpu')

    for counter in range(6125):
        print(counter)
        labels.append(noise_vectors[counter])

        img = Image.open(examples_path+'image_'+str(counter)+'.png')
        imarray = np.array(img, dtype=np.double)/255
        images.append(imarray)

    return images,labels
    

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

    examples_path = Path.cwd() / "examples"
    with open(examples_path / "noise.npy", "rb") as noise_file:
        noise_vectors = torch.from_numpy(np.load(noise_file)).to(device)
    
    train_images, train_labels = parse_data()

    

    print(len(train_labels))
    print(len(train_images))

main()