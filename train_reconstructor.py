from antigan import Reconstructor
import click
import numpy as np
from pathlib import Path
import torch
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

    train_labels= []
    train_images= []

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
    
    examples_path = Path.cwd() / "examples"
    img_path = '/Users/aanair/Library/Caches/pypoetry/virtualenvs/kth-scalable-antigan-OYKsXUJ2-py3.9/.guild/runs/14ecd370b97e4a2abe3cb8c7377d6009/'
    with open(examples_path / "noise.npy", "rb") as noise_file:
        noise_vectors = torch.from_numpy(np.load(noise_file)).to(device)
    
    for counter in range(6125):
        train_labels.append(noise_vectors[counter])
        
        img = Image.open(img_path+'image_'+str(counter)+'.png')
        
        imarray = np.array(img, dtype=np.double)/255
        train_images.append(imarray)

    train_dataloader = torch.utils.data.DataLoader(
    dataset=train_images,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)
    tensor_train_images = torch.tensor([train_images[i] for i in train], dtype=torch.float32, device=device)

    print(len(train_labels))
    print(len(train_images))

main()