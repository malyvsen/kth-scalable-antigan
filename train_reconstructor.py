from antigan import Reconstructor, Dataset
import click
from pathlib import Path
import pickle
import torch
from tqdm.auto import tqdm


@click.command()
@click.option("--num_epochs", type=int, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--num_conv_layers", type=int, default=3)
@click.option("--num_conv_channels", type=int, default=16)
@click.option("--kernel_size", type=int, default=3)
@click.option("--downsampling", type=int, default=4)
@click.option("--num_adapter_units", type=int, default=16)
@click.option("--num_workers", type=int, default=2)
@click.option("--device", type=str, default="cpu")
def main(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_conv_layers: int,
    num_conv_channels: int,
    kernel_size: int,
    downsampling: int,
    num_adapter_units: int,
    num_workers: int,
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
    criterion = torch.nn.MSELoss()

    dataset = Dataset(Path.cwd() / "examples")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    for epoch_idx in range(num_epochs):
        for images, vectors in tqdm(
            train_dataloader, desc=f"Epoch {epoch_idx + 1}/{num_epochs}"
        ):
            reconstructed_noise = reconstructor(images.to(device))
            loss = criterion(reconstructed_noise, vectors.to(device))
            print(f" Epoch: {epoch_idx+1} - Loss: {loss}"  )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with open("reconstructor.pkl", "wb") as reconstructor_file:
            pickle.dump(reconstructor, reconstructor_file)


main()
