from antigan import Reconstructor, Dataset
import click
from pathlib import Path
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange


@click.command()
@click.option("--num_batches", type=int, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--downsampling", type=int, default=64)
@click.option("--num_hidden_layers", type=int, default=2)
@click.option("--hidden_width", type=int, default=256)
@click.option("--num_workers", type=int, default=2)
@click.option("--device", type=str, default="cpu")
def main(
    num_batches: int,
    batch_size: int,
    learning_rate: float,
    downsampling: int,
    num_hidden_layers: int,
    hidden_width: int,
    num_workers: int,
    device: str,
):
    device = torch.device(device)
    reconstructor = Reconstructor(
        downsampling=downsampling,
        num_hidden_layers=num_hidden_layers,
        hidden_width=hidden_width,
    ).to(device)
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    dataset = Dataset(Path.cwd() / "examples")
    make_data_iterator = lambda: iter(
        torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    )
    data_iterator = make_data_iterator()

    summary_writer = SummaryWriter()
    for batch_idx in trange(num_batches):
        try:
            images, vectors = next(data_iterator)
        except StopIteration:
            data_iterator = make_data_iterator()
            images, vectors = next(data_iterator)
        reconstructed_noise = reconstructor(images.to(device))
        loss = criterion(reconstructed_noise, vectors.to(device))
        summary_writer.add_scalar("loss", loss.item(), global_step=batch_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with open("reconstructor.pkl", "wb") as reconstructor_file:
        pickle.dump(reconstructor, reconstructor_file)
    summary_writer.close()


main()
