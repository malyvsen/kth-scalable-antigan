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
@click.option("--learning_rate", type=float, default=5e-3)
@click.option("--num_conv_layers", type=int, default=3)
@click.option("--num_conv_channels", type=int, default=4)
@click.option("--kernel_size", type=int, default=3)
@click.option("--downsampling", type=int, default=4)
@click.option("--num_workers", type=int, default=2)
@click.option("--device", type=str, default="cpu")
def main(
    num_batches: int,
    batch_size: int,
    learning_rate: float,
    num_conv_layers: int,
    num_conv_channels: int,
    kernel_size: int,
    downsampling: int,
    num_workers: int,
    device: str,
):
    device = torch.device(device)
    reconstructor = Reconstructor(
        num_conv_layers=num_conv_layers,
        num_conv_channels=num_conv_channels,
        kernel_size=kernel_size,
        downsampling=downsampling,
    ).to(device)
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    dataset = Dataset(Path.cwd() / "examples")
    make_data_iterator = lambda: iter(
        torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
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
        reconstructed_logits = reconstructor(images.to(device))
        loss = criterion(reconstructed_logits, (vectors.to(device) > 0).float())
        summary_writer.add_scalar("loss", loss, global_step=batch_idx)
        summary_writer.add_scalar(
            "bit_accuracy",
            ((vectors < 0) == (reconstructed_logits < 0)).float().mean(),
            global_step=batch_idx,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with open("reconstructor.pkl", "wb") as reconstructor_file:
        pickle.dump(reconstructor, reconstructor_file)
    summary_writer.close()


main()
