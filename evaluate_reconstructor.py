from antigan import Dataset
import click
import math
import numpy as np
from pathlib import Path
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


@click.command()
@click.option("--batch_size", type=int, default=16)
@click.option("--num_workers", type=int, default=2)
@click.option("--device", type=str, default="cpu")
def main(
    batch_size: int,
    num_workers: int,
    device: str,
):
    device = torch.device(device)
    with open("reconstructor.pkl", "rb") as reconstructor_file:
        reconstructor = pickle.load(reconstructor_file).to(device)

    dataset = Dataset(Path.cwd() / "examples")
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    total_correct = np.zeros(128)
    for images, vectors in tqdm(data_loader):
        reconstructed_logits = reconstructor(images.to(device))
        total_correct += (
            ((vectors < 0) == (reconstructed_logits < 0)).detach().cpu().numpy().sum(0)
        )

    summary_writer = SummaryWriter()
    bit_accuracy = np.mean(total_correct / len(dataset))
    summary_writer.add_scalar("bit_accuracy", bit_accuracy)
    summary_writer.close()

    for redundance in [1, 3, 5, 7, 9]:
        accuracy = redundant_bit_accuracy(bit_accuracy, redundance)
        print(f"Bit accuracy with redundance of {redundance}: {accuracy * 100:.0f}%")
        print(
            f"Character accuracy with redundance of {redundance}: {accuracy ** 5 * 100:.0f}%"
        )
        print()


binomial_coefficient = lambda a, b: math.factorial(a) / (
    math.factorial(b) * math.factorial(a - b)
)
redundant_bit_accuracy = lambda bit_accuracy, redundance: sum(
    (
        binomial_coefficient(redundance, num_errors)
        * bit_accuracy ** (redundance - num_errors)
        * (1 - bit_accuracy) ** num_errors
    )
    for num_errors in range(redundance // 2 + 1)
)

main()
