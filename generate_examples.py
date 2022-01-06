from antigan import Generator, config
import click
import numpy as np
import torch
from tqdm.auto import tqdm


@click.command()
@click.option("--num_examples", type=int, required=True)
@click.option("--device", type=str, default="cpu")
def main(num_examples: int, device: str):
    generator = Generator.pretrained(
        truncation=config.truncation,
        class_id=config.class_id,
        device=torch.device(device),
    )
    noise_vectors = [generator.make_noise() for _ in range(num_examples)]
    np.save("noise.npy", [noise_vector.cpu().numpy() for noise_vector in noise_vectors])
    for example_idx, noise_vector in enumerate(
        tqdm(noise_vectors, desc="Generating examples")
    ):
        generator.generate(noise_vector).save(f"images/image_{example_idx}.png")


main()
