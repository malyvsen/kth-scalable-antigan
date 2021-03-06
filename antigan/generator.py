from dataclasses import dataclass
from functools import cached_property
import numpy as np
from PIL import Image
from pytorch_pretrained_biggan import (
    BigGAN,
    convert_to_images,
)
import torch
from .config import noise_intensity, noise_dimensionality


@dataclass(frozen=True)
class Generator:
    big_gan: BigGAN
    class_id: int
    device: torch.device

    @classmethod
    def pretrained(cls, class_id: int, device: torch.device):
        return cls(
            big_gan=BigGAN.from_pretrained("biggan-deep-512").to(device),
            class_id=class_id,
            device=device,
        )

    def generate(self, noise: torch.Tensor = None) -> Image:
        if noise is None:
            noise = self.make_noise()
        with torch.no_grad():
            output = self.big_gan(
                noise.unsqueeze(0), self.class_vector, noise_intensity
            )
        return convert_to_images(output)[0]

    def make_noise(self) -> torch.Tensor:
        return torch.from_numpy(
            np.random.choice(
                [-noise_intensity, noise_intensity], size=noise_dimensionality
            )
        ).to(self.device)

    @cached_property
    def class_vector(self):
        result = torch.zeros(1, 1000, device=self.device)
        result[:, self.class_id] = 1
        return result
