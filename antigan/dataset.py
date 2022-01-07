from functools import cached_property
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision as tv


class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples_path: Path):
        self.examples_path = examples_path

    def __getitem__(self, index):
        return (
            self.transform(Image.open(self.examples_path / f"image_{index}.png")),
            self.noise[index],
        )

    def __len__(self):
        return self.num_examples

    @cached_property
    def noise(self):
        return torch.from_numpy(
            np.load(self.examples_path / "noise.npy")[: self.num_examples]
        ).float()

    @cached_property
    def num_examples(self):
        return sum(1 for image_path in self.examples_path.glob("*.png"))

    @cached_property
    def transform(self):
        return tv.transforms.ToTensor()
