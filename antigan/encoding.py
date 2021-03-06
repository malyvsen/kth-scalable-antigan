import math
import numpy as np
import string
from .config import noise_intensity, noise_dimensionality, redundance


def text_to_noise(text: str) -> np.ndarray:
    padded_text = text + " " * (max_text_characters - len(text))
    noise_bits = [
        bit for character in padded_text for bit in character_to_noise(character)
    ]
    return np.concatenate(
        [
            noise_bits,
            np.random.choice(
                [-noise_intensity, noise_intensity],
                noise_dimensionality - len(noise_bits),
            ),
        ]
    ).astype(np.float32)


def character_to_noise(character: str):
    assert len(character) == 1
    index = allowed_characters.find(character)
    assert index >= 0
    binary = bin(index)[2:]
    padded_binary = "0" * (bits_per_certain_character - len(binary)) + binary
    return [
        -noise_intensity if bit == "0" else noise_intensity
        for bit in padded_binary
        for redundant_idx in range(redundance)
    ]


def noise_to_text(noise: np.ndarray) -> str:
    return "".join(
        noise_to_character(character_noise)
        for character_noise in np.split(
            noise[: max_text_characters * bits_per_character], max_text_characters
        )
    ).strip()


def noise_to_character(noise: np.ndarray):
    assert len(noise) == bits_per_character
    bits = [
        sum(noise[index] > 0 for index in indices) > redundance / 2
        for indices in zip(
            *[
                range(start_idx, bits_per_character, redundance)
                for start_idx in range(redundance)
            ]
        )
    ]
    binary = "".join("1" if bit else "0" for bit in bits)
    index = int(binary, 2)
    try:
        return allowed_characters[index]
    except IndexError:
        return "#"


allowed_characters = string.ascii_lowercase + " -,.?!"
bits_per_certain_character = math.ceil(math.log2(len(allowed_characters)))
bits_per_character = bits_per_certain_character * redundance
max_text_characters = noise_dimensionality // bits_per_character
