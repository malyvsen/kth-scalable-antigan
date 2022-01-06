from antigan import Generator, config
import math
import numpy as np
import pickle
from PIL import Image
import streamlit as st
import string
import torch
import torchvision as tv


def main():
    st.title("AntiGAN")
    st.write("Could these pictures of mountains be hosting secret messages?")
    options = {
        "Hide text in an image": generate,
        "Find the text hidden in an image": reconstruct,
    }
    choice = st.selectbox("What's your business?", options.keys())
    options[choice]()


def generate():
    text = st.text_input("Your message", max_chars=max_text_characters)
    if text == "":
        return
    if any(character not in allowed_characters for character in text):
        st.error("Only the following characters are allowed: " + allowed_characters)
        return
    padded_text = text + " " * (max_text_characters - len(text))
    noise_bits = [
        bit for character in padded_text for bit in character_to_noise(character)
    ]
    noise_vector = torch.tensor(
        noise_bits + [0] * (config.noise_dimensionality - len(noise_bits)),
        dtype=torch.float32,
    )

    generating_message = st.empty()
    generating_message.info("Generating, this might take a while")
    generator = load_generator()
    image = generator.generate(noise_vector)
    generating_message.empty()
    st.image(image, caption="This image contains your message. Share wisely!")


def reconstruct():
    image_file = st.file_uploader("Picture of mountains", type=["png", "jpg"])
    if image_file is None:
        return
    image = Image.open(image_file).resize((512, 512))
    decoding_message = st.empty()
    decoding_message.info("Decoding, this might take a while")
    reconstructor = load_reconstructor()
    noise = (
        reconstructor(tv.transforms.ToTensor()(image).unsqueeze(0))
        .squeeze(0)
        .detach()
        .numpy()
    )
    text = "".join(
        noise_to_character(character_noise)
        for character_noise in np.split(
            noise[: max_text_characters * bits_per_character], max_text_characters
        )
    )
    decoding_message.empty()
    st.write(f"The hidden text is: {text.strip()}")


def character_to_noise(character: str):
    assert len(character) == 1
    index = allowed_characters.find(character)
    assert index >= 0
    binary = bin(index)[2:]
    padded_binary = "0" * (bits_per_character - len(binary)) + binary
    return [
        -config.noise_intensity if bit == "0" else config.noise_intensity
        for bit in padded_binary
    ]


def noise_to_character(noise: np.ndarray):
    assert len(noise) == bits_per_character
    binary = "".join("0" if value < 0 else "1" for value in noise)
    index = int(binary, 2)
    try:
        return allowed_characters[index]
    except IndexError:
        return "?"


allowed_characters = (
    string.digits + string.ascii_letters + string.punctuation + string.whitespace
)
bits_per_character = math.ceil(math.log2(len(allowed_characters)))
max_text_characters = config.noise_dimensionality // bits_per_character


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_generator():
    return Generator.pretrained(
        class_id=config.class_id,
        device=torch.device("cpu"),
    )


@st.cache(show_spinner=False)
def load_reconstructor():
    with open("reconstructor.pkl", "rb") as reconstructor_file:
        return pickle.load(reconstructor_file)


main()
