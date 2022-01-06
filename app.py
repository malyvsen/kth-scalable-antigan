from antigan import Generator, config
import math
import pickle
import streamlit as st
import string
import torch


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
        bit for character in padded_text for bit in character_to_bits(character)
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
    text = "not implemented yet!"  # TODO: actually decode the image
    st.write(f"The hidden text is: {text.strip()}")


def character_to_bits(character: str):
    assert len(character) == 1
    index = allowed_characters.find(character)
    assert index >= 0
    binary = bin(index)[2:]
    padded_binary = "0" * (bits_per_character - len(binary)) + binary
    return [
        -config.truncation if bit == "0" else config.truncation for bit in padded_binary
    ]


allowed_characters = (
    string.digits + string.ascii_letters + string.punctuation + string.whitespace
)
bits_per_character = math.ceil(math.log2(len(allowed_characters)))
max_text_characters = config.noise_dimensionality // bits_per_character


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_generator():
    return Generator.pretrained(
        truncation=config.truncation,
        class_id=config.class_id,
        device=torch.device("cpu"),
    )


@st.cache(show_spinner=False)
def load_reconstructor():
    with open("reconstructor.pkl", "rb") as reconstructor_file:
        return pickle.load(reconstructor_file)


main()
