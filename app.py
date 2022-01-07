from antigan import Generator, config
from antigan.encoding import (
    text_to_noise,
    noise_to_text,
    allowed_characters,
    max_text_characters,
)
import pickle
from PIL import Image
import streamlit as st
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

    generating_message = st.empty()
    generating_message.info("Generating, this might take a while")
    generator = load_generator()
    image = generator.generate(text_to_noise(text))
    generating_message.empty()
    st.image(image, caption="This image contains your message. Share wisely!")


def reconstruct():
    image_file = st.file_uploader("Picture of mountains", type=["png", "jpg", "jpeg"])
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
    decoding_message.empty()
    st.write(f"The hidden text is: {noise_to_text(noise)}")


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
