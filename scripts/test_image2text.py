import os
from os.path import join

from PIL import Image

from gptme.images.image_to_text import image_to_text
import streamlit as st
import pandas as pd

test_image_directory = "test_images"

st.title("GPT-me Image2Text Test")

for filename in sorted(os.listdir(test_image_directory)):
    image = Image.open(join(test_image_directory, filename))
    prediction = image_to_text(image)

    c = st.container()
    c.image(image)
    c.markdown(
        f"**Generated Text:** {prediction['generated_text']}\n**Type:** {prediction['image_type']}"
    )
