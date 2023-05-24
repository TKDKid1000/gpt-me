from io import BytesIO

import requests
from PIL import Image

ImageLike = str | Image.Image | bytes


def imagelike_convert(image: ImageLike):
    if isinstance(image, str):
        response = requests.get(url=image)
        image = Image.open(BytesIO(response.content))
    elif isinstance(image, bytes):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise ValueError()

    return image
