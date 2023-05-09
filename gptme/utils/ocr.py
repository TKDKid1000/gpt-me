from io import BytesIO

import requests
from PIL import Image
from pytesseract import image_to_string

ImageLike = str | Image.Image | bytes


def image_ocr(image: ImageLike, lang="eng") -> str:
    if isinstance(image, str):
        response = requests.get(url=image)
        image = Image.open(BytesIO(response.content))
    elif isinstance(image, bytes):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise ValueError()

    try:
        text = image_to_string(image=image, lang=lang, timeout=2)
    except RuntimeError:
        text = ""

    return text.strip()
