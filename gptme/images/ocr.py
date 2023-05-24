from pytesseract import image_to_string

from gptme.images.image import ImageLike, imagelike_convert


def image_ocr(image: ImageLike, lang="eng") -> str:
    image = imagelike_convert(image)
    try:
        text = image_to_string(
            image=image, lang=lang, timeout=2
        )
    except RuntimeError:
        text = ""

    return text.strip()
