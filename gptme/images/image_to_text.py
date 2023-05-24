from PIL import Image

from gptme.images.image import ImageLike, imagelike_convert
from gptme.images.ocr import image_ocr
from gptme.models import clip_vit_base_patch32, vit_gpt2_image_captioning
from gptme.utils.timer import timer

document_labels = ["document", "paper"]
screenshot_labels = ["computer screenshot"]
photo_labels = ["person photo", "indoor photo", "outdoor photo"]

candidate_labels = document_labels + screenshot_labels + photo_labels


@timer
def image_to_text(image: ImageLike):
    image = imagelike_convert(image)
    image_type = clip_vit_base_patch32(images=image, candidate_labels=candidate_labels)[0]["label"]

    generated_text = ""

    if image_type in document_labels:
        generated_text = image_ocr(image)
    elif image_type in screenshot_labels:
        generated_text = image_ocr(image)
    else:
        image_caption = vit_gpt2_image_captioning(image, max_new_tokens=64)
        generated_text = image_caption[0]["generated_text"]

    return {"generated_text": generated_text, "image_type": image_type}
