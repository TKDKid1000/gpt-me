from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Primarily for on-device question answering and summarization.
# flan_t5_large = pipeline("text2text-generation", "google/flan-t5-large")

# For search embedding generation.
msmarco_distilbert_base_v4 = SentenceTransformer("msmarco-distilbert-base-v4")

# For conversation summarization.
distilbart_cnn_12_6_samsum = pipeline(
    "summarization", model="philschmid/distilbart-cnn-12-6-samsum"
)

# For image classification.
clip_vit_base_patch32 = pipeline(
    "zero-shot-image-classification", model="openai/clip-vit-base-patch32"
)

# For image caption generation.
vit_gpt2_image_captioning = pipeline(
    "image-to-text", model="nlpconnect/vit-gpt2-image-captioning"
)
