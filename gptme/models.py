from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Primarily for on-device question answering and summarization.
flan_t5_large = pipeline("text2text-generation", "google/flan-t5-large")

# For search embedding generation.
msmarco_distilroberta_base_v3 = SentenceTransformer("msmarco-distilroberta-base-v3")

# For conversation summarization.
distilbart_cnn_12_6_samsum = pipeline(
    "summarization", model="philschmid/distilbart-cnn-12-6-samsum"
)
