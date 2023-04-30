import os
import pickle
import sys

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(sys.argv[1], "rb") as embeddings_file:
    embeddings, transcript = pickle.load(embeddings_file)


def semantic_search(query, embeddings, transcript, top_k=5):
    top_k = min(top_k, len(transcript))

    query_embedding = model.encode(query, convert_to_tensor=True)

    if not isinstance(query_embedding, torch.Tensor):
        raise TypeError()

    scores = cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    for score, index in zip(top_results[0], top_results[1]):
        yield transcript[index], score


for response, score in semantic_search(
    sys.argv[2],
    embeddings=embeddings,
    transcript=transcript,
    top_k=10,
):
    print(response, score)
