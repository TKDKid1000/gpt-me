import os
import pickle
import sys

import torch
from sentence_transformers.util import cos_sim
from gptme.models import msmarco_distilroberta_base_v3


def semantic_search(query, embeddings, transcript, top_k=5):
    top_k = min(top_k, len(transcript))

    query_embedding = msmarco_distilroberta_base_v3.encode(
        query, convert_to_tensor=True
    )

    if not isinstance(query_embedding, torch.Tensor):
        raise TypeError()

    scores = cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    for score, index in zip(top_results[0], top_results[1]):
        yield "".join(transcript[index - 5 : index + 5]), score
