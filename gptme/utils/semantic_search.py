import torch
from sentence_transformers.util import cos_sim

from gptme.models import msmarco_distilbert_base_v4


def semantic_search(query, embeddings, transcript, top_k=5):
    top_k = min(top_k, len(transcript))

    query_embedding = msmarco_distilbert_base_v4.encode(query, convert_to_tensor=True)

    if not isinstance(query_embedding, torch.Tensor):
        raise TypeError()

    scores = cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    for score, index in zip(top_results[0], top_results[1]):
        yield "".join(transcript[index]), score
