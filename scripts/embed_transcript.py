import os
import pickle
import sys
from os import path
from time import time

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(sys.argv[1]) as transcript_file:
    transcript = transcript_file.readlines()

embeddings = model.encode(transcript, show_progress_bar=True, convert_to_tensor=True)

output_path = f".memories/{time()}/{path.split(sys.argv[1])[1]}.pickle"

os.makedirs(path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump([embeddings, transcript], f)
