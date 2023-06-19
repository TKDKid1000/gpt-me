import os
import pickle
import re
import sys
from argparse import ArgumentParser
from os import path
from time import time

from sentence_transformers import SentenceTransformer

msmarco_distilbert_base_v4 = SentenceTransformer("msmarco-distilbert-base-v4")

parser = ArgumentParser(
    prog="GPT-me Memory Summarizer",
    description="Summarizes memory files into a small enough size to be processed by GPT-me.",
)
parser.add_argument("filename")
parser.add_argument("-l", "--lines", type=int, default=10)

args = parser.parse_args(sys.argv[1:])

with open(args.filename, encoding="utf8") as transcript_file:
    transcript_lines = transcript_file.readlines()

# transcript = ["".join(transcript_lines[i : i + args.lines]) for i in range(0, len(transcript_lines))]

transcript = [""]
LAST_AUTHOR = ""

for line in transcript_lines:
    author = re.match(r"^(\w+):", line)
    if author is not None and author[1] != LAST_AUTHOR:
        transcript.append(line)
    else:
        transcript[-1] += line

    last_author = author[1] if author is not None else last_author

print(len(transcript))

embeddings = msmarco_distilbert_base_v4.encode(
    transcript, show_progress_bar=True, convert_to_tensor=True
)

output_path = f".memories/{time()}/{path.split(args.filename)[1]}.pickle"

os.makedirs(path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump([embeddings, transcript], f)
