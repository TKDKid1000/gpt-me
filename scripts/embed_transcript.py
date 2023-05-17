from argparse import ArgumentParser
import os
import pickle
import sys
from os import path
from time import time
import re

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("msmarco-distilroberta-base-v3")

parser = ArgumentParser(
    prog="GPT-me Memory Summarizer",
    description="Summarizes memory files into a small enough size to be processed by GPT-me.",
)
parser.add_argument("filename")
parser.add_argument("-l", "--lines", type=int, default=10)

args = parser.parse_args(sys.argv[1:])

with open(args.filename) as transcript_file:
    transcript_lines = transcript_file.readlines()

# transcript = ["".join(transcript_lines[i : i + args.lines]) for i in range(0, len(transcript_lines))]

transcript = [""]
last_author = ""

for line in transcript_lines:
    author = re.match(r"^(\w+):", line)
    if author != None and author[1] != last_author:
        transcript.append(line)
    else:
        transcript[-1] += line
    
    last_author = author[1] if author != None else last_author

print(len(transcript))

embeddings = model.encode(transcript, show_progress_bar=True, convert_to_tensor=True)

output_path = f".memories/{time()}/{path.split(args.filename)[1]}.pickle"

os.makedirs(path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump([embeddings, transcript], f)
