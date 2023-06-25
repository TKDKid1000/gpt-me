import os
import re
import sys
from argparse import ArgumentParser
from random import sample, seed
from time import time_ns

import openai
from dotenv import load_dotenv

from gptme.text_styler import TextStyler

load_dotenv(dotenv_path=".env")

openai.api_key = os.environ["OPENAI_KEY"]

parser = ArgumentParser(
    prog="GPT-me Style Extractor",
    description="Scans through a random sample of your messages to extract your approximate text style.",
)
parser.add_argument("filename")
parser.add_argument("-a", "--author", type=str)
parser.add_argument("-S", "--seed", type=int, default=time_ns())
parser.add_argument("-m", "--min_length", type=int, default=20)

args = parser.parse_args(sys.argv[1:])

with open(args.filename, encoding="utf8") as transcript_file:
    transcript_lines = transcript_file.readlines()


transcript = []

for line in transcript_lines:
    author = re.match(r"^(\w+):", line)
    if author is not None and author[1] == args.author:
        line = re.sub(r"^\w+: ", "", line)
        transcript.append(line)
    elif author is None:
        transcript[-1] += line

transcript = [line.strip() for line in transcript if len(line) >= args.min_length]

seed(args.seed)

message_sample = sample(transcript, 10)

for sample in message_sample:
    print(sample)

text_styler = TextStyler()

style = text_styler.extract_style("\n".join(message_sample))
