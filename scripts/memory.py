import os
import sys
from argparse import ArgumentParser
from os import path
from time import time
from typing import List

from gptrim import trim
from tqdm import tqdm
from transformers import pipeline

summarizer = pipeline("summarization", model="philschmid/distilbart-cnn-12-6-samsum")

MAX_TOKENS_SUMMARY = 16000
MAX_SIZE_MEMORY = 1600


def summarize_memory(memory_lines: List[str], section_length=20, levels=1, gptrim=True):
    source = memory_lines

    for level in range(levels):
        print(f"Beginning summary for level: {level}")
        section_summary = []
        for section in tqdm(range(0, len(source), section_length), desc=f"Level {level}"):
            source_text = "\n".join(
                source[section : min(section + section_length, len(source))]
            )
            try:
                summary = summarizer(
                    source_text,
                    max_length=128,
                    min_length=32,
                    do_sample=False,
                )
            except IndexError: # should only trim the messages if vital for summary
                summary = summarizer(
                    trim(source_text, remove_spaces=False, remove_punctuation=True),
                    max_length=128,
                    min_length=32,
                    do_sample=False,
                )
            section_summary.append(summary[0]["summary_text"])  # type: ignore
        source = section_summary

    return (
        trim("\n".join(source), remove_spaces=False, remove_punctuation=True)
        if gptrim
        else "\n".join(source)
    )

    # return summarizer(memory_text, max_length=250, min_length=50, do_sample=False)[0]["summary_text"] # type: ignore
    # tokens = tokenizer(memory_text, max_length=MAX_TOKENS_SUMMARY, return_tensors="pt", truncation=True)

    # print(tokenizer.batch_decode(model.generate(tokens["input_ids"], num_beams=2, min_length=0, max_length=MAX_SIZE_MEMORY))[0]) # type: ignore

    # input_ids = tokenizer([memory_text], return_tensors="pt")["input_ids"]

    # if not isinstance(input_ids, torch.Tensor):
    #     raise TypeError()

    # prediction = model.generate(input_ids)[0]

    # return tokenizer.decode(prediction, skip_special_tokens=True)

    # create_directory(f".memories/{level}/")
    # for x in range(0, math.ceil(len(token_ids) / MAX_TOKENS_SUMMARY)-1):
    #     chunk_token_ids = token_ids[x*MAX_TOKENS_SUMMARY:min((x+1)*MAX_TOKENS_SUMMARY, len(tokens))]
    #     tokens = tokenizer.convert_ids_to_tokens(chunk_token_ids)
    #     print(len(tokens))
    #     start = time.time()
    #     summary = model(tokens)
    #     print(summary)
    #     with open(f".memories/{level}/{x}", "w") as memory_file:
    #         duration = time.time() - start

    #         memory_file.write(summary[0]["summary_text"])
    #         print(f"Duration: {duration}\nSummarized section {x} as: \"{summary[0]['summary_text']}\"")


parser = ArgumentParser(
    prog="GPT-me Memory Summarizer",
    description="Summarizes memory files into a small enough size to be processed by GPT-me.",
)
parser.add_argument("filename")
parser.add_argument("-s", "--section_length", type=int, default=20)
parser.add_argument("-l", "--levels", type=int, default=1)
parser.add_argument("-t", "--trim", action="store_true", default=False)

args = parser.parse_args(sys.argv[1:])


output_path = f".memories/{time()}/{path.split(args.filename)[1]}"

os.makedirs(path.dirname(output_path), exist_ok=True)

with open(args.filename) as f:
    memories_text = f.readlines()

with open(output_path, "w") as f:
    f.write(
        summarize_memory(
            memories_text,
            section_length=args.section_length,
            levels=args.levels,
            gptrim=args.trim,
        )
    )
