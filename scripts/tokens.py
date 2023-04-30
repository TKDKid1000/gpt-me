import sys

import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

with open(sys.argv[1]) as file:
    tokens = encoding.encode(file.read())

print(len(tokens))
