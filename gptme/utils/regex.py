import re


def trim_lines(text: str):
    return re.sub(r"^ +", "", text, 0, re.MULTILINE)
