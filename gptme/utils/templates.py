import re
from typing import Iterator, Sequence

StringRepresentable = str | int | float | bool


def join_(*strings: StringRepresentable):
    return "".join(str(string) for string in strings)


def if_(condition: bool, value: StringRepresentable, _else: StringRepresentable = ""):
    return value if condition else _else


def for_(
    iterator: Iterator[StringRepresentable] | Sequence[StringRepresentable], sep="\n"
):
    return sep.join(list([str(value) for value in iterator]))


def trim_lines(text: str):
    return re.sub(r"^ +", "", text, 0, re.MULTILINE)
