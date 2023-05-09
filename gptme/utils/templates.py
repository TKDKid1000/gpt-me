from typing import Iterator, Sequence

StringRepresentable = str | int | float | bool


def _join(*strings: StringRepresentable):
    return "".join(str(string) for string in strings)


def _if(condition: bool, value: StringRepresentable, _else: StringRepresentable = ""):
    return value if condition else _else


def _for(
    iterator: Iterator[StringRepresentable] | Sequence[StringRepresentable], sep="\n"
):
    return sep.join(list([str(value) for value in iterator]))
