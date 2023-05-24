from dataclasses import asdict as _asdict
from types import LambdaType

__all__ = ["asdict"]


def asdict(obj):
    dictionary = _asdict(obj)

    for _, (key, value) in enumerate(dictionary.items()):
        if isinstance(value, LambdaType):
            dictionary[key] = value()

    return dictionary
