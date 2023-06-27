from dataclasses import asdict as _asdict
from types import LambdaType

__all__ = ["asdict"]


def asdict(obj):
    dictionary = _asdict(obj)
    updated_dictionary = {}

    for _, (key, value) in enumerate(dictionary.items()):
        if value is None:
            continue
        if isinstance(value, LambdaType):
            updated_dictionary[key] = value()
        else:
            updated_dictionary[key] = value

    return updated_dictionary
