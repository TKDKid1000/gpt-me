from dataclasses import dataclass
from typing import Callable, Literal

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from gptme.utils.dataclass import asdict


@dataclass
class Message:
    content: str | Callable[[], str]
    role: Literal["system", "user", "assistant"]


class Conversation:
    messages: list[Message] = []

    def __init__(self, messages: list[Message] = None) -> None:
        self.messages = messages if messages is not None else []

    def add_message(self, message: Message):
        self.messages.append(message)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(9))
    def get_completion_chat(
        self,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=-1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[asdict(message) for message in self.messages],
            temperature=temperature,
            max_tokens=None if max_tokens == -1 else max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        print(response)
        if not isinstance(response, dict):
            raise TypeError()

        return response

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(9))
    def get_completion_instruct(
        self,
        model="text-curie-001",
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        response = openai.Completion.create(
            model=model,
            prompt="\n".join(
                [
                    f"{message.role}: {message.content}"
                    for message in self.messages[
                        1:
                    ]  # remove the first message, that containing memories, in order to fit token limits
                    if message.role != "system"
                ]
            )
            + "\nassistant:",  # system messages are not included in instruct requests to save on tokens
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        if not isinstance(response, dict):
            raise TypeError()

        return response

    def count_tokens_chat(self, model="gpt-3.5-turbo"):
        """Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb"""
        try:
            tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 3
        tokens_per_name = 1
        tokens = 0

        for message in self.messages:
            tokens += tokens_per_message
            for key, value in asdict(message).items():
                tokens += len(tokenizer.encode(value))
                if key == "name":
                    tokens += tokens_per_name

        tokens += 3
        return tokens
