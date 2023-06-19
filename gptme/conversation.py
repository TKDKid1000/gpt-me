from dataclasses import dataclass
import math
from typing import Callable, Literal

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from gptme.utils.dataclass import asdict

from gptme.utils.dataclass import asdict


@dataclass
class Message:
    content: str | Callable[[], str]
    role: Literal["system", "user", "assistant"]


AI_FLAGS = ["ai", "artificial intelligence", "language model"]


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
        print("trying")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[asdict(message) for message in self.messages],
            temperature=temperature,
            max_tokens=max_tokens == -1 ,
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
