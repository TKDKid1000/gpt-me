import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal

import openai
from openai.error import RateLimitError
from tenacity import (
    retry,stop_after_attempt,wait_random_exponential
)


@dataclass
class Message:
    content: str
    role: Literal["system", "user", "assistant"]


class Conversation:
    messages: List[Message] = []

    def __init__(self, messages: List[Message] = []) -> None:
        self.messages = messages

    def add_message(self, message: Message):
        self.messages.append(message)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(9))
    def get_completion_chat(self, model="gpt-3.5-turbo"):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[asdict(message) for message in self.messages],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        if not isinstance(response, Dict):
            raise TypeError()

        return response

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(9))
    def get_completion_instruct(self, model="text-curie-001"):
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
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        if not isinstance(response, Dict):
            raise TypeError()

        return response
