from dataclasses import dataclass
from typing import Literal, List
import openai

@dataclass
class Message:
    content: str
    role: Literal["system", "user", "assistant"]

class Conversation:
    messages: List[Message] = []

    def __init__(self, messages: List[Message]=[]) -> None:
        self.messages = messages

    def add_message(self, message: Message):
        self.messages.append(message)

    def get_completion(self):
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[]
        )