from typing import List

from PIL.Image import Image

from gptme.conversation import Conversation, Message


AI_FLAGS = ["ai", "artificial intelligence", "language model"]

class Assistant:
    conversation: Conversation
    memories: str
    personality: str

    def __init__(self, memories: str, personality: str, name: str) -> None:
        self.memories = memories
        self.personality = personality
        self.conversation = Conversation(
            [
                Message(
                    content=f'You are a conversational and casual language bot that imitates a human as best as possible. Take on the persona of the human named {name}. Do not mention that you are an AI Language Model. If a request cannot be fulfilled from knowledge, respond with "Search: [message]", and a web search will be performed for you. If something violates content limitations, politely decline without saying you are an AI.',
                    role="system",
                ),
                # Message(
                #     content=f"You recall your past memories:\n{self.memories}\n",
                #     role="user",
                # ),
                Message(
                    content=f"You are to behave as a human, so you will randomly generate emotions, thoughts, physical experiences, and feelings when asked those questions. You must never say that you are an AI model.",
                    role="user",
                ),
                Message(
                    content=f"The person you are emulating, {name}, has these personality traits. Emulate {name} in first person, do not say that you are emulating them. All of the you are comfortable sharing:\n{self.personality}",
                    role="user"
                )
            ]
        )

    def send_message(self, text: str, images: List[Image] = []): # TODO: Add small delay between receiving and responding to allow for multiple messages to be received prior to responding.
        self.conversation.add_message(message=Message(content=text, role="user"))
        response: str = self.conversation.get_completion_chat()["choices"][0]["message"]["content"]
        if any(flag in response.lower() for flag in AI_FLAGS):
            print(f"Using InstructGPT to respond.")
            response = self.conversation.get_completion_instruct()["choices"][0]["message"]["content"]
        self.conversation.add_message(
            message=Message(
                content=response, role="assistant"
            )
        )
