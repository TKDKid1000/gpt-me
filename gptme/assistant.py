import json
from datetime import date
from typing import Sequence

import torch

import gptme.utils.templates as t
from gptme.conversation import AI_FLAGS, Conversation, Message
from gptme.images.ocr import ImageLike, image_ocr
from gptme.utils.dataclass import asdict
from gptme.utils.summarize import get_summarization_instruct
from gptme.utils.web_search import web_search


class Assistant:
    conversation: Conversation
    embeddings: torch.Tensor | None = None
    memories: list[str] | None = None
    personality: str

    def __init__(
        self,
        name: str,
        personality: str,
        embeddings: torch.Tensor | None = None,
        memories: list[str] | None = None,
    ) -> None:
        self.personality = personality

        if embeddings is not None:
            self.embeddings = embeddings

        if memories is not None:
            self.memories = memories

        # pylint: disable=line-too-long
        self.conversation = Conversation(
            [
                Message(
                    content=lambda: f"""You are a conversational and casual language bot that imitates a human as best as possible.
                    Take on the persona of the human named {name}.
                    Do not mention that you are an AI Language Model.
                    If a request cannot be fulfilled from knowledge, respond with "Search: [message]", and a web search will be performed for you.
                    If something violates content limitations, politely decline without saying you are an AI.
                    You are to behave as a human, so you will randomly generate emotions, thoughts, physical experiences, and feelings when asked those questions.
                    You must never say that you are an AI model.
                    The person you are emulating, {name}, has these personality traits.
                    Emulate {name} in first person, do not say that you are emulating them.
                    All of the you are comfortable sharing:
                    {self.personality}
                    Attempt to continue conversations, but do not overly suggest helping the other person.
                    
                    Messages will be formatted in sections.
                    The Images: section will contain transcribed image data.
                    The Memories: section will contain past memories searched.
                    The Message: section should be your primary focus and replied to.
                    
                    Current date: {date.today()}
                    """,
                    role="system",
                ),
                # Message(
                #     content=f"You recall your past memories:\n{self.memories}\n",
                #     role="user",
                # ),
            ]
        )

    def send_message(
        self, text: str, images: Sequence[ImageLike] = None
    ):  # TODO: Add small delay between receiving and responding to allow for multiple messages to be received prior to responding.
        image_texts = (
            [image_ocr(image) for image in images] if images is not None else []
        )

        message_content = t.join_(
            t.if_(
                len(image_texts) > 0,
                t.join_(
                    "\nImage Transcriptions:\n",
                    t.for_(image_texts),
                ),
            ),
            "\nMessage: ",
            text,
        )

        # TODO: Integrate message transcript semantic search system to determine what memories to search up – if any.
        # TODO: Integrate web search system if common information is unknown.

        print(message_content)
        # return

        self.conversation.add_message(
            message=Message(content=message_content, role="user")
        )
        response: str = self.conversation.get_completion_chat()["choices"][0][
            "message"
        ]["content"]

        if any(flag in response.lower() for flag in AI_FLAGS):
            print("ChatGPT response: ", response)
            print("Using InstructGPT to respond.")
            response = self.conversation.get_completion_instruct()["choices"][0]["text"]

        if response.startswith("Search: "):
            search_term = response[8:]
            print(f"Searching for {search_term}.")
            results = web_search(search_term)
            response = get_summarization_instruct(
                t.for_(result.snippet for result in results)
            )["choices"][0]["text"]

        assistant_message = Message(content=response, role="assistant")
        self.conversation.add_message(message=assistant_message)

        with open(
            ".memories/conversation.json", "w", encoding="utf8"
        ) as conversation_file:
            json.dump(
                list(asdict(message) for message in self.conversation.messages),
                conversation_file,
            )

        return assistant_message
