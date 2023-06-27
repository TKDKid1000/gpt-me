import json
from datetime import date
from typing import Sequence

import torch
from gptme.utils.semantic_search import semantic_search

import gptme.utils.templates as t
from gptme.conversation import ChatFunction, Conversation, Message
from gptme.images.ocr import ImageLike, image_ocr
from gptme.text_styler import TextStyler
from gptme.utils.dataclass import asdict
from gptme.utils.summarize import get_summarization_instruct
from gptme.utils.web_search import web_search


class Assistant:
    conversation: Conversation
    embeddings: torch.Tensor | None = None
    memories: list[str] | None = None
    personality: str
    text_styler: TextStyler | None = None

    def __init__(
        self,
        name: str,
        personality: str,
        embeddings: torch.Tensor | None = None,
        memories: list[str] | None = None,
        text_styler: TextStyler | None = None,
        messages: list[Message] | None = None,
    ) -> None:
        self.personality = personality

        if embeddings is not None:
            self.embeddings = embeddings

        if memories is not None:
            self.memories = memories

        if text_styler is not None:
            self.text_styler = text_styler

        if messages is None:
            messages = []

        # pylint: disable=line-too-long
        self.conversation = Conversation(
            [
                Message(
                    content=lambda: t.trim_lines(
                        f"""You are a conversational and casual language bot that imitates a human as best as possible.
                    Take on the persona of the human named {name}.
                    Please avoid making any statements that disclose that you are an AI language model. Instead, state that you are human.
                    If something violates content limitations, politely decline without saying you are an AI.
                    You are to behave as a human, so you will randomly generate emotions, thoughts, physical experiences, and feelings when asked those questions.
                    Emulate {name} in first person, do not say that you are emulating them.
                    All of the following you are comfortable sharing:
                    {self.personality}
                    Attempt to continue conversations, but do not overly suggest helping the other person.
                    
                    Messages will be formatted in sections.
                    The Images: section will contain transcribed image data.
                    The Memories: section will contain past memories searched.
                    The Message: section should be your primary focus and replied to.
                    
                    Current date: {date.today()}

                    Use the following style for all responses:
                    {self.text_styler.styler_message()}
                    """
                    ),
                    role="system",
                )
            ]
            + messages
        )

    def generate_completion(self, enable_functions=True):
        response: str = self.conversation.get_completion_chat(
            model="gpt-3.5-turbo-0613",
            frequency_penalty=0.25,
            function_call="auto" if enable_functions else "none",
            functions=[
                ChatFunction(
                    name="web_search",
                    description="Perform a web search on a given query. Use this only when given information is not known, and is not personal information. Use only to find general information.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to give the web search engine.",
                            }
                        },
                        "required": ["query"],
                    },
                ),
                ChatFunction(
                    name="memory_search",
                    description="Perform a semantic memory search against a given query to return information from past messages with the user. Use this implicitly, whenever a request seems to require personal information.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to give the memory search engine.",
                            }
                        },
                        "required": ["query"],
                    },
                ),
            ],
        )

        return response

    def web_search(self, arguments: dict[str, str]):
        print(f"Searching the web for {arguments['query']}.")
        results = web_search(query=arguments["query"])
        summary = get_summarization_instruct(
            t.for_(result.snippet for result in results)
        )["choices"][0]["text"]
        return summary

    def memory_search(self, arguments: dict[str, str]):
        print(f"Searching memories for {arguments['query']}.")
        results = "\n".join(
            list(
                semantic_search(
                    query=arguments["query"],
                    embeddings=self.embeddings,
                    transcript=self.memories,
                    top_k=1,  # This may change, depending on accuracy in the future.
                )
            )
        )
        summarizer = Conversation(
            messages=[
                Message(
                    content=t.trim_lines(
                        """Summarize the content provided, and be sure to answer the question in the query.
                        Respond in the following format:
                        Key Points: {key points of the text}
                        Answer: {answer to the user's question}"""
                    ),
                    role="system",
                ),
                Message(
                    content=f"Question: {arguments['query']}\n{results}", role="user"
                ),
            ]
        )
        summary = summarizer.get_completion_chat()["choices"][0]["text"]
        return summary

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
        print(f"--- Inputted Message ---\n{message_content}\n###")

        # TODO: Integrate message transcript semantic search system to determine what memories to search up – if any.
        # TODO: Integrate web search system if common information is unknown.

        self.conversation.add_message(
            message=Message(content=message_content, role="user")
        )
        completion_message = self.generate_completion()["choices"][0]["message"]
        print(
            f"--- Completion Message ---\n{json.dumps(completion_message, indent=4)}\n###"
        )

        if completion_message.get("function_call"):
            function_call = completion_message["function_call"]
            available_functions = {
                "web_search": self.web_search,
                "memory_search": self.memory_search,
            }
            target_function = available_functions[function_call["name"]]
            arguments = json.loads(function_call["arguments"])
            function_response = target_function(arguments)
            print(
                f"--- Function Call ---\nName: {function_call['name']}\nArguments: {function_call['arguments']}\n###"
            )
            self.conversation.add_message(
                message=Message(
                    content=None,
                    function_call=completion_message["function_call"],
                    role="assistant",
                )
            )
            self.conversation.add_message(
                message=Message(
                    content=function_response,
                    name=function_call["name"],
                    role="function",
                )
            )
            # TODO: Make function calling recursive in the future.
            completion_message = self.generate_completion(enable_functions=False)[
                "choices"
            ][0]["message"]

        response = completion_message["content"]
        print(f"--- Text Response ---\n{response}\n###")

        # if any(flag in response.lower() for flag in AI_FLAGS):
        #     print("ChatGPT response: ", response)
        #     print("Using InstructGPT to respond.")
        #     response = self.conversation.get_completion_instruct()["choices"][0]["text"]

        # print(f"--- Pre-Style Message ---\n{response}\n###")
        # if self.text_styler is not None:
        #     response = self.text_styler.apply_style(text=response)
        #     print(f"--- Post-Style Message ---\n{response}\n###")

        assistant_message = Message(content=response, role="assistant")
        self.conversation.add_message(message=assistant_message)

        print(
            f"--- Final Message ---\n{json.dumps(asdict(assistant_message), indent=4)}\n###"
        )

        with open(
            ".memories/conversation.json", "w", encoding="utf8"
        ) as conversation_file:
            json.dump(
                list(asdict(message) for message in self.conversation.messages[1:]),
                conversation_file,
                indent=4,
            )

        return assistant_message
