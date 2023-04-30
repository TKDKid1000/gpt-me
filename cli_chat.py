import os

import openai
from dotenv import load_dotenv

from gptme.assistant import Assistant

load_dotenv(dotenv_path=".env")

openai.api_key = os.environ["OPENAI_KEY"]

with open(os.environ["MEMORY_FILE"]) as memory_file:
    memories = memory_file.read()

with open("personality.txt") as personality_file:
    personality = personality_file.read()


assistant = Assistant(
    memories=memories, personality=personality, name=os.environ["ASSISTANT_NAME"]
)

while True:
    message = input("Katie: ")

    assistant.send_message(text=message)

    print(f"Rhone: {assistant.conversation.messages[-1].content}")
