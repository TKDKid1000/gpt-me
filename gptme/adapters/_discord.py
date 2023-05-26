from gptme.assistant import Assistant
from gptme import __version__
import discord


class DiscordAdapter(discord.Client):
    assistant: Assistant

    def __init__(self, assistant: Assistant, token: str) -> None:
        self.assistant = assistant

    async def on_ready(self):
        print(f"Logged on as {self.user}, emulating using GPT-me v{__version__}.")

    async def on_message(self, message: discord.Message):
        images = [
            attachment.url
            for attachment in message.attachments
            if attachment.content_type.startswith("image/")
        ]

        content = message.clean_content
