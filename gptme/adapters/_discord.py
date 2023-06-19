import discord

from gptme import __version__
from gptme.assistant import Assistant


class DiscordAdapter(discord.Client):
    assistant: Assistant
    channel_id: str
    cooldown: int

    def __init__(self, assistant: Assistant, channel_id: str) -> None:
        super().__init__()
        self.assistant = assistant
        self.channel_id = channel_id

    async def on_ready(self):
        print(f"Logged on as {self.user}, emulating using GPT-me v{__version__}.")

    async def on_message(self, message: discord.Message):
        if message.channel.id != self.channel_id or message.author.id == self.user.id:
            return

        images = [
            attachment.url
            for attachment in message.attachments
            if attachment.content_type.startswith("image/")
        ]

        content = message.clean_content

        async with message.channel.typing():
            response = self.assistant.send_message(text=content, images=images)

            await message.channel.send(response.content)
