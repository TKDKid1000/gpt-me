from gptme.conversation import Conversation, Message
from gptme.utils.templates import trim_lines


class TextStyler:
    def extract_style(self, text: str):
        extractor = Conversation(
            [
                Message(
                    content=trim_lines(
                        """Describe the level of the following with exactly one word:
                        Formality: {casual, neutral, polite, formal, professional}
                        Style: {style words}
                        Grammar: {novice, beginner, intermediate, competent, advanced}
                        Punctuation: {always, often, seldom, never}
                        Diction: {diction level}
                        Vocabulary: {basic, intermediate, competent, proficient, advanced}
                        """
                    ),
                    role="system",
                ),
                Message(content=text, role="system"),
            ]
        )

        style = extractor.get_completion_chat()["choices"][0]["message"]["content"]
        return style
