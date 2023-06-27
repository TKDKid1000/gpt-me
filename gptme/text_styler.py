import re
from typing import Literal, TypedDict

import gptme.utils.templates as t
from gptme.conversation import Conversation, Message


class Style(TypedDict):
    formality: Literal["casual", "neutral", "polite", "formal", "professional"]
    style: str
    grammar: Literal["novice", "beginner", "intermediate", "competent", "advanced"]
    punctuation: Literal["always", "often", "seldom", "never"]
    diction: str
    vocabulary: Literal["basic", "intermediate", "competent", "proficient", "advanced"]
    contractions: Literal["always", "often", "seldom", "never"]
    ellipsis: Literal["always", "often", "seldom", "never"]
    proper_capitalization: Literal["yes", "no"]
    mood: str
    abbreviations: Literal["always", "often", "seldom", "never"]


class NoStyleProvidedError(Exception):
    pass


class TextStyler:
    style: Style

    def extract_style(self, text: str):
        extractor = Conversation(
            [
                Message(
                    content=t.trim_lines(
                        """Describe the following from the sample text.
                        Do not include commentary.
                        Use options if specified.

                        Formality: {casual, neutral, polite, formal, professional}
                        Style: {five words to describe style}
                        Grammar: {novice, beginner, intermediate, competent, advanced}
                        Punctuation: {always, often, sometimes, seldom, never}
                        Diction: {three words to describe diction}
                        Vocabulary: {basic, intermediate, competent, proficient, advanced}
                        Contractions: {always, often, sometimes, seldom, never}
                        Ellipsis: {always, often, sometimes, seldom, never}
                        Proper Capitalization: {yes, no}
                        Mood: {five words to describe mood}
                        Abbreviations: {always, often, sometimes, seldom, never}
                        """
                    ),
                    role="system",
                ),
                Message(content=text, role="user"),
            ]
        )

        style_text: str = extractor.get_completion_chat(
            temperature=0.7, frequency_penalty=1
        )["choices"][0]["message"]["content"]
        matches = re.findall(r"^(.+): (.+)", style_text.lower(), re.MULTILINE)
        style: Style = {
            match[0].replace(" ", "_").strip(): match[1].strip() for match in matches
        }
        self.style = style

        return style

    def styler_message(self):
        return "\n".join(
            f"{category.upper()}: {self.style[category]}" for category in self.style
        )

    def apply_style(self, text):
        if self.style is None:
            raise NoStyleProvidedError("Style not found.")
        styler = Conversation(
            [
                Message(
                    content=t.join_(
                        "Rewrite the message and use this following style:\n",
                        t.for_(
                            f"{category.upper()}: {self.style[category]}"
                            for category in self.style
                        ),
                    ),
                    role="system",
                ),
                Message(content=text, role="system"),
            ]
        )

        styled: str = styler.get_completion_chat()["choices"][0]["message"]["content"]

        if self.style["proper_capitalization"] == "no":
            styled = styled.lower()

        return styled
