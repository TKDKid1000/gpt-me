from dataclasses import dataclass
from urllib.parse import parse_qs, quote, urlparse

import requests
from bs4 import BeautifulSoup

import re
from gptme.conversation import Conversation, Message


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def web_search(query: str):
    url = "https://html.duckduckgo.com/html/?q=" + quote(query)
    html = requests.get(url, headers={"user-agent": "gpt-me/0.0.1"}).text
    soup = BeautifulSoup(html, "html.parser")

    results: list[SearchResult] = []

    for result in soup.select("div.web-result"):
        title_el = result.select_one("a.result__a")
        if title_el is None:
            continue

        url = parse_qs(urlparse(title_el.attrs["href"]).query)["uddg"][0]

        snippet_el = result.select_one("a.result__snippet")
        if snippet_el is None:
            continue

        results.append(SearchResult(title_el.get_text(), url, snippet_el.get_text()))

    return results

def reduce_whitespace(text: str):
    return re.sub(r"[\s]+", ' ', text, 0)

def web_question_answer(url: str, question: str):
    html = requests.get(url, headers={"user-agent": "gpt-me/0.0.1"}).text
    soup = BeautifulSoup(html, "html.parser")

    text = reduce_whitespace(soup.get_text())
    print(text)

    summarizer = Conversation(
        messages=[
            Message(
                content="""You are a web question answering tool. Answer the provided question, and include the exact paragraph from which you got the answer.
                Respond in a format like this:
                Answer: (your generated answer)
                Source: (paragraph/section of text from which the answer was generated)""",
                role="system",
            ),
            Message(
                content=f"""{text}
                
                Question: {question}""",
                role="user",
            ),
        ]
    )

    response = summarizer.get_completion_chat(max_tokens=100)

    return response
