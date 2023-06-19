from dataclasses import dataclass
from urllib.parse import parse_qs, quote, urlparse

import requests
from bs4 import BeautifulSoup

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


def web_summarize(url: str, question: str):
    html = requests.get(url, headers={"user-agent": "gpt-me/0.0.1"}).text
    soup = BeautifulSoup(html, "html.parser")

    summary = soup.get_text()

    # answer = flan_t5_large(f"Q: {question}\n\n{summary}")

    # return answer
