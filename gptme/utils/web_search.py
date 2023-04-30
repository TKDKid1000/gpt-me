from urllib.parse import parse_qs, quote, urlparse

import requests
from bs4 import BeautifulSoup, Tag


def web_search(query: str):
    url = "https://html.duckduckgo.com/html/?q=" + quote(query)
    print(url)
    html = requests.get(url, headers={"user-agent": "gpt-me/0.0.1"}).text
    soup = BeautifulSoup(html, "html.parser")

    results = []

    for result in soup.select("div.web-result"):
        search_result = {}
        title_el = result.select_one("a.result__a")
        if title_el is None:
            continue
        search_result["title"] = title_el.get_text()

        url = parse_qs(urlparse(title_el.attrs["href"]).query)["uddg"][0]
        search_result["url"] = url

        snippet_el = result.select_one("a.result__snippet")
        if snippet_el is None:
            continue
        search_result["snippet"] = snippet_el.get_text()

        results.append(search_result)

    return results
