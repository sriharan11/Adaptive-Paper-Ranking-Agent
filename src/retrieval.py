import time
import urllib.error
import urllib.parse
import urllib.request as libreq

import feedparser
from paper import Paper


MIN_REQUEST_INTERVAL = 3.0
RETRY_DELAY = 5.0
_last_request_time: float | None = None


def _fetch_arxiv(url: str) -> bytes:
    global _last_request_time

    for attempt in range(2):
        if _last_request_time is not None:
            elapsed = time.monotonic() - _last_request_time
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        _last_request_time = time.monotonic()

        try:
            with libreq.urlopen(url) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 1:
                raise

            print("arXiv rate limit reached; waiting 5 seconds before retrying once.")
            time.sleep(RETRY_DELAY)

    raise RuntimeError("arXiv request failed")


def retrieve_papers(search_query: str) -> list[Paper]:
    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": 0,
        "max_results": 10,
    })

    url = f"https://export.arxiv.org/api/query?{params}"
    result = _fetch_arxiv(url)

    feed = feedparser.parse(result)
    papers: list[Paper] = []

    for entry in feed.entries:
        paper = Paper(
            arxiv_id=entry.id,
            paper_title=entry.title,
            abstract=entry.summary,
            year=entry.published[:4],
            author_names=[author.name for author in entry.authors],
        )
        papers.append(paper)

    return papers
