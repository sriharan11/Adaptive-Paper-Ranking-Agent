from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request as libreq

import feedparser
from paper import Paper


MIN_REQUEST_INTERVAL = 3.0
MAX_REQUEST_ATTEMPTS = 3
RATE_LIMIT_RETRY_DELAY = 30.0
SERVER_RETRY_DELAY = 10.0
RETRYABLE_HTTP_CODES = {429, 502, 503, 504}
REQUEST_TIME_FILE = Path(tempfile.gettempdir()) / "adaptive_paper_arxiv_request_time"
_last_request_time: float | None = None


def _wait_for_request_slot() -> None:
    global _last_request_time

    request_times = []
    if _last_request_time is not None:
        request_times.append(_last_request_time)

    try:
        request_times.append(float(REQUEST_TIME_FILE.read_text(encoding="utf-8")))
    except (FileNotFoundError, OSError, ValueError):
        pass

    if request_times:
        elapsed = max(0.0, time.time() - max(request_times))
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    _last_request_time = time.time()
    try:
        REQUEST_TIME_FILE.write_text(str(_last_request_time), encoding="utf-8")
    except OSError:
        pass


def _fetch_arxiv(url: str) -> bytes:
    last_error = None

    for attempt in range(MAX_REQUEST_ATTEMPTS):
        _wait_for_request_slot()

        try:
            with libreq.urlopen(url) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_HTTP_CODES:
                raise

            last_error = error
            if attempt == MAX_REQUEST_ATTEMPTS - 1:
                break

            retry_delay = (
                RATE_LIMIT_RETRY_DELAY
                if error.code == 429
                else SERVER_RETRY_DELAY
            )
            if error.code == 429 and error.headers:
                try:
                    retry_delay = max(
                        retry_delay,
                        float(
                            error.headers.get(
                                "Retry-After", RATE_LIMIT_RETRY_DELAY
                            )
                        ),
                    )
                except (TypeError, ValueError):
                    pass

            print(
                f"arXiv temporarily unavailable ({error.code}); "
                f"retrying in {retry_delay:g} seconds..."
            )
            time.sleep(retry_delay)

    raise RuntimeError(
        "arXiv is temporarily unavailable after "
        f"{MAX_REQUEST_ATTEMPTS} attempts "
        f"(last HTTP status: {last_error.code})."
    ) from last_error


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
