import urllib.parse
import urllib.request as libreq
import feedparser
from paper import Paper


def retrieve_papers(search_query: str) -> list[Paper]:
    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": 0,
        "max_results": 10,
    })

    with libreq.urlopen(f"https://export.arxiv.org/api/query?{params}") as response:
        result = response.read()

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
