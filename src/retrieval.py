import urllib.parse
import urllib.request as libreq
import feedparser
from paper import Paper

query = input("Enter your research question or query: ").strip()

# Encode spaces and special characters so the query is safe to include in a URL.
params = urllib.parse.urlencode({
    "search_query": f"all:{query}",
    "start": 0,
    "max_results": 5,
})

with libreq.urlopen(f"https://export.arxiv.org/api/query?{params}") as response:
    # Convert the response from raw bytes into readable text.
    result = response.read()
feed = feedparser.parse(result)
papers = []

for entry in feed.entries:
    paper = Paper(
        arxiv_id=entry.id,
        paper_title=entry.title,
        abstract=entry.summary,
        year=entry.published[:4],
        author_names=[author.name for author in entry.authors],
    )
    papers.append(paper)

for paper in papers:
    print(paper.arxiv_id)
    print(paper.paper_title)
    print(paper.abstract)
    print(paper.year)
    print(paper.author_names)
    print()

