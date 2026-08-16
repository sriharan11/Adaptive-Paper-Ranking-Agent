from dataclasses import dataclass


@dataclass
class Paper:
    arxiv_id: str
    paper_title: str
    abstract: str
    year: str
    author_names: list[str]
