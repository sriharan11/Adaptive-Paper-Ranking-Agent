# Adaptive Paper Ranking Agent

An explainable research-paper retrieval and ranking MVP built for the **LEC AI Engineering Intern build assessment**.

## Project overview

The program accepts a natural-language research question, extracts its main concepts with spaCy, and converts those concepts into an arXiv search query. It requests a batch of 10 papers from the arXiv public API and performs one adaptive broadening step if the initial search is too narrow.

Every returned paper is scored using four signals:

- NLI directness: whether the abstract supports a hypothesis generated from the question.
- Concept similarity: how closely the question's noun-chunk concepts match concepts in the title and abstract.
- Semantic similarity: whole-question similarity against the combined paper title and abstract.
- Recency: a deliberately small preference for newer papers.

The system combines these signals into a ranked result list and produces concise, deterministic trade-off explanations for the top three papers. It is a small, inspectable assessment project rather than a production-ready research system.

## Architecture and flow

```text
research question
    -> spaCy noun-chunk concept extraction
    -> arXiv AND query construction
    -> request a batch of 10 papers from arXiv
    -> adaptively broaden once if the search is too narrow
    -> calculate NLI, concept, semantic, and recency scores
    -> final weighted ranking
    -> concise top-3 trade-off explanations
```

The source files have focused responsibilities:

- `src/ranking.py`: user input, concept extraction, adaptive query decision, scoring, ranking, and output.
- `src/retrieval.py`: rate-limited arXiv requests, Atom-feed parsing, and `Paper` creation.
- `src/paper.py`: the `Paper` dataclass used to hold retrieved metadata.

## Ranking criteria

The final score is calculated as:

```text
final_score =
    0.55 * nli_directness
    + 0.20 * concept_similarity
    + 0.15 * semantic_similarity
    + 0.10 * recency_score
```

| Signal | Weight | Purpose |
| --- | ---: | --- |
| NLI directness | 55% | Tests whether the abstract entails a concrete hypothesis derived from the user's question. |
| Concept similarity | 20% | Rewards papers whose noun-chunk concepts semantically match the question's concepts. |
| Semantic similarity | 15% | Measures broad similarity between the complete question and the paper's title plus abstract. |
| Recency | 10% | Provides a modest preference for newer work without allowing age to dominate relevance. |

NLI receives the highest weight because it is the signal most directly aimed at whether a paper addresses the question's intent, rather than merely sharing the same topic or vocabulary.

Concept similarity compares each normalized question concept with every normalized paper concept using spaCy vectors. The best paper-concept match is retained for each question concept, and those best matches are averaged. Concept pairs without usable vectors are skipped.

Recency uses a linear 10-year window. A paper from the current year scores `1.0`, the score falls by `0.1` per year, and papers at least 10 years old score `0.0`. Missing, invalid, or future years safely receive `0.0`.

## Adaptive search behaviour

The second search is **not a fixed fallback query**. The program first builds an arXiv query by joining normalized noun-chunk concepts with `AND`, then requests a batch of 10 papers from arXiv.

If fewer than 10 papers are returned, the search is classified as `TOO NARROW` and broadened once:

1. Look for concepts containing one of the normalized generic terms `technique`, `method`, `approach`, `strategy`, or `way`.
2. If several generic concepts exist, remove the shortest normalized phrase.
3. If none is generic, remove the shortest concept overall.
4. Rebuild the `AND` query from the remaining concepts.
5. Call arXiv again and use the second batch for scoring and ranking.

There is intentionally no multi-step refinement loop yet. arXiv calls are spaced by at least three seconds, including across script restarts. HTTP `429`, `502`, `503`, and `504` responses receive up to three total attempts. Retries wait at least 30 seconds for `429` (or a longer `Retry-After` value) and 10 seconds for `502`, `503`, or `504`.

## Models and tools

- **spaCy `en_core_web_md`** for tokenization, lemmatization, noun chunks, vectors, and semantic similarity.
- **Hugging Face Transformers** for the zero-shot/NLI pipeline.
- **`typeform/distilbert-base-uncased-mnli`** for abstract-to-hypothesis directness scoring.
- **feedparser** for parsing the arXiv Atom response.
- **arXiv public API** for paper metadata.
- Python standard-library `urllib` for HTTP requests and URL encoding.

The hypothesis builder uses a few transparent templates for questions beginning with “what techniques,” “what methods,” “what approaches,” “how does,” or “how do.” Other questions use a generic direct-address fallback.

## Setup on Windows

Python 3.10 or newer is recommended.

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the spaCy language model separately because it is not included in `requirements.txt`:

```bat
python -m spacy download en_core_web_md
```

Run the actual program entry point from the repository root:

```bat
python .\src\ranking.py
```

Do not run `src/retrieval.py` as the entry point; it contains reusable retrieval functions and therefore exits without prompting for a question. The Hugging Face model is downloaded on its first use, so the initial run can take longer and requires an internet connection.

## Development example

Example question:

```text
What techniques can reduce hallucinations in large language models when answering medical questions?
```

spaCy extracts normalized noun-chunk concepts from the question and the program joins them into a restrictive arXiv `AND` query. If that initial query returns fewer than 10 papers, the generic/low-priority concept heuristic removes one concept, prints the broader query, and retrieves a second batch before scoring.

## Engineering decisions and trade-offs

- **Whole-document spaCy similarity was too broad on its own.** It remains useful as a supporting signal, but semantically related papers can receive similarly high values without directly answering the question.
- **Exact concept equality was too brittle.** Inflection and phrasing differences caused reasonable matches to be missed, so concepts are normalized and compared with vector similarity instead.
- **Concrete-hypothesis NLI better represents directness.** Testing each abstract against a question-derived statement is more aligned with query intent than generic labels such as “relevant.”
- **Recency is intentionally low-weight.** A newer but weakly relevant paper should not displace an older paper that addresses the question directly.
- **Explainability is preferred over unnecessary complexity.** The adaptive rule, weighted score, and top-three explanations use visible Python heuristics rather than an opaque orchestration framework or another explanation-generating model.

## Limitations

- Concept-removal and hypothesis-generation heuristics cover only simple language patterns.
- spaCy similarity scores can cluster at high values and may not separate close candidates strongly.
- NLI directness is imperfect and evaluates only the paper abstract, not the full text.
- arXiv availability and rate limits can delay or prevent a run despite retry handling.
- No citation-based evidence or citation-network signal is included.
- Ranking weights are manually selected rather than learned or validated on a labelled benchmark.
- A broadened second query can still return fewer than 10 papers.

## Future work

- Add citation counts and retrieve papers that cite the strongest candidates.
- Improve adaptive query refinement beyond a single concept-removal step.
- Learn or validate ranking weights on relevance judgements.
- Evaluate stronger sentence embeddings or dedicated rerankers.
- Cache arXiv results to reduce repeated API calls and improve resilience.
- Test retrieval quality and ranking behaviour across broader research domains.

## Three-minute demo

The demo will show:

1. Entering a natural-language research question.
2. Inspecting the initial concept-based arXiv query.
3. Observing the `TOO NARROW` decision and broader second query when triggered.
4. Reviewing the final weighted ranking and its component scores.
5. Reading the concise strength, trade-off, and decision explanations for the top three papers.
