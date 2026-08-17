from datetime import date

import spacy
from retrieval import retrieve_papers
from transformers import pipeline


GENERIC_TERMS = {"technique", "method", "approach", "strategy", "way"}
SCORE_COMPONENTS = [
    ("nli_directness", "NLI directness", 0.55),
    ("concept_similarity", "concept similarity", 0.20),
    ("semantic_similarity", "semantic similarity", 0.15),
    ("recency_score", "recency", 0.10),
]


def extract_concepts(doc) -> list[tuple[str, ...]]:
    concepts = []

    for chunk in doc.noun_chunks:
        concept = tuple(
            token.lemma_.lower()
            for token in chunk
            if not token.is_stop and not token.is_punct
        )
        if concept:
            concepts.append(concept)

    return concepts


def build_search_query(concepts: list[tuple[str, ...]]) -> str:
    concept_phrases = [" ".join(concept) for concept in concepts]
    return " AND ".join(f'all:"{phrase}"' for phrase in concept_phrases)


def choose_concept_to_remove(
    concepts: list[tuple[str, ...]],
) -> tuple[str, ...]:
    generic_concepts = [
        concept
        for concept in concepts
        if any(term in concept for term in GENERIC_TERMS)
    ]
    candidates = generic_concepts or concepts
    return min(candidates, key=lambda concept: len(" ".join(concept)))


def calculate_concept_similarity(question_concepts, paper_concepts) -> float:
    if not question_concepts or not paper_concepts:
        return 0.0

    best_match_scores = []
    for question_concept in question_concepts:
        pair_scores = [
            question_concept.similarity(paper_concept)
            for paper_concept in paper_concepts
            if question_concept.has_vector and paper_concept.has_vector
        ]
        best_match = max(pair_scores, default=0.0)
        best_match_scores.append(best_match)

    return sum(best_match_scores) / len(best_match_scores)


def build_hypothesis(question: str) -> str:
    cleaned_question = question.strip().removesuffix("?").strip()
    lowercase_question = cleaned_question.lower()

    concept_starts = {
        "what techniques": "techniques",
        "what methods": "methods",
        "what approaches": "approaches",
    }
    for question_start, concept_name in concept_starts.items():
        if lowercase_question.startswith(question_start):
            remainder = cleaned_question[len(question_start):].strip()
            return (
                f"This paper presents or evaluates {concept_name} "
                f"that {remainder}."
            )

    for question_start in ("how does", "how do"):
        if lowercase_question.startswith(question_start):
            remainder = cleaned_question[len(question_start):].strip()
            return f"This paper investigates {remainder}."

    return (
        "This paper directly addresses the research question: "
        f"{cleaned_question}"
    )


def calculate_recency(year) -> tuple[int | None, float]:
    try:
        publication_year = int(year)
    except (TypeError, ValueError):
        return None, 0.0

    current_year = date.today().year
    if publication_year < 1900 or publication_year > current_year:
        return None, 0.0

    age = current_year - publication_year
    recency_score = max(0.0, 1.0 - age / 10)
    return publication_year, recency_score


def build_justification(ranked_papers, index: int) -> dict[str, str]:
    paper = ranked_papers[index]
    relevance_components = SCORE_COMPONENTS[:3]
    strongest_key, strongest_label, _ = max(
        relevance_components,
        key=lambda component: component[2] * paper[component[0]],
    )
    strength = (
        f"{strongest_label} is its strongest weighted relevance "
        f"signal at {paper[strongest_key]:.3f}."
    )

    if len(ranked_papers) == 1:
        weakest_key, weakest_label, _ = min(
            relevance_components,
            key=lambda component: paper[component[0]],
        )
        return {
            "strength": strength,
            "trade_off": (
                f"Its weakest signal is {weakest_label} at "
                f"{paper[weakest_key]:.3f}."
            ),
            "decision": "It ranks first because it is the only retrieved paper.",
        }

    neighbour = ranked_papers[1] if index == 0 else ranked_papers[index - 1]
    disadvantages = [
        (weight * (neighbour[key] - paper[key]), key, label)
        for key, label, weight in SCORE_COMPONENTS
        if neighbour[key] > paper[key]
    ]

    if disadvantages:
        _, disadvantage_key, disadvantage_label = max(disadvantages)
        trade_off = (
            f'It trails {neighbour["title"]} most on {disadvantage_label} '
            f'({paper[disadvantage_key]:.3f} vs '
            f'{neighbour[disadvantage_key]:.3f}).'
        )
    else:
        weakest_key, weakest_label, _ = min(
            relevance_components,
            key=lambda component: paper[component[0]],
        )
        trade_off = (
            f"Its weakest signal is {weakest_label} at "
            f"{paper[weakest_key]:.3f}, though it does not trail the nearest "
            "competitor there."
        )

    if index == 0:
        advantages = [
            (weight * (paper[key] - neighbour[key]), key, label)
            for key, label, weight in SCORE_COMPONENTS
            if paper[key] > neighbour[key]
        ]
        if advantages:
            _, _, advantage_label = max(advantages)
            decision = (
                f'Its stronger {advantage_label} leaves its final score above '
                f'{neighbour["title"]} ({paper["final_score"]:.3f} vs '
                f'{neighbour["final_score"]:.3f}).'
            )
        else:
            decision = (
                f'It ties {neighbour["title"]} at '
                f'{paper["final_score"]:.3f} and retains the earlier sort '
                "position."
            )
    else:
        decision = (
            f'The weighted trade-off leaves it below {neighbour["title"]} '
            f'({paper["final_score"]:.3f} vs '
            f'{neighbour["final_score"]:.3f}).'
        )

    return {
        "strength": strength,
        "trade_off": trade_off,
        "decision": decision,
    }


nlp = spacy.load("en_core_web_md")
question = input("Enter your research question: ").strip()
question_doc = nlp(question)
question_concepts = extract_concepts(question_doc)
question_concept_docs = [nlp(" ".join(concept)) for concept in question_concepts]
search_query = build_search_query(question_concepts)
print("Search query:", search_query)

papers = retrieve_papers(search_query)
if len(papers) < 10:
    print("Search was TOO NARROW.")
    removed_concept = choose_concept_to_remove(question_concepts)
    broader_concepts = question_concepts.copy()
    broader_concepts.remove(removed_concept)
    broader_query = build_search_query(broader_concepts)

    print("Removed concept:", " ".join(removed_concept))
    print("Broader query:", broader_query)
    papers = retrieve_papers(broader_query)

hypothesis = build_hypothesis(question)
print("Generated hypothesis:", hypothesis)
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
)

scored_papers = []
for paper in papers:
    paper_text = paper.paper_title + " " + paper.abstract
    paper_doc = nlp(paper_text)
    similarity = question_doc.similarity(paper_doc)
    paper_concepts = extract_concepts(paper_doc)
    paper_concept_docs = [nlp(" ".join(concept)) for concept in paper_concepts]
    concept_similarity_score = calculate_concept_similarity(
        question_concept_docs, paper_concept_docs
    )
    classification = zero_shot_classifier(
        paper.abstract,
        candidate_labels=[hypothesis],
        hypothesis_template="{}",
        multi_label=True,
    )
    nli_directness_score = classification["scores"][0]
    publication_year, recency_score = calculate_recency(paper.year)
    final_score = (
        0.55 * nli_directness_score
        + 0.20 * concept_similarity_score
        + 0.15 * similarity
        + 0.10 * recency_score
    )

    scored_papers.append(
        {
            "title": paper.paper_title,
            "abstract": paper.abstract,
            "year": publication_year,
            "semantic_similarity": similarity,
            "concept_similarity": concept_similarity_score,
            "nli_directness": nli_directness_score,
            "recency_score": recency_score,
            "final_score": final_score,
        }
    )

ranked_papers = sorted(
    scored_papers,
    key=lambda paper: paper["final_score"],
    reverse=True,
)

for rank, paper in enumerate(ranked_papers, start=1):
    display_year = paper["year"] if paper["year"] is not None else "Unknown"
    print(f'{rank}. {paper["title"]} ({display_year})')
    print(
        f'   Final: {paper["final_score"]:.3f} | '
        f'NLI: {paper["nli_directness"]:.3f} | '
        f'Concept: {paper["concept_similarity"]:.3f} | '
        f'Semantic: {paper["semantic_similarity"]:.3f} | '
        f'Recency: {paper["recency_score"]:.3f}'
    )

    if rank <= 3:
        justification = build_justification(ranked_papers, rank - 1)
        print("   - Strength:", justification["strength"])
        print("   - Trade-off:", justification["trade_off"])
        print("   - Decision:", justification["decision"])

    print()
