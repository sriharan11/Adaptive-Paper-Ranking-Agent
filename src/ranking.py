import spacy
from retrieval import retrieve_papers
from transformers import pipeline


GENERIC_TERMS = {"technique", "method", "approach", "strategy", "way"}


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

    scored_papers.append(
        {
            "title": paper.paper_title,
            "abstract": paper.abstract,
            "semantic_similarity": similarity,
            "concept_similarity": concept_similarity_score,
        }
    )

top_papers = sorted(
    scored_papers,
    key=lambda paper: paper["concept_similarity"],
    reverse=True,
)[:5]

for rank, paper in enumerate(top_papers, start=1):
    print("Rank:", rank)
    print("Title:", paper["title"])
    print("Semantic similarity:", paper["semantic_similarity"])
    print("Concept similarity:", paper["concept_similarity"])
    print("Abstract:", paper["abstract"])
    print()

experiment_titles = [
    "Optimizing Medical Question-Answering Systems",
    "MedHallu",
]
experiment_hypothesis = build_hypothesis(question)
print("Generated hypothesis:", experiment_hypothesis)
experiment_papers = [
    paper
    for title in experiment_titles
    for paper in papers
    if title.lower() in paper.paper_title.lower()
]

if experiment_papers:
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
    )

    for paper in experiment_papers:
        classification = zero_shot_classifier(
            paper.abstract,
            candidate_labels=[experiment_hypothesis],
            hypothesis_template="{}",
            multi_label=True,
        )
        entailment_score = classification["scores"][0]

        print("Paper:", paper.paper_title)
        print("Hypothesis tested:", experiment_hypothesis)
        print("Entailment/relevance score:", entailment_score)
