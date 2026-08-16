import spacy
from retrieval import retrieve_papers


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


nlp = spacy.load("en_core_web_md")
question = input("Enter your research question: ").strip()
question_doc = nlp(question)
question_concepts = extract_concepts(question_doc)
concept_phrases = [" ".join(concept) for concept in question_concepts]
search_query = " AND ".join(
    f'all:"{phrase}"' for phrase in concept_phrases
)
print("Search query:", search_query)

papers = retrieve_papers(search_query)
for paper in papers:
    paper_text = paper.paper_title + " " + paper.abstract
    paper_doc = nlp(paper_text)
    similarity = question_doc.similarity(paper_doc)
    paper_concepts = set(extract_concepts(paper_doc))
    matched_concepts = sum(
        concept in paper_concepts for concept in question_concepts
    )
    concept_coverage = (
        matched_concepts / len(question_concepts) if question_concepts else 0.0
    )

    print(paper.paper_title)
    print("Semantic similarity:", similarity)
    print("Concept coverage:", concept_coverage)
    print()
