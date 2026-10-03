from app.retrieval import search_documents
from app.query_expansion import expand_query


QUESTIONS = [
    "How often are employees paid?",
    "Can employees work remotely?",
    "How should employees report harassment?",
]


for question in QUESTIONS:
    print("\n" + "=" * 100)
    print(f"QUESTION: {question}")
    print(f"EXPANDED: {expand_query(question)}")
    print("=" * 100)

    results = search_documents(
        question,
        limit=3,
        use_expansion=True,
    )

    for rank, result in enumerate(results, start=1):
        chunk = result[0]
        distance = result[2]

        print(
            f"\n#{rank} "
            f"{chunk.section_title} "
            f"| Similarity: {1 - distance:.4f}"
        )

        print("-" * 100)
        print(chunk.content)