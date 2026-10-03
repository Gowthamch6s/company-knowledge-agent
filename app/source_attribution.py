from app.embeddings import create_embedding


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Calculate cosine similarity between two normalized vectors.

    Our embedding function already normalizes embeddings,
    so cosine similarity is simply the dot product.
    """

    return sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )


def select_supporting_sources(
    answer: str,
    evidence: list[dict],
    limit: int = 1,
) -> list[dict]:
    """
    Select the retrieved chunks that are most semantically
    aligned with the generated answer.

    This prevents every retrieved candidate from being
    presented to the user as if it supported the answer.
    """

    if not answer or not evidence:
        return []

    answer_embedding = create_embedding(answer)

    scored_sources = []

    for item in evidence:

        content = item.get("content", "")

        if not content:
            continue

        content_embedding = create_embedding(content)

        score = cosine_similarity(
            answer_embedding,
            content_embedding,
        )

        scored_sources.append(
            {
                **item,
                "support_score": score,
            }
        )

    scored_sources.sort(
        key=lambda item: item["support_score"],
        reverse=True,
    )

    return scored_sources[:limit]
if __name__ == "__main__":

    from app.hybrid_retrieval import hybrid_search

    question = "When does Nexus pay its employees?"

    results = hybrid_search(
        question,
        limit=3,
        candidate_limit=10,
    )

    evidence = []

    for result in results:
        chunk = result["chunk"]

        evidence.append(
            {
                "filename": result["filename"],
                "page": chunk.page_number,
                "section": chunk.section_title,
                "content": chunk.content,
            }
        )

    answer = (
        "Employees are paid on a semi-monthly basis, "
        "on the 15th and final day of each month."
    )

    sources = select_supporting_sources(
        answer,
        evidence,
        limit=3,
    )

    print("\nSOURCE ATTRIBUTION")
    print("=" * 80)

    for rank, source in enumerate(
        sources,
        start=1,
    ):
        print(
            f"#{rank} "
            f"{source['section']} | "
            f"Page {source['page']} | "
            f"Support: {source['support_score']:.4f}"
        )