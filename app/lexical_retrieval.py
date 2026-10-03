import re

from rank_bm25 import BM25Okapi

from app.database import SessionLocal
from app.models import DocumentChunk


def tokenize(text: str) -> list[str]:
    """
    Convert text into simple lowercase word tokens.

    Example:

    "Employees are paid semi-monthly."

    becomes:

    [
        "employees",
        "are",
        "paid",
        "semi",
        "monthly",
    ]
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


def lexical_search(
    query: str,
    limit: int = 5,
):
    """
    Search document chunks using BM25 lexical ranking.
    """

    session = SessionLocal()

    try:
        # Load all document chunks.
        chunks = (
            session.query(DocumentChunk)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

        if not chunks:
            return []

        # -----------------------------------------------------
        # Build searchable text for every chunk
        # -----------------------------------------------------

        documents = []

        for chunk in chunks:

            section = chunk.section_title or ""

            searchable_text = (
                f"{section} {chunk.content}"
            )

            documents.append(
                tokenize(searchable_text)
            )

        # -----------------------------------------------------
        # Create BM25 index
        # -----------------------------------------------------

        bm25 = BM25Okapi(documents)

        # Convert user question into tokens.
        query_tokens = tokenize(query)

        # Ask BM25 to score every document.
        scores = bm25.get_scores(query_tokens)

        # -----------------------------------------------------
        # Combine chunks with their scores
        # -----------------------------------------------------

        ranked_results = []

        for chunk, score in zip(
            chunks,
            scores,
        ):
            ranked_results.append(
                {
                    "chunk": chunk,
                    "score": float(score),
                }
            )

        # Highest score first.
        ranked_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked_results[:limit]

    finally:
        session.close()


if __name__ == "__main__":

    query = "When does Nexus pay its employees?"

    results = lexical_search(
        query,
        limit=5,
    )

    print("\nBM25 LEXICAL SEARCH")
    print("=" * 80)

    print(f"Query: {query}")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        chunk = result["chunk"]

        print("\n" + "-" * 80)

        print(f"Rank: #{rank}")

        print(
            f"Section: "
            f"{chunk.section_title}"
        )

        print(
            f"Page: "
            f"{chunk.page_number}"
        )

        print(
            f"BM25 score: "
            f"{result['score']:.4f}"
        )