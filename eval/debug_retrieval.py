from app.retrieval import search_documents
from app.reranker import rerank_results


QUERY = "Can employees work from home?"


def debug_query():
    print("\nQUERY:")
    print(QUERY)

    print("\n" + "=" * 70)
    print("STAGE 1 — PGVECTOR CANDIDATES")
    print("=" * 70)

    candidates = search_documents(
        QUERY,
        limit=10,
    )

    for rank, result in enumerate(candidates, start=1):
        chunk = result[0]
        distance = result[2]

        similarity = 1 - distance

        print(
            f"#{rank:<2} "
            f"Section: {chunk.section_title} | "
            f"Similarity: {similarity:.4f}"
        )

    print("\n" + "=" * 70)
    print("STAGE 2 — AFTER RERANKING")
    print("=" * 70)

    reranked = rerank_results(
        QUERY,
        candidates,
    )

    for rank, result in enumerate(reranked, start=1):
        chunk = result["chunk"]

        print(
    f"#{rank:<2} "
    f"Section: {chunk.section_title} | "
    f"Vector: {result['vector_similarity']:.4f} | "
    f"Reranker: {result['reranker_score']:.4f} | "
    f"Combined: {result['combined_score']:.4f}"
)


if __name__ == "__main__":
    debug_query()