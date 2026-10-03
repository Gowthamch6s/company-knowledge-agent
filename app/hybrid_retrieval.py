from app.retrieval import search_documents
from app.lexical_retrieval import lexical_search


def hybrid_search(
    query: str,
    limit: int = 5,
    candidate_limit: int = 10,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
):
    """
    Combine vector retrieval and BM25 lexical retrieval
    using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    query:
        User's natural-language question.

    limit:
        Number of final hybrid results to return.

    candidate_limit:
        Number of candidates retrieved independently
        from vector search and BM25.

    rrf_k:
        Constant used by Reciprocal Rank Fusion.
        A common default is 60.
    """

    # ---------------------------------------------------------
    # STAGE 1 — VECTOR RETRIEVAL
    # ---------------------------------------------------------

    vector_results = search_documents(
        query,
        limit=candidate_limit,
        use_expansion=True,
    )

    # ---------------------------------------------------------
    # STAGE 2 — BM25 RETRIEVAL
    # ---------------------------------------------------------

    bm25_results = lexical_search(
        query,
        limit=candidate_limit,
    )

    # ---------------------------------------------------------
    # STAGE 3 — RECIPROCAL RANK FUSION
    # ---------------------------------------------------------
    #
    # We will store every unique chunk here.
    #
    # Key:
    #     chunk.id
    #
    # Value:
    #     information about that chunk and its rankings
    # ---------------------------------------------------------

    fused = {}

    # ---------------------------------------------------------
    # Add vector rankings
    # ---------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        chunk = result[0]
        filename = result[1]
        distance = result[2]

        chunk_id = chunk.id

        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk": chunk,
                "filename": filename,
                "vector_rank": None,
                "vector_similarity": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        fused[chunk_id]["vector_rank"] = rank

        fused[chunk_id]["vector_similarity"] = float(
            1 - distance
        )

        # RRF contribution from vector search.
        fused[chunk_id]["rrf_score"] += (
            vector_weight/ (rrf_k + rank)
        )

    # ---------------------------------------------------------
    # Add BM25 rankings
    # ---------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        chunk = result["chunk"]
        bm25_score = result["score"]

        chunk_id = chunk.id

        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk": chunk,
                "filename": None,
                "vector_rank": None,
                "vector_similarity": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        fused[chunk_id]["bm25_rank"] = rank
        fused[chunk_id]["bm25_score"] = bm25_score

        # RRF contribution from BM25.
        fused[chunk_id]["rrf_score"] += (
            bm25_weight / (rrf_k + rank)
        )

    # ---------------------------------------------------------
    # STAGE 4 — SORT BY FUSED SCORE
    # ---------------------------------------------------------

    ranked_results = list(
        fused.values()
    )

    ranked_results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:limit]


# =============================================================
# MANUAL TEST
# =============================================================

if __name__ == "__main__":

    question = (
        "When does Nexus pay its employees?"
    )

    results = hybrid_search(
        question,
        limit=5,
        candidate_limit=10,
    )

    print("\nHYBRID RETRIEVAL")
    print("=" * 90)

    print(f"Query: {question}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        chunk = result["chunk"]

        print("\n" + "-" * 90)

        print(f"FINAL RANK: #{rank}")

        print(
            f"Section: "
            f"{chunk.section_title}"
        )

        print(
            f"Page: "
            f"{chunk.page_number}"
        )

        print(
            f"Vector rank: "
            f"{result['vector_rank']}"
        )

        print(
            f"Vector similarity: "
            f"{result['vector_similarity']}"
        )

        print(
            f"BM25 rank: "
            f"{result['bm25_rank']}"
        )

        print(
            f"BM25 score: "
            f"{result['bm25_score']}"
        )

        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
        )