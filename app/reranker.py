from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


def min_max_normalize(values):
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [1.0 for _ in values]

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


def rerank_results(
    query: str,
    results,
    vector_weight: float = 0.5,
    reranker_weight: float = 0.5,
):
    if not results:
        return []

    pairs = []
    vector_similarities = []

    for result in results:
        chunk = result[0]
        distance = result[2]

        pairs.append(
            [query, chunk.content]
        )

        vector_similarities.append(
            1 - distance
        )

    reranker_scores = reranker.predict(pairs)

    normalized_vectors = min_max_normalize(
        vector_similarities
    )

    normalized_reranker = min_max_normalize(
        [float(score) for score in reranker_scores]
    )

    reranked = []

    for (
        result,
        vector_similarity,
        reranker_score,
        normalized_vector,
        normalized_cross_encoder,
    ) in zip(
        results,
        vector_similarities,
        reranker_scores,
        normalized_vectors,
        normalized_reranker,
    ):

        combined_score = (
            vector_weight * normalized_vector
            + reranker_weight * normalized_cross_encoder
        )

        reranked.append(
            {
                "chunk": result[0],
                "filename": result[1],
                "vector_distance": result[2],
                "vector_similarity": vector_similarity,
                "reranker_score": float(reranker_score),
                "combined_score": combined_score,
            }
        )

    reranked.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    return reranked