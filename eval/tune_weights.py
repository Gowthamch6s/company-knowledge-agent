from app.retrieval import search_documents
from app.reranker import rerank_results

from eval.retrieval_tests import TEST_CASES


WEIGHTS = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0),
]


def find_rank(results, expected_section):
    for rank, result in enumerate(results, start=1):
        section = result["chunk"].section_title or ""

        if section.startswith(expected_section):
            return rank

    return None


def evaluate_weights(vector_weight, reranker_weight):
    hit_at_1 = 0
    hit_at_3 = 0
    ranks = []

    for test in TEST_CASES:

        # Stage 1
        candidates = search_documents(
            test["query"],
            limit=10,
        )

        # Stage 2 with selected weights
        results = rerank_results(
            test["query"],
            candidates,
            vector_weight=vector_weight,
            reranker_weight=reranker_weight,
        )

        results = results[:3]

        rank = find_rank(
            results,
            test["expected_section"],
        )

        ranks.append(rank)

        if rank == 1:
            hit_at_1 += 1

        if rank is not None and rank <= 3:
            hit_at_3 += 1

    return hit_at_1, hit_at_3, ranks


def run_experiment():

    print("\nWEIGHT TUNING EXPERIMENT")
    print("=" * 85)

    print(
        f"{'Vector':<10}"
        f"{'Reranker':<12}"
        f"{'Hit@1':<10}"
        f"{'Hit@3':<10}"
        f"Ranks"
    )

    print("-" * 85)

    for vector_weight, reranker_weight in WEIGHTS:

        hit1, hit3, ranks = evaluate_weights(
            vector_weight,
            reranker_weight,
        )

        rank_display = [
            f"#{rank}" if rank is not None else "MISS"
            for rank in ranks
        ]

        print(
            f"{vector_weight:<10.1f}"
            f"{reranker_weight:<12.1f}"
            f"{hit1}/5{'':<7}"
            f"{hit3}/5{'':<7}"
            f"{rank_display}"
        )


if __name__ == "__main__":
    run_experiment()