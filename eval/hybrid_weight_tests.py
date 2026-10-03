from app.hybrid_retrieval import hybrid_search
from eval.hybrid_retrieval_tests import (
    TEST_CASES,
    section_matches,
)


WEIGHTS = [
    (1.0, 1.0),
    (0.9, 1.1),
    (0.8, 1.2),
    (0.7, 1.3),
    (0.6, 1.4),
]


def find_rank(
    question,
    expected,
    vector_weight,
    bm25_weight,
):
    results = hybrid_search(
        question,
        limit=5,
        candidate_limit=10,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        chunk = result["chunk"]

        if section_matches(
            chunk.section_title,
            expected,
        ):
            return rank

    return None


def main():
    print("\nHYBRID WEIGHT EXPERIMENT")
    print("=" * 90)

    for vector_weight, bm25_weight in WEIGHTS:

        hit1 = 0
        hit3 = 0
        ranks = []

        for test in TEST_CASES:

            rank = find_rank(
                test["question"],
                test["expected"],
                vector_weight,
                bm25_weight,
            )

            if rank == 1:
                hit1 += 1

            if rank is not None and rank <= 3:
                hit3 += 1

            if rank is None:
                ranks.append("MISS")
            else:
                ranks.append(f"#{rank}")

        print(
            f"Vector={vector_weight:.1f}  "
            f"BM25={bm25_weight:.1f}  |  "
            f"Hit@1: {hit1}/10  |  "
            f"Hit@3: {hit3}/10  |  "
            f"Ranks: {ranks}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()