from app.retrieval import search_documents
from app.reranker import rerank_results
from eval.retrieval_tests import TEST_CASES


def find_rank(results, expected_section, result_type="vector"):
    for rank, result in enumerate(results, start=1):

        if result_type == "vector":
            chunk = result[0]
        else:
            chunk = result["chunk"]

        section = chunk.section_title or ""

        if section.startswith(expected_section):
            return rank

    return None


def evaluate_vector(use_expansion):
    hit1 = 0
    hit3 = 0
    ranks = []

    for test in TEST_CASES:

        results = search_documents(
            test["query"],
            limit=3,
            use_expansion=use_expansion,
        )

        rank = find_rank(
            results,
            test["expected_section"],
            result_type="vector",
        )

        ranks.append(rank)

        if rank == 1:
            hit1 += 1

        if rank is not None and rank <= 3:
            hit3 += 1

    return hit1, hit3, ranks


def evaluate_fusion():
    hit1 = 0
    hit3 = 0
    ranks = []

    for test in TEST_CASES:

        candidates = search_documents(
            test["query"],
            limit=10,
            use_expansion=True,
        )

        results = rerank_results(
            test["query"],
            candidates,
            vector_weight=0.7,
            reranker_weight=0.3,
        )[:3]

        rank = find_rank(
            results,
            test["expected_section"],
            result_type="reranked",
        )

        ranks.append(rank)

        if rank == 1:
            hit1 += 1

        if rank is not None and rank <= 3:
            hit3 += 1

    return hit1, hit3, ranks


def format_ranks(ranks):
    return [
        f"#{rank}" if rank is not None else "MISS"
        for rank in ranks
    ]


if __name__ == "__main__":

    print("\nRETRIEVAL STRATEGY COMPARISON")
    print("=" * 90)

    vector = evaluate_vector(
        use_expansion=False
    )

    expanded = evaluate_vector(
        use_expansion=True
    )

    fusion = evaluate_fusion()

    print(
        f"{'Strategy':<35}"
        f"{'Hit@1':<12}"
        f"{'Hit@3':<12}"
        f"Ranks"
    )

    print("-" * 90)

    print(
        f"{'Vector only':<35}"
        f"{vector[0]}/5{'':<9}"
        f"{vector[1]}/5{'':<9}"
        f"{format_ranks(vector[2])}"
    )

    print(
        f"{'Vector + Query Expansion':<35}"
        f"{expanded[0]}/5{'':<9}"
        f"{expanded[1]}/5{'':<9}"
        f"{format_ranks(expanded[2])}"
    )

    print(
        f"{'Expansion + 70/30 Fusion':<35}"
        f"{fusion[0]}/5{'':<9}"
        f"{fusion[1]}/5{'':<9}"
        f"{format_ranks(fusion[2])}"
    )

    print("=" * 90)