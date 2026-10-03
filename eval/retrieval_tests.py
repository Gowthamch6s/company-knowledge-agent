from app.retrieval import retrieve_and_rerank


TEST_CASES = [
    {
        "name": "Vacation",
        "query": "How much vacation do employees get?",
        "expected_section": "4.1",
    },
    {
        "name": "Remote work",
        "query": "Can employees work from home?",
        "expected_section": "2.3",
    },
    {
        "name": "Pay frequency",
        "query": "How often do employees get paid?",
        "expected_section": "3.1",
    },
    {
        "name": "Laptop security",
        "query": "What security measures are required for company laptops?",
        "expected_section": "5.1",
    },
    {
        "name": "December shutdown",
        "query": "What happens during the final week of December?",
        "expected_section": "4.3",
    },
]


def find_expected_rank(results, expected_section):
    for rank, result in enumerate(results, start=1):
        section = result["chunk"].section_title or ""

        if section.startswith(expected_section):
            return rank

    return None


def run_evaluation():
    hit_at_1 = 0
    hit_at_3 = 0

    print("\nCOMPANY KNOWLEDGE AGENT — RETRIEVAL EVALUATION")
    print("=" * 70)

    for test in TEST_CASES:
        results = retrieve_and_rerank(
            test["query"],
            candidate_limit=5,
            final_limit=3,
        )

        rank = find_expected_rank(
            results,
            test["expected_section"],
        )

        if rank == 1:
            hit_at_1 += 1

        if rank is not None and rank <= 3:
            hit_at_3 += 1

        rank_display = (
            f"#{rank}"
            if rank is not None
            else "MISS"
        )

        print(
            f"{test['name']:<20} "
            f"Expected: {test['expected_section']:<5} "
            f"Rank: {rank_display}"
        )

    total = len(TEST_CASES)

    print("\n" + "=" * 70)
    print(f"Hit@1: {hit_at_1}/{total} = {hit_at_1 / total:.0%}")
    print(f"Hit@3: {hit_at_3}/{total} = {hit_at_3 / total:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()