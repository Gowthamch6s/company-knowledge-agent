from app.hybrid_retrieval import hybrid_search
from app.lexical_retrieval import lexical_search
from app.retrieval import search_documents


TEST_CASES = [
    {
        "name": "Vacation",
        "question": "How much vacation do employees get?",
        "expected": "4.1",
    },
    {
        "name": "Remote work",
        "question": "Can employees work remotely?",
        "expected": "2.3",
    },
    {
        "name": "Pay frequency",
        "question": "How often are employees paid?",
        "expected": "3.1",
    },
    {
        "name": "Pay date paraphrase",
        "question": "When does Nexus pay its employees?",
        "expected": "3.1",
    },
    {
        "name": "Laptop security",
        "question": (
            "What security measures are required "
            "for company laptops?"
        ),
        "expected": "5.1",
    },
    {
        "name": "December shutdown",
        "question": (
            "What happens during the final week "
            "of December?"
        ),
        "expected": "4.3",
    },
    {
        "name": "Performance reviews",
        "question": (
            "When do formal performance reviews happen?"
        ),
        "expected": "3.2",
    },
    {
        "name": "Working hours",
        "question": "What are the standard working hours?",
        "expected": "2.3",
    },
    {
        "name": "Harassment",
        "question": (
            "How should employees report harassment?"
        ),
        "expected": "2.2",
    },
    {
        "name": "Code ownership",
        "question": (
            "Who owns software created during employment?"
        ),
        "expected": "5.2",
    },
]


def section_matches(
    section_title,
    expected,
):
    if not section_title:
        return False

    return section_title.startswith(expected)


def vector_rank(question, expected):
    results = search_documents(
        question,
        limit=5,
        use_expansion=True,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        chunk = result[0]

        if section_matches(
            chunk.section_title,
            expected,
        ):
            return rank

    return None


def bm25_rank(question, expected):
    results = lexical_search(
        question,
        limit=5,
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


def hybrid_rank(question, expected):
    results = hybrid_search(
        question,
        limit=5,
        candidate_limit=10,
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


def display_rank(rank):
    if rank is None:
        return "MISS"

    return f"#{rank}"


def main():
    vector_hit1 = 0
    bm25_hit1 = 0
    hybrid_hit1 = 0

    vector_hit3 = 0
    bm25_hit3 = 0
    hybrid_hit3 = 0

    print("\nHYBRID RETRIEVAL EVALUATION")
    print("=" * 100)

    for test in TEST_CASES:

        question = test["question"]
        expected = test["expected"]

        v_rank = vector_rank(
            question,
            expected,
        )

        b_rank = bm25_rank(
            question,
            expected,
        )

        h_rank = hybrid_rank(
            question,
            expected,
        )

        if v_rank == 1:
            vector_hit1 += 1

        if b_rank == 1:
            bm25_hit1 += 1

        if h_rank == 1:
            hybrid_hit1 += 1

        if v_rank is not None and v_rank <= 3:
            vector_hit3 += 1

        if b_rank is not None and b_rank <= 3:
            bm25_hit3 += 1

        if h_rank is not None and h_rank <= 3:
            hybrid_hit3 += 1

        print(
            f"{test['name']:<22} "
            f"Expected: {expected:<4} | "
            f"Vector: {display_rank(v_rank):<5} | "
            f"BM25: {display_rank(b_rank):<5} | "
            f"Hybrid: {display_rank(h_rank):<5}"
        )

    total = len(TEST_CASES)

    print("\n" + "=" * 100)

    print(
        f"{'Strategy':<20}"
        f"{'Hit@1':<15}"
        f"{'Hit@3':<15}"
    )

    print("-" * 50)

    print(
        f"{'Vector + Expansion':<20}"
        f"{vector_hit1}/{total:<13}"
        f"{vector_hit3}/{total}"
    )

    print(
        f"{'BM25':<20}"
        f"{bm25_hit1}/{total:<13}"
        f"{bm25_hit3}/{total}"
    )

    print(
        f"{'Equal RRF Hybrid':<20}"
        f"{hybrid_hit1}/{total:<13}"
        f"{hybrid_hit3}/{total}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()