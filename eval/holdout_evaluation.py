from eval.holdout_tests import HOLDOUT_TEST_CASES

from app.grounded_answer import generate_grounded_answer
from app.hybrid_retrieval import hybrid_search


def build_evidence(results):

    evidence = []

    for result in results:

        chunk = result["chunk"]

        evidence.append(
            {
                "filename": result["filename"],
                "page": chunk.page_number,
                "section": chunk.section_title,
                "content": chunk.content,
                "similarity": result["vector_similarity"],
                "rrf_score": result["rrf_score"],
                "vector_rank": result["vector_rank"],
                "bm25_rank": result["bm25_rank"],
            }
        )

    return evidence


def main():

    passed = 0
    false_positives = 0
    false_negatives = 0

    print("\nHOLDOUT EVALUATION")
    print("=" * 110)

    for index, test in enumerate(
        HOLDOUT_TEST_CASES,
        start=1,
    ):

        question = test["question"]
        expected = test["expected"]

        # -----------------------------------------------------
        # Hybrid retrieval
        # -----------------------------------------------------

        results = hybrid_search(
            question,
            limit=3,
            candidate_limit=10,
        )

        evidence = build_evidence(results)

        # -----------------------------------------------------
        # Grounded generation
        # -----------------------------------------------------

        result = generate_grounded_answer(
            question,
            evidence,
        )

        predicted = result["answerable"]
        answer = result["answer"]

        correct = predicted == expected

        if correct:
            passed += 1

        elif predicted and not expected:
            false_positives += 1

        elif not predicted and expected:
            false_negatives += 1

        status = "PASS" if correct else "FAIL"

        print(
            f"{index:>2}. "
            f"{status:<4} | "
            f"Expected: {str(expected):<5} | "
            f"Predicted: {str(predicted):<5} | "
            f"{question}"
        )

        # -----------------------------------------------------
        # Debug failures only
        # -----------------------------------------------------

        if not correct:

            print(
                f"    ANSWER: {answer}"
            )

            print(
                "    RETRIEVED:"
            )

            for rank, item in enumerate(
                evidence,
                start=1,
            ):

                print(
                    f"      #{rank} "
                    f"{item['section']} | "
                    f"Vector rank: "
                    f"{item['vector_rank']} | "
                    f"BM25 rank: "
                    f"{item['bm25_rank']} | "
                    f"RRF: "
                    f"{item['rrf_score']:.6f}"
                )

            print()

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    total = len(HOLDOUT_TEST_CASES)

    accuracy = (
        passed / total * 100
        if total
        else 0
    )

    print("\n" + "=" * 110)

    print(
        f"Accuracy: "
        f"{passed}/{total} "
        f"= {accuracy:.1f}%"
    )

    print(
        f"False positives: "
        f"{false_positives}"
    )

    print(
        f"False negatives: "
        f"{false_negatives}"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()