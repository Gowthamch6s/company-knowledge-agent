from eval.evidence_tests import TEST_CASES

from app.grounded_answer import generate_grounded_answer
from app.hybrid_retrieval import hybrid_search


def build_evidence(results):
    """
    Convert hybrid retrieval results into the evidence format
    expected by generate_grounded_answer().
    """

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
    """
    Evaluate the complete hybrid grounded-answer pipeline.

    Pipeline:

        Question
            ↓
        Vector Retrieval + Query Expansion
            +
        BM25 Lexical Retrieval
            ↓
        Reciprocal Rank Fusion
            ↓
        Top 3 Evidence Chunks
            ↓
        Grounded Llama 3
            ↓
        Answer / Abstain
    """

    passed = 0
    false_positives = 0
    false_negatives = 0

    print("\nHYBRID GROUNDED ANSWER EVALUATION")
    print("=" * 110)

    for index, test in enumerate(
        TEST_CASES,
        start=1,
    ):
        question = test["question"]
        expected_answerable = test["expected"]

        # -----------------------------------------------------
        # STEP 1 — Hybrid retrieval
        # -----------------------------------------------------

        results = hybrid_search(
            question,
            limit=3,
            candidate_limit=10,
        )

        evidence = build_evidence(results)

        # -----------------------------------------------------
        # STEP 2 — Generate grounded answer
        # -----------------------------------------------------

        result = generate_grounded_answer(
            question,
            evidence,
        )

        predicted_answerable = result["answerable"]
        answer = result["answer"]

        # -----------------------------------------------------
        # STEP 3 — Compare with expected answerability
        # -----------------------------------------------------

        correct = (
            predicted_answerable
            == expected_answerable
        )

        if correct:
            passed += 1

        elif (
            predicted_answerable
            and not expected_answerable
        ):
            # System answered something that should
            # have been rejected.
            false_positives += 1

        elif (
            not predicted_answerable
            and expected_answerable
        ):
            # System abstained even though the answer
            # exists in the company documents.
            false_negatives += 1

        # -----------------------------------------------------
        # STEP 4 — Print result
        # -----------------------------------------------------

        status = "PASS" if correct else "FAIL"

        print(
            f"{index:>2}. "
            f"{status:<4} | "
            f"Expected: "
            f"{str(expected_answerable):<5} | "
            f"Predicted: "
            f"{str(predicted_answerable):<5} | "
            f"{question}"
        )

        # -----------------------------------------------------
        # Show detailed debugging only for failures
        # -----------------------------------------------------

        if not correct:
            print(
                f"    ANSWER: {answer}"
            )

            if evidence:
                print(
                    "    RETRIEVED EVIDENCE:"
                )

                for rank, item in enumerate(
                    evidence,
                    start=1,
                ):
                    similarity = item["similarity"]

                    if similarity is None:
                        similarity_text = "N/A"
                    else:
                        similarity_text = (
                            f"{similarity:.4f}"
                        )

                    print(
                        f"      #{rank} "
                        f"{item['section']}"
                    )

                    print(
                        f"         "
                        f"Vector rank: "
                        f"{item['vector_rank']} | "
                        f"BM25 rank: "
                        f"{item['bm25_rank']}"
                    )

                    print(
                        f"         "
                        f"Vector similarity: "
                        f"{similarity_text} | "
                        f"RRF: "
                        f"{item['rrf_score']:.6f}"
                    )

            else:
                print(
                    "    RETRIEVED EVIDENCE: None"
                )

            print()

    # ---------------------------------------------------------
    # FINAL METRICS
    # ---------------------------------------------------------

    total = len(TEST_CASES)

    accuracy = (
        passed / total * 100
        if total > 0
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