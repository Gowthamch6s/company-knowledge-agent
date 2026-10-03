from app.embeddings import create_embedding
from app.retrieval import search_documents
from eval.evidence_tests import TEST_CASES


def cosine_similarity(vector_a, vector_b):
    return sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )


def semantic_score(question, evidence_text):
    question_embedding = create_embedding(question)
    evidence_embedding = create_embedding(evidence_text)

    return cosine_similarity(
        question_embedding,
        evidence_embedding,
    )


if __name__ == "__main__":
    print("\nSEMANTIC EVIDENCE EXPERIMENT")
    print("=" * 110)

    answerable_scores = []
    unanswerable_scores = []

    for number, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected = test["expected"]

        results = search_documents(
            question,
            limit=3,
            use_expansion=True,
        )

        top_chunk = results[0][0]

        score = semantic_score(
            question,
            top_chunk.content,
        )

        if expected:
            answerable_scores.append(score)
        else:
            unanswerable_scores.append(score)

        label = "ANSWERABLE" if expected else "UNANSWERABLE"

        print(
            f"{number:>2}. "
            f"{label:<12} | "
            f"Score: {score:.4f} | "
            f"Section: {str(top_chunk.section_title):<45} | "
            f"{question}"
        )

    print("\n" + "=" * 110)

    print("ANSWERABLE")
    print(
        f"Minimum: {min(answerable_scores):.4f} | "
        f"Maximum: {max(answerable_scores):.4f} | "
        f"Average: "
        f"{sum(answerable_scores) / len(answerable_scores):.4f}"
    )

    print("\nUNANSWERABLE")
    print(
        f"Minimum: {min(unanswerable_scores):.4f} | "
        f"Maximum: {max(unanswerable_scores):.4f} | "
        f"Average: "
        f"{sum(unanswerable_scores) / len(unanswerable_scores):.4f}"
    )

    print("=" * 110)