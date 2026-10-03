from app.embeddings import create_embedding
from app.retrieval import search_documents
from eval.evidence_tests import TEST_CASES


def cosine_similarity(vector_a, vector_b):
    return sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )


def get_semantic_score(question, evidence_text):
    question_embedding = create_embedding(question)
    evidence_embedding = create_embedding(evidence_text)

    return cosine_similarity(
        question_embedding,
        evidence_embedding,
    )


# ------------------------------------------------------------
# Calculate scores once
# ------------------------------------------------------------

scored_tests = []

for test in TEST_CASES:
    results = search_documents(
        test["question"],
        limit=3,
        use_expansion=True,
    )

    top_chunk = results[0][0]

    score = get_semantic_score(
        test["question"],
        top_chunk.content,
    )

    scored_tests.append(
        {
            "question": test["question"],
            "expected": test["expected"],
            "score": score,
        }
    )


# ------------------------------------------------------------
# Try different thresholds
# ------------------------------------------------------------

thresholds = [
    0.25,
    0.28,
    0.30,
    0.32,
    0.33,
    0.34,
    0.345,
    0.346,
    0.35,
    0.36,
    0.38,
    0.40,
]


print("\nSEMANTIC THRESHOLD EXPERIMENT")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<15}"
    f"{'False Positives':<20}"
    f"{'False Negatives':<20}"
)

print("-" * 75)


for threshold in thresholds:
    correct = 0
    false_positives = 0
    false_negatives = 0

    for test in scored_tests:
        predicted = test["score"] >= threshold
        expected = test["expected"]

        if predicted == expected:
            correct += 1

        if predicted and not expected:
            false_positives += 1

        if not predicted and expected:
            false_negatives += 1

    accuracy = correct / len(scored_tests)

    print(
        f"{threshold:<12.3f}"
        f"{accuracy:<15.1%}"
        f"{false_positives:<20}"
        f"{false_negatives:<20}"
    )