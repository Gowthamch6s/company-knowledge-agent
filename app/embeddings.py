from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_DIMENSION


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def create_embedding(text: str) -> list[float]:
    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )

    embedding = embedding.tolist()

    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, "
            f"but received {len(embedding)}."
        )

    return embedding


if __name__ == "__main__":
    test_text = (
        "Full-time employees receive 18 days "
        "of paid time off per calendar year."
    )

    embedding = create_embedding(test_text)

    print(f"Text: {test_text}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")