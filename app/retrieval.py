from sqlalchemy import select

from app.query_expansion import expand_query
from app.database import SessionLocal
from app.embeddings import create_embedding
from app.models import Document, DocumentChunk
from app.reranker import rerank_results


def search_documents(
    query: str,
    limit: int = 5,
    use_expansion: bool = True,
):
    search_query = (
        expand_query(query)
        if use_expansion
        else query
    )

    query_embedding = create_embedding(search_query)

    session = SessionLocal()

    try:
        distance = DocumentChunk.embedding.cosine_distance(
            query_embedding
        )

        statement = (
            select(
                DocumentChunk,
                Document.filename,
                distance.label("distance"),
            )
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )

        results = session.execute(statement).all()

        return results

    finally:
        session.close()


def retrieve_and_rerank(
    query: str,
    candidate_limit: int = 5,
    final_limit: int = 3,
):
    # Stage 1: fast vector search
    candidates = search_documents(
        query,
        limit=candidate_limit,
    )

    # Stage 2: precise cross-encoder reranking
    reranked = rerank_results(
        query,
        candidates,
    )

    return reranked[:final_limit]


if __name__ == "__main__":
    test_queries = [
        "How much vacation do employees get?",
        "Can employees work from home?",
        "How often do employees get paid?",
        "What security measures are required for company laptops?",
        "What happens during the final week of December?",
    ]

    for query in test_queries:
        print("\n" + "#" * 80)
        print(f"QUERY: {query}")
        print("#" * 80)

        results = retrieve_and_rerank(
            query,
            candidate_limit=5,
            final_limit=3,
        )

        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]

            vector_similarity = (
                1 - result["vector_distance"]
            )

            print()
            print(f"RESULT #{rank}")
            print(f"Document: {result['filename']}")
            print(f"Page: {chunk.page_number}")
            print(f"Chunk: {chunk.chunk_index}")

            print(
                f"Vector similarity: "
                f"{vector_similarity:.4f}"
            )

            print(
                f"Reranker score: "
                f"{result['reranker_score']:.4f}"
            )

            print("-" * 60)
            print(chunk.content[:500])