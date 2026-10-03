from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph import graph


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Company Knowledge Agent",
    description=(
        "A grounded company knowledge assistant using "
        "hybrid retrieval, pgvector, BM25, RRF, "
        "LangGraph, and a local LLM."
    ),
    version="1.0.0",
)


# ============================================================
# 2. REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Question about the company documents.",
    )


class Source(BaseModel):
    filename: str | None = None
    section: str | None = None
    page: int | None = None


class AskResponse(BaseModel):
    question: str
    answerable: bool
    answer: str
    sources: list[Source]


# ============================================================
# 3. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():
    """
    Lightweight API health check.

    This confirms that the FastAPI application is running.
    """

    return {
        "status": "ok",
        "service": "company-knowledge-agent",
    }


# ============================================================
# 4. ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):
    """
    Run a question through the complete LangGraph RAG pipeline.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = graph.invoke(
            {
                "question": question,
                "evidence": [],
                "answerable": False,
                "answer": "",
            }
        )

    except Exception as exc:
        # Do not expose internal stack traces or database details
        # through the public API.
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question.",
        ) from exc

    # --------------------------------------------------------
    # Build source metadata
    # --------------------------------------------------------

    sources = []

    seen = set()

    for item in result.get("evidence", []):

        source_key = (
            item.get("filename"),
            item.get("section"),
            item.get("page"),
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        sources.append(
            Source(
                filename=item.get("filename"),
                section=item.get("section"),
                page=item.get("page"),
            )
        )

    return AskResponse(
        question=question,
        answerable=result["answerable"],
        answer=result["answer"],
        sources=sources,
    )