from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pathlib import Path

from fastapi.responses import FileResponse

from app.graph import graph
from app.source_attribution import select_supporting_sources


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

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        STATIC_DIR / "index.html"
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

    # --------------------------------------------------------
    # Clean the incoming question
    # --------------------------------------------------------

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # --------------------------------------------------------
    # Run the LangGraph pipeline
    # --------------------------------------------------------

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
        # Do not expose internal database/model errors
        # through the public API.
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question.",
        ) from exc

    # --------------------------------------------------------
    # Select supporting sources
    # --------------------------------------------------------

    sources = []

    # Only attach a source when the system actually
    # produced a supported answer.
    #
    # If the system abstains, sources remain empty.
    if result["answerable"]:

        supporting_evidence = select_supporting_sources(
            answer=result["answer"],
            evidence=result.get("evidence", []),
            limit=1,
        )

        for item in supporting_evidence:
            sources.append(
                Source(
                    filename=item.get("filename"),
                    section=item.get("section"),
                    page=item.get("page"),
                )
            )

    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return AskResponse(
        question=question,
        answerable=result["answerable"],
        answer=result["answer"],
        sources=sources,
    )