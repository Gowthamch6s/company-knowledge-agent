from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.hybrid_retrieval import hybrid_search
from app.grounded_answer import generate_grounded_answer


# ============================================================
# 1. STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    evidence: list[dict]
    answerable: bool
    answer: str


# ============================================================
# 2. HYBRID RETRIEVAL NODE
# ============================================================

def retrieve_node(state: AgentState):
    """
    Retrieve the most relevant evidence using the project's
    hybrid retrieval pipeline:

        Dense retrieval
            +
        BM25 lexical retrieval
            ↓
        Reciprocal Rank Fusion
            ↓
        Top 3 chunks
    """

    question = state["question"]

    results = hybrid_search(
        question,
        limit=3,
        candidate_limit=10,
    )

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

    return {
        "evidence": evidence
    }


# ============================================================
# 3. GROUNDED ANSWER NODE
# ============================================================

def grounded_answer_node(state: AgentState):
    """
    Generate an answer using only the retrieved evidence.

    generate_grounded_answer() is responsible for deciding
    whether the evidence supports the requested information.

    If it does not, the system abstains instead of inventing
    a company policy.
    """

    question = state["question"]
    evidence = state["evidence"]

    result = generate_grounded_answer(
        question,
        evidence,
    )

    return {
        "answerable": result["answerable"],
        "answer": result["answer"],
    }


# ============================================================
# 4. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(AgentState)

builder.add_node(
    "retrieve",
    retrieve_node,
)

builder.add_node(
    "grounded_answer",
    grounded_answer_node,
)


# START -> HYBRID RETRIEVAL
builder.add_edge(
    START,
    "retrieve",
)


# RETRIEVAL -> GROUNDED ANSWER
builder.add_edge(
    "retrieve",
    "grounded_answer",
)


# GROUNDED ANSWER -> END
builder.add_edge(
    "grounded_answer",
    END,
)


graph = builder.compile()


# ============================================================
# 5. MANUAL TEST
# ============================================================

if __name__ == "__main__":

    questions = [
        "How much vacation do employees get?",
        "When does Nexus pay its employees?",
        "Can unused PTO be carried over to the next year?",
    ]

    for question in questions:

        result = graph.invoke(
            {
                "question": question,
                "evidence": [],
                "answerable": False,
                "answer": "",
            }
        )

        print("\n" + "=" * 90)

        print(f"QUESTION: {result['question']}")

        print(
            f"ANSWERABLE: "
            f"{result['answerable']}"
        )

        print(
            f"ANSWER: "
            f"{result['answer']}"
        )

        print("\nSOURCES:")

        for rank, item in enumerate(
            result["evidence"],
            start=1,
        ):
            print(
                f"  #{rank} "
                f"{item['section']} "
                f"(Page {item['page']})"
            )