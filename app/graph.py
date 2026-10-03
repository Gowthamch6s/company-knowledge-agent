from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.retrieval import search_documents
from app.query_expansion import expand_query
from app.llm import generate_answer


# ============================================================
# 1. STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    evidence: list[dict]
    evidence_sufficient: bool
    answer: str


# ============================================================
# 2. RETRIEVAL NODE
# ============================================================

def retrieve_node(state: AgentState):
    question = state["question"]

    results = search_documents(
        question,
        limit=3,
        use_expansion=True,
    )

    evidence = []

    for result in results:
        chunk = result[0]
        filename = result[1]
        distance = result[2]

        evidence.append(
            {
                "filename": filename,
                "page": chunk.page_number,
                "section": chunk.section_title,
                "content": chunk.content,
                "similarity": float(1 - distance),
            }
        )

    return {
        "evidence": evidence
    }


# ============================================================
# 3. EVIDENCE CHECKING NODE
# ============================================================

def check_evidence_node(state: AgentState):
    question = state["question"]
    evidence = state["evidence"]

    if not evidence:
        return {
            "evidence_sufficient": False
        }

    expanded_question = expand_query(question).lower()

    top_evidence = evidence[0]["content"].lower()

    important_terms = [
        word
        for word in expanded_question.split()
        if len(word) > 3
    ]

    matches = [
        word
        for word in important_terms
        if word in top_evidence
    ]

    sufficient = len(matches) >= 2

    return {
        "evidence_sufficient": sufficient
    }


# ============================================================
# 4. ANSWER NODE
# ============================================================

def answer_node(state: AgentState):
    question = state["question"]
    evidence = state["evidence"]

    context_parts = []

    for item in evidence:
        context_parts.append(
            f"""
Document: {item['filename']}
Section: {item['section']}
Page: {item['page']}

{item['content']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a company knowledge assistant.

Answer the employee's question using ONLY the information
contained in the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent company policies.
3. If the context does not support a claim, do not make that claim.
4. Answer clearly and concisely.
5. Do not create fake citations.
6. Use the exact numbers, dates, and policy details from the context.

CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:
"""

    answer = generate_answer(prompt)

    return {
        "answer": answer
    }


# ============================================================
# 5. NO-ANSWER NODE
# ============================================================

def no_answer_node(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough evidence "
            "in the available company documents."
        )
    }


# ============================================================
# 6. ROUTER
# ============================================================

def route_evidence(state: AgentState):
    if state["evidence_sufficient"]:
        return "answer"

    return "no_answer"


# ============================================================
# 7. BUILD THE LANGGRAPH
# ============================================================

builder = StateGraph(AgentState)

builder.add_node(
    "retrieve",
    retrieve_node,
)

builder.add_node(
    "check_evidence",
    check_evidence_node,
)

builder.add_node(
    "answer",
    answer_node,
)

builder.add_node(
    "no_answer",
    no_answer_node,
)


# START -> RETRIEVE
builder.add_edge(
    START,
    "retrieve",
)


# RETRIEVE -> CHECK EVIDENCE
builder.add_edge(
    "retrieve",
    "check_evidence",
)


# CHECK EVIDENCE -> ANSWER OR NO ANSWER
builder.add_conditional_edges(
    "check_evidence",
    route_evidence,
    {
        "answer": "answer",
        "no_answer": "no_answer",
    },
)


# ANSWER -> END
builder.add_edge(
    "answer",
    END,
)


# NO ANSWER -> END
builder.add_edge(
    "no_answer",
    END,
)


graph = builder.compile()


# ============================================================
# 8. TEST THE GRAPH
# ============================================================

if __name__ == "__main__":
    question = "How much vacation do employees get?"

    result = graph.invoke(
        {
            "question": question,
            "evidence": [],
            "evidence_sufficient": False,
            "answer": "",
        }
    )

    print("\nQUESTION")
    print(result["question"])

    print("\nTOP EVIDENCE")

    if result["evidence"]:
        top = result["evidence"][0]

        print(f"Section: {top['section']}")
        print(f"Page: {top['page']}")
        print(
            f"Similarity: "
            f"{top['similarity']:.4f}"
        )

    print("\nEVIDENCE SUFFICIENT")
    print(result["evidence_sufficient"])

    print("\nANSWER")
    print(result["answer"])