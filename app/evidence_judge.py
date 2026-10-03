from app.llm import generate_answer


def judge_evidence(
    question: str,
    evidence: list[dict],
) -> bool:
    """
    Decide whether the retrieved evidence directly supports
    answering the user's question.
    """

    if not evidence:
        return False

    # ---------------------------------------------------------
    # Build context from retrieved evidence
    # ---------------------------------------------------------
    context_parts = []

    for item in evidence:
        context_parts.append(
            f"""
Section: {item['section']}
Page: {item['page']}

{item['content']}
"""
        )

    context = "\n".join(context_parts)

    # ---------------------------------------------------------
    # Evidence-judging prompt
    # ---------------------------------------------------------
    prompt = f"""
You are verifying whether a company document contains the
specific information requested by a user.

You are NOT answering the question.
You are NOT judging whether the evidence is merely related
to the topic.

Determine whether the evidence explicitly states, or directly
supports, the information required to answer the exact question.

STRICT RULES:

1. Use ONLY the provided evidence.
2. Topic similarity alone is not enough.
3. Do NOT assume missing company policies.
4. Do NOT fill gaps using common business practices.
5. Do NOT infer one policy from another related policy.
6. Normal paraphrases and synonyms are allowed.
   For example:
   - vacation can mean paid time off / PTO
   - work from home can mean remote work
   - pay frequency can refer to a pay schedule
7. If the evidence discusses the same topic but does not state
   the requested detail, return NOT_SUPPORTED.
8. If answering would require guessing a missing policy detail,
   return NOT_SUPPORTED.
9. Return SUPPORTED only when the requested information can be
   directly obtained from the evidence.
10. Return exactly one label:

SUPPORTED
NOT_SUPPORTED


EXAMPLE 1

Question:
How many PTO days do full-time employees receive?

Evidence:
Full-time employees accrue 20 days of PTO per calendar year.

Label:
SUPPORTED


EXAMPLE 2

Question:
How much vacation do full-time employees receive?

Evidence:
Full-time employees accrue 20 days of PTO per calendar year.

Label:
SUPPORTED


EXAMPLE 3

Question:
Can unused PTO be carried over to the next year?

Evidence:
Full-time employees accrue 20 days of PTO per calendar year.

Label:
NOT_SUPPORTED


EXAMPLE 4

Question:
Can employees cash out unused PTO?

Evidence:
Employees accrue 20 days of PTO annually and PTO requests
must be submitted through the HR portal.

Label:
NOT_SUPPORTED


EXAMPLE 5

Question:
When are employees paid?

Evidence:
Employees are paid on a semi-monthly basis, on the 15th
and final day of each month.

Label:
SUPPORTED


QUESTION:
{question}

EVIDENCE:
{context}

LABEL:
"""

    # ---------------------------------------------------------
    # Use the larger Llama 3 model as the evidence judge
    # ---------------------------------------------------------
    result = generate_answer(
        prompt,
        model="llama3:latest",
    )

    # Temporary debug output so we can see exactly
    # what the model returned.
    print(
        f"\n[DEBUG JUDGE RAW OUTPUT]: "
        f"{repr(result)}"
    )

    classification = result.strip().upper()

    return classification == "SUPPORTED"


# =============================================================
# TEST THE EVIDENCE JUDGE
# =============================================================

if __name__ == "__main__":
    from app.retrieval import search_documents

    questions = [
        # Should be TRUE
        "How much vacation do employees get?",

        # Should be FALSE
        "Can unused PTO be carried over to the next year?",

        # Should be TRUE
        "How often are employees paid?",

        # Should be FALSE
        "Does the December shutdown count against employee PTO?",
    ]

    for question in questions:

        # Retrieve the strongest evidence only.
        results = search_documents(
            question,
            limit=1,
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

        # Ask the evidence judge whether the retrieved
        # evidence actually supports the question.
        sufficient = judge_evidence(
            question,
            evidence,
        )

        print("\n" + "=" * 80)
        print(f"Question: {question}")
        print(f"Evidence sufficient: {sufficient}")