from app.llm import generate_answer


ABSTAIN_MESSAGE = (
    "I couldn't find this information "
    "in the available company documents."
)


def generate_grounded_answer(
    question: str,
    evidence: list[dict],
) -> dict:
    """
    Generate an answer using only retrieved company evidence.

    Returns:
        {
            "answerable": True/False,
            "answer": "..."
        }
    """

    # ---------------------------------------------------------
    # No evidence = cannot answer
    # ---------------------------------------------------------
    if not evidence:
        return {
            "answerable": False,
            "answer": ABSTAIN_MESSAGE,
        }

    # ---------------------------------------------------------
    # Build context from ALL retrieved evidence
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
    # Grounded-answer prompt
    # ---------------------------------------------------------
    prompt = f"""
You are a company knowledge assistant.

Answer the user's question using ONLY the provided evidence.

IMPORTANT RULES:

1. Do NOT use outside knowledge.

2. Do NOT invent company policies.

3. Do NOT assume missing details.

4. Related information does NOT prove an unstated policy.

5. Pay close attention to qualifiers and specific policy types
in the question.

A broader related policy does NOT answer a narrower question.

Examples:

- A general PTO policy does NOT establish a parental leave policy.
- A remote-work policy does NOT establish whether employees
  may work remotely from another country.
- A sick-leave allowance does NOT establish whether additional
  sick leave is available after hospitalization.
- A PTO accrual policy does NOT establish whether unused PTO
  can be cashed out or carried over.

The evidence must support the specific policy, condition,
restriction, or qualifier asked about.

6. Every factual claim in your answer must be directly
supported by the provided evidence.

7. If the exact requested information is absent,
unspecified, or requires guessing, you MUST abstain.

8. Your response MUST begin with exactly ONE of these:

ANSWER:
ABSTAIN:

9. Use ANSWER: only when the evidence directly supports
the information requested.

10. Use ABSTAIN: when the requested information cannot
be directly determined from the evidence.

11. If abstaining, output exactly:

ABSTAIN: {ABSTAIN_MESSAGE}


EXAMPLE 1

Question:
How much vacation do employees get?

Evidence:
Full-time employees accrue 20 days of PTO per calendar year.

Output:
ANSWER: Full-time employees accrue 20 days of PTO per calendar year.


EXAMPLE 2

Question:
Can unused PTO be carried over to the next year?

Evidence:
Full-time employees accrue 20 days of PTO per calendar year.

Output:
ABSTAIN: {ABSTAIN_MESSAGE}


EXAMPLE 3

Question:
When are employees paid?

Evidence:
Employees are paid on a semi-monthly basis, on the 15th
and final day of each month.

Output:
ANSWER: Employees are paid on a semi-monthly basis,
on the 15th and final day of each month.


EXAMPLE 4

Question:
Does the December shutdown count against employee PTO?

Evidence:
The company observes a winter year-end shutdown during
the final week of December.

Output:
ABSTAIN: {ABSTAIN_MESSAGE}


QUESTION:
{question}

EVIDENCE:
{context}

OUTPUT:
"""

    # ---------------------------------------------------------
    # Generate response with local Llama 3
    # ---------------------------------------------------------
    raw_answer = generate_answer(
        prompt,
        model="llama3:latest",
    )

    raw_answer = raw_answer.strip()

    # ---------------------------------------------------------
    # Model explicitly abstained
    # ---------------------------------------------------------
    if raw_answer.upper().startswith("ABSTAIN:"):
        return {
            "answerable": False,
            "answer": ABSTAIN_MESSAGE,
        }

    # ---------------------------------------------------------
    # Model says the question is answerable
    # ---------------------------------------------------------
    if raw_answer.upper().startswith("ANSWER:"):

        answer = raw_answer[
            len("ANSWER:"):
        ].strip()

        # -----------------------------------------------------
        # Defensive abstention detection
        # -----------------------------------------------------
        lower_answer = answer.lower()

        abstention_phrases = [
            "couldn't find",
            "could not find",
            "does not mention",
            "doesn't mention",
            "not mentioned",
            "does not explicitly mention",
            "doesn't explicitly mention",
            "does not specifically mention",
            "doesn't specifically mention",
            "does not explicitly state",
            "doesn't explicitly state",
            "not explicitly stated",
            "does not explicitly discuss",
            "doesn't explicitly discuss",
            "does not indicate",
            "doesn't indicate",
            "not specified",
            "does not specify",
            "doesn't specify",
            "no information",
            "not provided",
            "cannot determine",
            "can't determine",
        ]

        if any(
            phrase in lower_answer
            for phrase in abstention_phrases
        ):
            return {
                "answerable": False,
                "answer": ABSTAIN_MESSAGE,
            }

        return {
            "answerable": True,
            "answer": answer,
        }

    # ---------------------------------------------------------
    # Fail closed
    # ---------------------------------------------------------
    # If the model ignores ANSWER:/ABSTAIN:, don't guess.
    # ---------------------------------------------------------
    return {
        "answerable": False,
        "answer": ABSTAIN_MESSAGE,
    }


# =============================================================
# HYBRID RETRIEVAL MANUAL TEST
# =============================================================

if __name__ == "__main__":

    from app.hybrid_retrieval import hybrid_search

    questions = [
        "How much vacation do employees get?",
        "When does Nexus pay its employees?",
        "Can unused PTO be carried over to the next year?",
        "Does the December shutdown count against employee PTO?",
    ]

    for question in questions:

        # -----------------------------------------------------
        # Retrieve TOP 3 using hybrid retrieval
        # -----------------------------------------------------
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
                    "similarity": (
                        result["vector_similarity"]
                    ),
                    "rrf_score": (
                        result["rrf_score"]
                    ),
                }
            )

        # -----------------------------------------------------
        # Generate grounded answer from all 3 chunks
        # -----------------------------------------------------
        answer_result = generate_grounded_answer(
            question,
            evidence,
        )

        # -----------------------------------------------------
        # Print result
        # -----------------------------------------------------
        print("\n" + "=" * 90)

        print(
            f"QUESTION: {question}"
        )

        print(
            f"ANSWERABLE: "
            f"{answer_result['answerable']}"
        )

        print(
            f"ANSWER: "
            f"{answer_result['answer']}"
        )

        print("\nRETRIEVED EVIDENCE:")

        for rank, item in enumerate(
            evidence,
            start=1,
        ):
            print(
                f"  #{rank} "
                f"{item['section']} "
                f"(RRF: {item['rrf_score']:.6f})"
            )