QUERY_SYNONYMS = {
    "vacation": [
        "paid time off",
        "PTO",
    ],
    "work from home": [
        "remote work",
        "hybrid work",
        "remote-friendly",
    ],
    "pay": [
        "pay schedule",
        "payroll",
        "compensation",
    ],
    "laptop": [
        "company-issued laptop",
        "hardware",
        "device security",
    ],
}


def expand_query(query: str) -> str:
    expanded_terms = []

    query_lower = query.lower()

    for term, synonyms in QUERY_SYNONYMS.items():

        if term in query_lower:
            expanded_terms.extend(synonyms)

    if not expanded_terms:
        return query

    expansion = ", ".join(expanded_terms)

    return f"{query} Related terms: {expansion}"