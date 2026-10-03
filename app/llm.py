import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:latest"


def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()


if __name__ == "__main__":
    prompt = """
Use only the following context.

Context:
Full-time employees accrue 20 days of PTO per calendar year.

Question:
How much PTO do full-time employees receive per year?

Answer briefly.
"""

    answer = generate_answer(prompt)

    print("\nANSWER")
    print(answer)