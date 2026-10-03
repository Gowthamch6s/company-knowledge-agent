import os

import requests

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

DEFAULT_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest",
)


# ============================================================
# LLM GENERATION
# ============================================================

def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a response using Ollama.

    OLLAMA_URL can point to:
    - localhost during normal development
    - host.docker.internal when running inside Docker
    """

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


# ============================================================
# MANUAL TEST
# ============================================================

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

    print("\nOLLAMA URL")
    print(OLLAMA_URL)

    print("\nMODEL")
    print(DEFAULT_MODEL)

    print("\nANSWER")
    print(answer)