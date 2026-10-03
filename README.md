# Company Knowledge Agent

A grounded Retrieval-Augmented Generation (RAG) system for answering questions over internal company documents.

The system combines dense semantic retrieval, BM25 lexical search, Reciprocal Rank Fusion (RRF), LangGraph orchestration, grounded local LLM generation, semantic source attribution, FastAPI, PostgreSQL + pgvector, and Docker.

Unlike a basic document chatbot, the system is designed to **abstain when the available documents do not support an answer** rather than inventing missing company policies.

---

## Demo

The interface answers employee questions using company documents and displays the source supporting each grounded response.

![Company Knowledge Agent Demo](assets/ui-demo.png)

---

## Key Features

- Hybrid semantic + lexical retrieval
- SentenceTransformer embeddings
- PostgreSQL + pgvector vector storage
- BM25 keyword retrieval
- Reciprocal Rank Fusion (RRF)
- Query expansion for semantic retrieval
- LangGraph-based RAG orchestration
- Local LLM inference through Ollama
- Grounded answer generation
- Explicit answer/abstain behavior
- Defensive fail-closed output handling
- Semantic source attribution
- FastAPI REST API
- Lightweight browser UI
- Dockerized application and database
- Retrieval and answer-quality evaluation suites
- Separate holdout evaluation

---

## Why This Project?

Enterprise knowledge assistants have a different failure mode from general-purpose chatbots.

If an employee asks:

> Can unused PTO be carried over to next year?

and the handbook only states:

> Full-time employees accrue 20 days of PTO per calendar year.

the system should **not infer a carryover policy**.

This project therefore separates two problems:

1. **Retrieve the most relevant evidence.**
2. **Determine whether that evidence actually supports the requested information.**

If the documents do not support the answer, the agent returns:

```text
I couldn't find this information in the available company documents.
```

This makes abstention a first-class behavior of the system.

---

## System Architecture

![Company Knowledge Agent Architecture](assets/architecture.png)

The application follows this high-level flow:

```text
Employee Question
       |
       v
FastAPI /ask
       |
       v
LangGraph
       |
       v
Hybrid Retrieval
   /         \
  v           v
Dense       BM25
Vector      Search
Search
  \           /
   \         /
       v
Reciprocal Rank Fusion
       |
       v
Top Retrieved Evidence
       |
       v
Grounded Answer Generation
       |
       +----------------------+
       |                      |
       v                      v
Evidence supports       Evidence does not
the question            support the question
       |                      |
       v                      v
     ANSWER                 ABSTAIN
       |
       v
Semantic Source Attribution
       |
       v
Structured API Response
       |
       v
Web UI
```

### Infrastructure

```text
Docker Compose
|
+-- FastAPI Application
|
+-- PostgreSQL 17 + pgvector
|
+-- Persistent PostgreSQL Volume

Ollama
|
+-- Local Llama Model
```

---

## Retrieval Pipeline

### 1. Dense Retrieval

The semantic retrieval pipeline converts queries and document chunks into embeddings and searches the pgvector-backed document store.

Query expansion is used to improve semantic retrieval for alternate phrasings.

### 2. BM25 Lexical Retrieval

A separate BM25 retriever indexes both:

```text
section title + chunk content
```

This gives the system strong exact-term and keyword matching in addition to semantic similarity.

### 3. Reciprocal Rank Fusion

The vector and BM25 rankings are combined using Reciprocal Rank Fusion:

```text
RRF(d) = Σ weight / (k + rank(d))
```

The implementation currently uses:

```text
k = 60
vector_weight = 1.0
bm25_weight = 1.0
```

Each retriever independently produces candidate chunks before the final fused ranking is calculated.

The LangGraph pipeline uses the top three fused chunks as evidence for answer generation.

---

## Why Hybrid Retrieval?

Dense and lexical retrieval fail in different ways.

The retrieval benchmark demonstrates this directly.

| Strategy | Hit@1 | Hit@3 |
|---|---:|---:|
| Vector + Query Expansion | 9/10 | 9/10 |
| BM25 | 9/10 | **10/10** |
| Equal-Weight RRF Hybrid | 9/10 | **10/10** |

One benchmark query paraphrasing the payroll date was missed by vector retrieval but ranked **#1 by BM25**. Hybrid RRF retained the correct section at **#2**, keeping it available to the downstream grounded-answer stage.

Hybrid retrieval therefore provides complementary semantic and lexical signals instead of relying on one retrieval strategy.

---

## Grounded Answer Generation

Retrieved evidence is passed to a local LLM with strict grounding rules.

The model is instructed to:

- use only retrieved evidence;
- avoid outside knowledge;
- avoid inventing company policies;
- avoid assuming missing details;
- distinguish broader policies from narrower questions;
- support every factual claim with evidence;
- abstain when the requested information is unspecified.

The model must return one of two structured prefixes:

```text
ANSWER:
```

or:

```text
ABSTAIN:
```

The application parses this result into:

```python
{
    "answerable": True | False,
    "answer": "..."
}
```

### Fail-Closed Design

The grounding layer also performs defensive abstention detection.

If the LLM returns an apparent answer containing language such as:

```text
not specified
does not mention
cannot determine
not provided
```

the system converts the response into an abstention.

If the model ignores the required `ANSWER:` / `ABSTAIN:` protocol entirely, the system also abstains instead of guessing what the model intended.

---

## Source Attribution

Retrieval candidates and answer citations are deliberately treated as different concepts.

A retrieved chunk may be relevant to the question without actually supporting the generated answer.

After a supported answer is generated:

1. The answer is embedded.
2. Each retrieved evidence chunk is embedded.
3. Cosine similarity is calculated between the answer and each chunk.
4. Evidence is ranked by semantic support.
5. The strongest supporting source is returned through the API.

This prevents every retrieved candidate from being displayed as though it directly supports the answer.

Abstained responses return no supporting sources.

---

## LangGraph Workflow

The current graph is intentionally small and explicit:

```text
START
  |
  v
retrieve
  |
  v
grounded_answer
  |
  v
END
```

### `retrieve`

Runs hybrid retrieval and stores:

- filename
- page
- section
- content
- vector similarity
- vector rank
- BM25 rank
- RRF score

### `grounded_answer`

Determines whether the retrieved evidence supports the question and either produces a grounded answer or abstains.

Keeping these stages explicit makes the pipeline easier to inspect, evaluate, and extend.

---

## Evaluation

Evaluation was treated as part of the system rather than as an afterthought.

### Grounded Answer Evaluation

Development evaluation:

```text
39 / 40 correct
97.5% accuracy
0 false positives
1 false negative
```

The single failure was conservative: relevant remote-work evidence was retrieved, but the answer layer abstained.

Importantly, the development evaluation produced **zero false positives**, meaning none of the tested unsupported questions were incorrectly presented as supported answers.

### Holdout Evaluation

A separate 24-question holdout suite contained both supported and unsupported questions.

```text
24 / 24 correct
100.0% holdout accuracy
0 false positives
0 false negatives
```

The holdout suite included:

```text
12 answerable questions
12 unanswerable questions
```

Examples included:

- payroll dates;
- federal-holiday payroll behavior;
- PTO notice requirements;
- sick leave for family members;
- VPN requirements;
- MFA requirements;
- proprietary code and public AI assistants;
- intellectual-property ownership;
- unsupported PTO carryover rules;
- unsupported severance policies;
- unsupported overseas relocation rules;
- unsupported reimbursement policies.

These results describe performance on the project's evaluation suites and should not be interpreted as a claim of universal accuracy.

---

## Example

### Supported Question

```text
Question:
When does Nexus pay its employees?

Answerable:
true

Answer:
Employees are paid on a semi-monthly basis, on the 15th and
final day of each month.

Source:
employee_handbook.pdf
Section 3.1 Pay Periods & Direct Deposit
Page 3
```

### Unsupported Question

```text
Question:
Can unused PTO be carried over to the next year?

Answerable:
false

Answer:
I couldn't find this information in the available company documents.

Sources:
[]
```

The system retrieves PTO-related evidence but correctly avoids treating a PTO accrual policy as proof of an unstated carryover policy.

---

## API

The application exposes a FastAPI REST API.

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "company-knowledge-agent"
}
```

### Ask a Question

```http
POST /ask
Content-Type: application/json
```

Request:

```json
{
  "question": "When does Nexus pay its employees?"
}
```

Response:

```json
{
  "question": "When does Nexus pay its employees?",
  "answerable": true,
  "answer": "Employees are paid on a semi-monthly basis, on the 15th and final day of each month.",
  "sources": [
    {
      "filename": "employee_handbook.pdf",
      "section": "3.1 Pay Periods & Direct Deposit",
      "page": 3
    }
  ]
}
```

The API validates questions using Pydantic and limits incoming questions to 500 characters.

Internal database/model exceptions are converted into a generic HTTP 500 response rather than exposing internal implementation details.

---

## Web Interface

FastAPI serves a lightweight browser interface at:

```text
http://localhost:8000/
```

The UI allows employees to ask questions and displays either:

- a grounded answer with its supporting source; or
- a clear unsupported-answer state.

FastAPI's interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

## Technology Stack

### AI / Retrieval

- Sentence Transformers
- Hugging Face
- Local Llama models
- Ollama
- BM25
- Reciprocal Rank Fusion
- LangGraph

### Backend

- Python 3.13
- FastAPI
- Pydantic
- SQLAlchemy
- Psycopg

### Data

- PostgreSQL 17
- pgvector

### Infrastructure

- Docker
- Docker Compose

### Document Processing

- PyMuPDF

---

## Running Locally

### Prerequisites

Install:

- Python 3.13
- Docker Desktop
- Ollama
- Git

Clone the repository:

```bash
git clone https://github.com/Gowthamch6s/company-knowledge-agent.git
cd company-knowledge-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Ollama

The application uses a local Ollama model for generation.

Check installed models:

```bash
ollama list
```

Pull the configured model if necessary:

```bash
ollama pull llama3.2
```

Verify Ollama is running before starting the application.

The Docker configuration can access the host Ollama service through:

```text
host.docker.internal
```

---

## Environment Configuration

Copy:

```text
.env.example
```

to:

```text
.env
```

Example configuration:

```env
POSTGRES_USER=knowledge_user
POSTGRES_PASSWORD=change_me
POSTGRES_DB=company_knowledge

DATABASE_URL=postgresql+psycopg://knowledge_user:change_me@localhost:5432/company_knowledge

EMBEDDING_DIMENSION=384
```

Do not commit `.env` or real credentials to Git.

---

## Docker

Build the application:

```bash
docker compose build app
```

Start the stack:

```bash
docker compose up
```

Docker Compose starts:

```text
company_knowledge_agent
company_knowledge_db
```

The API is exposed on:

```text
http://localhost:8000
```

PostgreSQL is exposed on:

```text
localhost:5432
```

Check container status:

```bash
docker compose ps
```

Stop the stack:

```bash
docker compose down
```

The PostgreSQL data is stored in a persistent Docker volume.

---

## Running Evaluations

### Hybrid Retrieval Benchmark

```bash
python -m eval.hybrid_retrieval_tests
```

Recorded benchmark:

```text
Strategy             Hit@1    Hit@3
-----------------------------------
Vector + Expansion    9/10     9/10
BM25                  9/10    10/10
Equal RRF Hybrid      9/10    10/10
```

### Grounded Answer Evaluation

```bash
python -m eval.grounded_answer_tests
```

Recorded result:

```text
Accuracy: 39/40 = 97.5%
False positives: 0
False negatives: 1
```

### Holdout Evaluation

```bash
python -m eval.holdout_evaluation
```

Recorded result:

```text
Accuracy: 24/24 = 100.0%
False positives: 0
False negatives: 0
```

---

## Design Principles

### Grounding Over Fluency

A polished answer is not useful if it invents company policy.

The system prioritizes evidence support over producing an answer to every question.

### Abstention Is a Feature

An enterprise knowledge system should be able to say:

```text
I don't have enough information.
```

rather than silently filling gaps with model knowledge.

### Retrieval Is Not Citation

Relevant retrieval candidates are not automatically treated as supporting citations.

Source attribution happens after answer generation.

### Hybrid Retrieval Over One Retrieval Method

Semantic and lexical retrieval provide complementary signals.

RRF allows both to contribute without requiring their raw scores to be directly comparable.

### Evaluation Before Optimization

Retrieval quality and answerability behavior are measured separately, including a dedicated holdout set.

---

## Current Limitations

This project is intentionally scoped as a local company-knowledge prototype.

Current limitations include:

- evaluation corpus is relatively small;
- BM25 currently constructs its index from stored chunks at query time;
- source attribution returns only the strongest supporting source through the API;
- local LLM behavior depends on the installed Ollama model;
- benchmark results are specific to the included evaluation corpus;
- the current document collection is small compared with a production enterprise knowledge base.

Potential production improvements include persistent lexical indexing, reranking, multi-document ingestion workflows, authentication, authorization, observability, caching, asynchronous inference, larger evaluation datasets, and document-level access controls.

---

## What I Learned

This project explored several problems that appear when moving beyond a basic RAG demo:

- why vector similarity alone is not enough;
- how BM25 complements semantic retrieval;
- how Reciprocal Rank Fusion combines heterogeneous rankings;
- why retrieval relevance does not guarantee answerability;
- how to design explicit abstention behavior;
- why citations should represent answer support rather than retrieval alone;
- how to evaluate answerable and unanswerable questions separately;
- how holdout evaluation helps detect overfitting to development questions;
- how to expose a RAG pipeline through a typed API;
- how to containerize the application and vector database.

---

## Future Work

Possible extensions include:

- cross-encoder reranking;
- persistent BM25 / full-text indexing;
- conversational memory;
- multi-document collections;
- document upload and automatic indexing;
- authentication and role-based access control;
- document-level permissions;
- streaming responses;
- tracing and observability;
- retrieval caching;
- automated regression evaluation;
- larger adversarial and holdout test suites;
- production deployment.

---

## Author

**Gowtham Sai Chimmana**

Built as an end-to-end AI engineering project focused on grounded enterprise knowledge retrieval, reliable RAG behavior, evaluation, and deployment.