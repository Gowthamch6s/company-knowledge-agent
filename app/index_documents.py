from pathlib import Path

from app.database import SessionLocal
from app.embeddings import create_embedding
from app.ingestion import chunk_pages, extract_pdf
from app.models import Document, DocumentChunk


DOCUMENTS_DIR = Path("documents")


def index_pdf(pdf_path: Path):
    print(f"Indexing: {pdf_path.name}")

    pages = extract_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    session = SessionLocal()

    try:
        document = Document(
            filename=pdf_path.name,
        )

        session.add(document)

        # Send the document INSERT to PostgreSQL so
        # PostgreSQL assigns an ID.
        session.flush()

        print(f"Document ID: {document.id}")

        for chunk in chunks:
            print(
                f"Embedding chunk {chunk['chunk_index']} "
                f"(page {chunk['page_number']})..."
            )

            embedding = create_embedding(
                chunk["content"]
            )

            db_chunk = DocumentChunk(
                document_id=document.id,
                page_number=chunk["page_number"],
                section_title=chunk["section_title"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                embedding=embedding,
            )

            session.add(db_chunk)

        session.commit()

        print()
        print("Document indexed successfully!")
        print(f"Stored {len(chunks)} chunks.")

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    pdf_path = DOCUMENTS_DIR / "employee_handbook.pdf"

    index_pdf(pdf_path)