from pathlib import Path
import re

import fitz


DOCUMENTS_DIR = Path("documents")


def extract_pdf(pdf_path: Path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append(
            {
                "page_number": page_number,
                "text": text.strip(),
            }
        )

    document.close()

    return pages


def chunk_text(
    text: str,
    chunk_size: int = 800,
):
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = (
            f"{current_chunk}\n{paragraph}".strip()
            if current_chunk
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_pages(pages):
    all_chunks = []
    chunk_index = 0

    # Matches headings such as:
    # 1.1 A Message from Leadership
    # 2.3 Attendance, Hours, and Remote Work Policy
    # 4.1 Paid Time Off (PTO)
    # 5.2 Intellectual Property (IP) & Code Ownership
    section_pattern = re.compile(
        r"^\d+\.\d+\s+.+$"
    )

    current_section = None
    current_content = []
    current_page = None

    def save_section():
        nonlocal chunk_index

        if not current_content:
            return

        content = "\n".join(current_content).strip()

        if not content:
            return

        all_chunks.append(
            {
                "chunk_index": chunk_index,
                "page_number": current_page,
                "section_title": current_section,
                "content": content,
            }
        )

        chunk_index += 1

    for page in pages:
        lines = [
            line.strip()
            for line in page["text"].split("\n")
            if line.strip()
        ]

        for line in lines:

            if section_pattern.match(line):

                # Save previous section before starting
                # the new one.
                save_section()

                current_section = line
                current_content = [line]
                current_page = page["page_number"]

            else:
                if current_page is None:
                    current_page = page["page_number"]

                current_content.append(line)

    # Save final section
    save_section()

    return all_chunks


if __name__ == "__main__":
    pdf_path = DOCUMENTS_DIR / "employee_handbook.pdf"

    pages = extract_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"PDF: {pdf_path.name}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Sections created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 70)

        print(
            f"CHUNK {chunk['chunk_index']} "
            f"| PAGE {chunk['page_number']}"
        )

        print(
            f"SECTION: {chunk['section_title']}"
        )

        print("=" * 70)

        print(chunk["content"])