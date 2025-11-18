import argparse
import os
import shutil
import re
from pypdf import PdfReader
from get_embedding_function import get_embedding_function
import chromadb


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    documents = []
    # Walk the data directory and read all PDFs, page by page.
    for root, _dirs, files in os.walk(DATA_PATH):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fname)
            try:
                reader = PdfReader(path)
            except Exception:
                print(f"Warning: failed to read PDF: {path}")
                continue
            for i, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                documents.append({
                    "page_content": text,
                    "metadata": {"source": path, "page": i},
                })
    return documents


def split_documents(documents: list):
    # Table-detection helper: returns (is_table, rows)
    def detect_table(text: str):
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return False, None

        # Heuristics: presence of pipe separators or many lines with multi-space
        pipe_count = sum(1 for l in lines if '|' in l)
        multi_space_count = sum(1 for l in lines if re.search(r"\s{2,}", l))
        comma_separated_count = sum(1 for l in lines if l.count(',') >= 2)

        if pipe_count >= max(1, len(lines) // 5):
            # split using pipe
            rows = [ [c.strip() for c in l.split('|') if c.strip()] for l in lines ]
            return True, rows
        if multi_space_count >= max(2, len(lines) // 4):
            rows = [ [c.strip() for c in re.split(r"\s{2,}", l) if c.strip()] for l in lines ]
            return True, rows
        if comma_separated_count >= max(2, len(lines) // 3):
            rows = [ [c.strip() for c in l.split(',') if c.strip()] for l in lines ]
            return True, rows

        return False, None

    chunk_size = 800
    chunk_overlap = 80
    chunks = []

    for doc in documents:
        text = doc.get("page_content", "")
        page = doc.get("metadata", {}).get("page")
        source = doc.get("metadata", {}).get("source")

        is_table, rows = detect_table(text)
        if is_table and rows:
            # Create a table-aware chunk: serialize rows into a readable preview
            # Keep a small preview in metadata for display; use a flattened
            # representation as the chunk content for embedding.
            preview_rows = []
            for r in rows[:10]:
                preview_rows.append(' | '.join(r))
            table_preview = '\n'.join(preview_rows)
            flattened = '\n'.join(' | '.join(r) for r in rows)
            chunks.append({
                "page_content": flattened,
                "metadata": {
                    "source": source,
                    "page": page,
                    "is_table": True,
                    "table_preview": table_preview,
                    "table_rows": len(rows),
                },
            })
            continue

        # Non-table text: chunk by characters with overlap
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "page_content": chunk_text,
                "metadata": {"source": source, "page": page, "is_table": False},
            })
            start = end - chunk_overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break

    return chunks


def add_to_chroma(chunks: list):
    # Use chromadb client directly to avoid depending on LangChain's adapter.
    # Construct a client using the current chromadb API (no legacy Settings).
    # Make the client persistent on disk so other processes can read the data.
    from chromadb.config import Settings
    client = chromadb.Client(settings=Settings(is_persistent=True, persist_directory=CHROMA_PATH))
    # Collection name 'documents' is used; create if missing.
    try:
        collection = client.get_collection("documents")
    except Exception:
        collection = client.create_collection("documents")

    # Calculate IDs and prepare lists for Chroma's collection.add.
    chunks_with_ids = calculate_chunk_ids(chunks)

    texts = [c["page_content"] for c in chunks_with_ids]
    metadatas = [c["metadata"] for c in chunks_with_ids]
    ids = [c["metadata"].get("id") for c in chunks_with_ids]

    # Determine existing IDs to avoid duplicates (best-effort).
    try:
        existing = collection.get(include=["ids"])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    new_texts = []
    new_metadatas = []
    new_ids = []
    for t, m, id_ in zip(texts, metadatas, ids):
        if id_ not in existing_ids:
            new_texts.append(t)
            new_metadatas.append(m)
            new_ids.append(id_)

    if new_texts:
        print(f"👉 Adding new documents: {len(new_texts)}")
        # Compute embeddings in batch using the project's embedding function.
        embed_fn = get_embedding_function()
        try:
            embeddings = embed_fn(new_texts)
        except Exception:
            # Fallback: compute one-by-one
            embeddings = [embed_fn(t) for t in new_texts]

        collection.add(documents=new_texts, metadatas=new_metadatas, ids=new_ids, embeddings=embeddings)
        try:
            client.persist()
        except Exception:
            pass
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source")
        page = chunk.get("metadata", {}).get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.setdefault("metadata", {})["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()