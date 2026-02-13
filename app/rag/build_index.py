from __future__ import annotations

# Import the dataclass decorator to define a simple config object.
from dataclasses import dataclass

# Import List for type hints.
from typing import List

# Path helps us work with filesystem paths safely/cross-platform.
from pathlib import Path

# Document is LangChain’s standard container for text + metadata.
from langchain_core.documents import Document

# Text splitter that chunks documents into overlapping segments for RAG.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local embedding model wrapper (uses sentence-transformers under the hood).
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma is a local vector database used to store embeddings and perform similarity search.
from langchain_community.vectorstores import Chroma

import re


@dataclass
class IndexConfig:
    # Where Chroma should persist its data on disk.
    persist_dir: str = "indexes/chroma"

    # A logical name for the collection of vectors.
    collection_name: str = "class5_science"

    # Directory that contains your course markdown/text files.
    course_dir: str = "course_data"

    # Approx size of each text chunk.
    chunk_size: int = 1200

    # Overlap between chunks to preserve context across chunk boundaries.
    chunk_overlap: int = 150

    # Name of the local embedding model.
    # This is a good default and widely used for semantic search.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Allowed file extensions for ingestion.
    allowed_exts: tuple[str, ...] = (".md", ".txt")

def infer_lesson_id(filename: str) -> int | None:
    """
    Extracts the lesson/chapter number from filenames like:
      lesson1_plants.txt
      lesson12_animals.md
    Returns None if not found.
    """
    m = re.search(r"lesson\s*(\d+)", filename.lower())
    return int(m.group(1)) if m else None


def load_docs_from_path(course_dir: str, allowed_exts: tuple[str, ...]) -> List[Document]:
    """
    Loads all course files directly from the filesystem and wraps them as LangChain Documents.
    """
    # Resolve the course directory path (absolute path).
    base_dir = Path(course_dir).resolve()

    # Fail fast if the directory does not exist.
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Course directory not found: {base_dir}")

    # Create an empty list to store Document objects.
    docs: List[Document] = []

    # Iterate through all files under course_dir (recursively).
    for path in sorted(base_dir.rglob("*")):
        # Skip directories.
        if not path.is_file():
            continue

        # Only ingest allowed file types.
        if path.suffix.lower() not in allowed_exts:
            continue

        # Read the file text.
        text = path.read_text(encoding="utf-8")

        # Store a relative path as the source for cleaner citations.
        source = str(path.relative_to(base_dir))

        # Example: "lesson1_plants.txt" -> lesson_id = 1
        lesson_id = infer_lesson_id(path.name)

        # Wrap raw text + metadata into a Document. Metadata is used for citations later.
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,           # e.g., "lesson2_plants.txt"
                    "abs_path": str(path),      # optional: full path for debugging
                    "chapter_id": lesson_id,
                },
            )
        )

    # Fail fast if no docs were found (helps catch wrong folder path).
    if not docs:
        raise RuntimeError(f"No .md/.txt files found under: {base_dir}")

    # Return the list of loaded documents.
    return docs


def main() -> None:
    # Create a config instance so values are easy to change in one place.
    cfg = IndexConfig()

    # Load your lesson documents directly from the filesystem.
    raw_docs = load_docs_from_path(cfg.course_dir, cfg.allowed_exts)

    # Create a chunker that splits documents into overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "#"],
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    # Split the raw documents into smaller chunks for embedding and retrieval.
    chunks = splitter.split_documents(raw_docs)

    # Create an embedding function using a local sentence-transformers model.
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model_name)

    # Create a Chroma vector store from the chunks and persist it to disk.
    vectordb = Chroma.from_documents(
        documents=chunks,                      # The chunked documents we want to store.
        embedding=embeddings,                  # The embedding function for semantic indexing.
        persist_directory=cfg.persist_dir,     # Where to store the DB on disk.
        collection_name=cfg.collection_name,   # Logical name of this dataset.
    )

    # Persist the vector DB so it can be reused later during Q&A.
    vectordb.persist()

    # Print a summary so you know indexing succeeded.
    print(f"Indexed {len(raw_docs)} files into {len(chunks)} chunks.")
    print(f"Chroma saved at: {cfg.persist_dir}, collection: {cfg.collection_name}")


# Run main only when executing this file directly.
if __name__ == "__main__":
    main()
