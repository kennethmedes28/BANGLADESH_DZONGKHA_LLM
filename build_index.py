"""
build_index.py — One-time script to populate your FAISS vector store.
Run this before launching the pipeline for the first time.

Usage:
    python build_index.py --source ./docs --index faiss_index
"""

import argparse
import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pipeline import FAISSManager


def build(source_dir: str, index_path: str, chunk_size: int, chunk_overlap: int):
    print(f"📂 Loading documents from '{source_dir}' …")

    # ── Load PDFs ──────────────────────────────────────────────────────
    pdf_loader = DirectoryLoader(
        source_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    # ── Load plain text / markdown ─────────────────────────────────────
    txt_loader = DirectoryLoader(
        source_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    docs = pdf_loader.load() + txt_loader.load()
    print(f"   Loaded {len(docs)} raw documents.")

    # ── Chunk ──────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")

    # ── Embed + Index ──────────────────────────────────────────────────
    mgr = FAISSManager(index_path=index_path)
    mgr.build_from_documents(chunks)
    print(f"\n✅ FAISS index saved to '{index_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from a document directory.")
    parser.add_argument("--source",  default="./docs",       help="Directory containing source files")
    parser.add_argument("--index",   default="faiss_index",  help="Output FAISS index directory")
    parser.add_argument("--chunk",   type=int, default=500,  help="Chunk size (tokens)")
    parser.add_argument("--overlap", type=int, default=50,   help="Chunk overlap (tokens)")
    args = parser.parse_args()

    # Make sure OPENAI_API_KEY is set (needed for embeddings)
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running this script.")

    build(args.source, args.index, args.chunk, args.overlap)
