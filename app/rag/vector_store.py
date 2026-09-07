"""
Manages the FAISS vector index: creating it from document chunks, saving
it to disk, loading it back, and adding new documents to an existing
index (so multiple PDFs can be uploaded over time).

Kept intentionally simple: a single local FAISS index on disk, no
external vector database service.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.embeddings import get_embeddings_model
from app.utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILE = "index.faiss"


def index_exists() -> bool:
    settings = get_settings()
    return os.path.exists(os.path.join(settings.vector_store_dir, _INDEX_FILE))


def load_index() -> FAISS:
    """Load the FAISS index from disk. Raises FileNotFoundError if none exists."""
    settings = get_settings()
    if not index_exists():
        raise FileNotFoundError("No vector store found. Upload a document first.")

    return FAISS.load_local(
        settings.vector_store_dir,
        get_embeddings_model(),
        allow_dangerous_deserialization=True,  # safe: we only ever load our own local index
    )


def add_documents(chunks: list[Document]) -> FAISS:
    """
    Add chunks to the vector store, creating a new index if one doesn't
    exist yet, or merging into the existing one otherwise.
    """
    settings = get_settings()
    os.makedirs(settings.vector_store_dir, exist_ok=True)
    embeddings = get_embeddings_model()

    if index_exists():
        store = load_index()
        store.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunk(s) to existing vector store")
    else:
        store = FAISS.from_documents(chunks, embeddings)
        logger.info(f"Created new vector store with {len(chunks)} chunk(s)")

    store.save_local(settings.vector_store_dir)
    return store
