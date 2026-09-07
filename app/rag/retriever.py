"""
Retrieval logic: given a user question, find the most relevant chunks
from the FAISS index.

This module is deliberately separate from qa_service so that retrieval
can be tested/reasoned about independently of the LLM call.
"""

from langchain_core.documents import Document

from app.config import get_settings
from app.rag.vector_store import load_index
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_relevant_chunks(question: str, top_k: int | None = None) -> list[Document]:
    """
    Return the top-k most relevant chunks for a question via FAISS
    similarity search. `top_k` defaults to the configured RETRIEVAL_TOP_K.
    """
    settings = get_settings()
    store = load_index()
    k = top_k or settings.retrieval_top_k

    results = store.similarity_search(question, k=k)
    logger.info(f"Retrieved {len(results)} chunk(s) for question: {question[:60]!r}")
    return results
