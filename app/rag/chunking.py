"""
Splits page-level Documents into smaller overlapping chunks suitable for
embedding and retrieval.

Why RecursiveCharacterTextSplitter with ~800/150 chars:
- Large enough that each chunk still contains meaningful, self-contained
  context for the LLM to reason about.
- Small enough that similarity search stays precise (a chunk that mixes
  too many topics produces noisy embeddings).
- The overlap prevents a sentence/idea from being awkwardly cut in half
  right at a chunk boundary, which would otherwise hurt retrieval
  quality for questions about that boundary content.
Both values are configurable via environment variables.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split page-level documents into smaller chunks.

    Each resulting chunk keeps its parent page's metadata (source, page)
    and gets an additional `chunk_id` for traceability/citations.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    logger.info(f"Split {len(documents)} page(s) into {len(chunks)} chunk(s)")
    return chunks
