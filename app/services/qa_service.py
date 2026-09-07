
from functools import lru_cache

from langchain_ollama import ChatOllama

from app.config import get_settings
from app.prompts.qa_prompt import build_qa_prompt, format_context
from app.rag.retriever import retrieve_relevant_chunks
from app.schemas.schemas import SourceCitation
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NoDocumentsIndexedError(Exception):
    """Raised when a question is asked before any document has been indexed."""


@lru_cache
def _get_llm() -> ChatOllama:
    return ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

def _build_citations(chunks) -> list[SourceCitation]:
    unique_keys: set[tuple[str, int | None]] = set()

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        unique_keys.add((source, page))

    sorted_keys = sorted(unique_keys, key=lambda k: (k[0], k[1] if k[1] is not None else -1))
    return [SourceCitation(source=source, page=page) for source, page in sorted_keys]


def answer_question(question: str) -> tuple[str, list[SourceCitation]]:
    """
    Run the full RAG answer flow: retrieve -> prompt -> LLM -> citations.

    Raises NoDocumentsIndexedError if the vector store doesn't exist yet.
    """
    try:
        chunks = retrieve_relevant_chunks(question)
    except FileNotFoundError as exc:
        raise NoDocumentsIndexedError(str(exc)) from exc

    if not chunks:
        return "I cannot find this information in the provided documents.", []

    prompt = build_qa_prompt()
    context = format_context(chunks)
    chain = prompt | _get_llm()

    response = chain.invoke({"context": context, "question": question})
    answer_text = response.content

    citations = _build_citations(chunks)
    logger.info(f"Answered question with {len(citations)} source(s)")

    return answer_text, citations
