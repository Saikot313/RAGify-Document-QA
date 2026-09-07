"""
Prompt template for document-grounded question answering.

Kept in its own module (separate from API/service logic) so the prompt
engineering is easy to find, read, and iterate on independently of the
rest of the application.
"""

from langchain_core.prompts import ChatPromptTemplate

QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided document context.

Rules:
- Base your answer strictly on the given context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, \
say clearly: "I cannot find this information in the provided documents."
- Do not make up facts, numbers, or sources that are not in the context.
- Keep answers concise and directly useful — avoid unnecessary padding.
- If multiple parts of the context are relevant, synthesize them into one \
coherent answer rather than listing them separately.
"""

QA_HUMAN_PROMPT = """Context from the document(s):
---
{context}
---

Question: {question}

Answer the question using only the context above."""


def build_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            ("human", QA_HUMAN_PROMPT),
        ]
    )


def format_context(chunks) -> str:
    """
    Turn retrieved chunks into a single context string, each chunk
    labeled with its source and page so the LLM's reasoning stays
    traceable (even though citations are built from metadata directly,
    not parsed from the LLM's output).
    """
    parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page: {page}]\n{chunk.page_content}")
    return "\n\n".join(parts)
