from langchain_core.documents import Document

from app.rag.chunking import split_documents


def test_split_documents_produces_multiple_chunks_for_long_text():
    long_text = "RAG stands for Retrieval-Augmented Generation. " * 60
    docs = [Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1
        assert "chunk_id" in chunk.metadata


def test_split_documents_keeps_short_text_as_single_chunk():
    docs = [Document(page_content="Short text.", metadata={"source": "a.pdf", "page": 1})]

    chunks = split_documents(docs)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_chunk_ids_are_sequential_across_multiple_pages():
    docs = [
        Document(page_content="First page content here.", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Second page content here.", metadata={"source": "a.pdf", "page": 2}),
    ]

    chunks = split_documents(docs)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert ids == list(range(len(chunks)))
