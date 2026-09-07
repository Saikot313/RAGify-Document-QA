"""
Shared pytest fixtures.

Key ideas:
- Every test gets its own isolated upload/vector-store directory (tmp_path)
  so tests never interfere with each other or leave real artifacts behind.
- OpenAI embedding and chat calls are mocked via dedicated fixtures so the
  test suite does not require a real API key or spend any credits.
"""

import os

# Set a dummy key before anything imports app.config, so app startup's
# config validation doesn't fail in the test environment.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.rag import embeddings as embeddings_module
from app.services import qa_service


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Point uploads/vector-store at a temp dir and reset cached settings/clients."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))

    get_settings.cache_clear()
    embeddings_module.get_embeddings_model.cache_clear()
    qa_service._get_llm.cache_clear()

    yield

    get_settings.cache_clear()
    embeddings_module.get_embeddings_model.cache_clear()
    qa_service._get_llm.cache_clear()


@pytest.fixture
def client():
    """A fresh TestClient per test (app is re-imported lazily via app.main)."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes(tmp_path):
    """A small, real 2-page PDF used across extraction/upload/ask tests."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(
        0, 10, "LangChain connects large language models with external data "
        "sources such as FAISS vector databases."
    )
    pdf.add_page()
    pdf.multi_cell(
        0, 10, "The retrieval-augmented generation approach reduces "
        "hallucination by grounding answers in retrieved context."
    )

    path = tmp_path / "sample.pdf"
    pdf.output(str(path))
    return path.read_bytes()


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Replace real OpenAI embedding calls with deterministic fake vectors."""

    def fake_embed_documents(self, texts):
        return [np.random.rand(1536).tolist() for _ in texts]

    def fake_embed_query(self, text):
        return np.random.rand(1536).tolist()

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings.embed_documents", fake_embed_documents)
    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings.embed_query", fake_embed_query)


@pytest.fixture
def mock_llm_answer(monkeypatch):
    """Replace the real ChatOpenAI call with a fixed, predictable response."""
    from unittest.mock import MagicMock

    fake_response = MagicMock()
    fake_response.content = "This is a mocked answer grounded in the retrieved context."

    monkeypatch.setattr("langchain_openai.ChatOpenAI.invoke", lambda self, *a, **kw: fake_response)
    return fake_response
