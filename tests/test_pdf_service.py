import pytest

from app.services.pdf_service import (
    PDFExtractionError,
    PDFValidationError,
    extract_documents,
    save_pdf,
    validate_pdf,
)


class DummyUploadFile:
    """Minimal stand-in for FastAPI's UploadFile — only `.filename` is needed here."""

    def __init__(self, filename: str):
        self.filename = filename


def test_validate_pdf_rejects_non_pdf_extension():
    with pytest.raises(PDFValidationError):
        validate_pdf(DummyUploadFile("notes.txt"), b"hello world")


def test_validate_pdf_rejects_empty_file():
    with pytest.raises(PDFValidationError):
        validate_pdf(DummyUploadFile("empty.pdf"), b"")


def test_validate_pdf_accepts_valid_pdf(sample_pdf_bytes):
    validate_pdf(DummyUploadFile("sample.pdf"), sample_pdf_bytes)  # should not raise


def test_save_and_extract_documents_returns_correct_page_metadata(sample_pdf_bytes):
    path = save_pdf("sample.pdf", sample_pdf_bytes)
    documents = extract_documents(path, source_name="sample.pdf")

    assert len(documents) == 2
    assert documents[0].metadata == {"source": "sample.pdf", "page": 1}
    assert documents[1].metadata == {"source": "sample.pdf", "page": 2}
    assert "LangChain" in documents[0].page_content


def test_extract_documents_raises_on_unreadable_file(tmp_path):
    bad_path = tmp_path / "bad.pdf"
    bad_path.write_bytes(b"this is not a real pdf file")

    with pytest.raises(PDFExtractionError):
        extract_documents(str(bad_path), source_name="bad.pdf")
