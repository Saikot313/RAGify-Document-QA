"""
Handles everything related to a raw uploaded PDF file:
- basic validation (extension, size)
- saving to disk
- text extraction (per page, with page-number metadata)
- basic text cleaning

This module does NOT know about chunking, embeddings, or the vector
store — it only turns "a PDF on disk" into "clean LangChain Documents".
"""

import os
import re
from pathlib import Path

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PDFValidationError(Exception):
    """Raised when an uploaded file fails basic validation."""


class PDFExtractionError(Exception):
    """Raised when text cannot be extracted from a PDF."""


def validate_pdf(file: UploadFile, file_bytes: bytes) -> None:
    """Validate file extension and size before we do any real work."""
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise PDFValidationError("Only .pdf files are supported.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise PDFValidationError(
            f"File too large ({size_mb:.1f} MB). "
            f"Max allowed size is {settings.max_file_size_mb} MB."
        )

    if size_mb == 0:
        raise PDFValidationError("Uploaded file is empty.")


def save_pdf(filename: str, file_bytes: bytes) -> str:
    """Save the raw uploaded PDF to the upload directory and return its path."""
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Avoid path traversal / weird characters in filenames.
    safe_filename = re.sub(r"[^\w.\-]", "_", filename)
    file_path = Path(settings.upload_dir) / safe_filename

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Saved uploaded PDF: {safe_filename}")
    return str(file_path)


def _clean_text(text: str) -> str:
    """Light cleanup: collapse excessive whitespace/newlines from PDF extraction."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_documents(file_path: str, source_name: str) -> list[Document]:
    """
    Extract text from a PDF, one LangChain Document per page.

    Each Document's metadata includes:
      - source: original filename (used for citations)
      - page: 1-indexed page number (used for citations)

    Raises PDFExtractionError if no extractable text is found (e.g. a
    scanned/image-only PDF, which this project does not OCR).
    """
    try:
        loader = PyPDFLoader(file_path)
        raw_pages = loader.load()
    except Exception as exc:
        raise PDFExtractionError(f"Could not read PDF: {exc}") from exc

    documents: list[Document] = []
    for raw_page in raw_pages:
        cleaned = _clean_text(raw_page.page_content)
        if not cleaned:
            continue  # skip blank pages

        page_number = raw_page.metadata.get("page", 0) + 1  # PyPDFLoader is 0-indexed
        documents.append(
            Document(
                page_content=cleaned,
                metadata={"source": source_name, "page": page_number},
            )
        )

    if not documents:
        raise PDFExtractionError(
            "No extractable text found in this PDF. "
            "It may be a scanned/image-only document."
        )

    logger.info(f"Extracted {len(documents)} page(s) with text from {source_name}")
    return documents
