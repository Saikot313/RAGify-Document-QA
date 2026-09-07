"""
Ties together PDF extraction, chunking, and vector store indexing into
a single "ingest this uploaded file" operation. This is what the
/documents/upload endpoint calls.
"""

from fastapi import UploadFile

from app.rag import chunking, vector_store
from app.services import pdf_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


def ingest_pdf(file: UploadFile, file_bytes: bytes) -> dict:
    """
    Validate, save, extract, chunk, and index an uploaded PDF.

    Returns a summary dict used to build the API response.
    Raises PDFValidationError / PDFExtractionError on failure, which the
    API layer translates into the appropriate HTTP error.
    """
    pdf_service.validate_pdf(file, file_bytes)
    file_path = pdf_service.save_pdf(file.filename, file_bytes)

    documents = pdf_service.extract_documents(file_path, source_name=file.filename)
    chunks = chunking.split_documents(documents)
    vector_store.add_documents(chunks)

    logger.info(f"Ingested '{file.filename}': {len(documents)} page(s), {len(chunks)} chunk(s)")

    return {
        "filename": file.filename,
        "pages_extracted": len(documents),
        "chunks_indexed": len(chunks),
    }
