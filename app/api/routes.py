"""
API route definitions.

Endpoints:
    GET  /health
    POST /documents/upload
    POST /ask
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.rag.vector_store import index_exists
from app.schemas.schemas import AskRequest, AskResponse, HealthResponse, UploadResponse
from app.services import ingestion_service, qa_service
from app.services.pdf_service import PDFExtractionError, PDFValidationError
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", vector_store_ready=index_exists())


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    file_bytes = await file.read()

    try:
        result = ingestion_service.ingest_pdf(file, file_bytes)
    except PDFValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert unexpected errors into a clean 500
        logger.error(f"Unexpected error during upload: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the document. Please try again.",
        ) from exc

    return UploadResponse(**result, message="Document indexed successfully.")


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    try:
        answer, sources = qa_service.answer_question(question)
    except qa_service.NoDocumentsIndexedError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents indexed yet. Upload a PDF first.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - e.g. OpenAI API errors, network errors
        logger.error(f"Error answering question: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to get an answer from the language model. Please try again.",
        ) from exc

    return AskResponse(answer=answer, sources=sources)
