"""
FastAPI application entrypoint.
"""

import sys

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from app.config import get_settings

    get_settings()  # fail fast with a clear message if required env vars are missing
except ValidationError as exc:
    logger.error(
        "Missing or invalid configuration. Did you set OPENAI_API_KEY? "
        f"Details: {exc}"
    )
    sys.exit(1)

from app.api.routes import router  # noqa: E402 (import after config check)

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("RAG Document Q&A API starting up")
    yield


app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload PDFs and ask questions about them using Retrieval-Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# Open CORS for this portfolio project so the simple static frontend can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
