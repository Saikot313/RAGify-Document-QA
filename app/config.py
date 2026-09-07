"""
Application configuration.

All configurable values are loaded from environment variables (via a .env
file in development). Nothing sensitive is hardcoded here.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- OpenAI ---
    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Retrieval ---
    retrieval_top_k: int = 4

    # --- Storage ---
    upload_dir: str = "data/uploads"
    vector_store_dir: str = "data/vector_store"

    # --- Upload limits ---
    max_file_size_mb: int = 20

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance so we don't re-read env vars every call."""
    return Settings()
