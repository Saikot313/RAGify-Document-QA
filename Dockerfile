# Single-stage build — this is a small portfolio app, no need for
# multi-stage builds or extra infrastructure.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer when only
# application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and static frontend.
COPY app/ ./app/
COPY frontend/ ./frontend/

# Runtime data directories (uploaded PDFs + FAISS index). Mount these as
# a volume in production if you want data to survive container restarts.
RUN mkdir -p data/uploads data/vector_store

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
