def test_upload_rejects_non_pdf_file(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


def test_upload_indexes_a_valid_pdf(client, sample_pdf_bytes, mock_embeddings):
    response = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "sample.pdf"
    assert body["pages_extracted"] == 2
    assert body["chunks_indexed"] >= 1


def test_upload_makes_health_report_vector_store_ready(client, sample_pdf_bytes, mock_embeddings):
    client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )

    health = client.get("/health").json()
    assert health["vector_store_ready"] is True
