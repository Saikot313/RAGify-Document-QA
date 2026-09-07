def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_before_any_document_indexed_returns_400(client):
    response = client.post("/ask", json={"question": "What is this document about?"})

    assert response.status_code == 400
    assert "upload" in response.json()["detail"].lower()


def test_ask_returns_answer_and_sources_after_indexing(
    client, sample_pdf_bytes, mock_embeddings, mock_llm_answer
):
    upload_response = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert upload_response.status_code == 201

    ask_response = client.post("/ask", json={"question": "What does LangChain do?"})

    assert ask_response.status_code == 200
    body = ask_response.json()
    assert body["answer"] == mock_llm_answer.content
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["source"] == "sample.pdf"
    assert body["sources"][0]["page"] in (1, 2)
