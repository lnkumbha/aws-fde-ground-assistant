from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_loaded_documents():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["documents_loaded"] >= 5
    assert payload["chunks_indexed"] >= 5


def test_ask_returns_answer_with_citations_for_known_question():
    response = client.post(
        "/ask", json={"question": "How long are application logs retained?"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answered"] is True
    assert payload["citations"]
    assert payload["citations"][0]["source"] == "data-retention.md"
    assert "30 days" in payload["answer"]


def test_ask_abstains_for_unknown_question():
    response = client.post(
        "/ask", json={"question": "What is the office cafeteria menu?"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answered"] is False
    assert payload["reason"] == "insufficient_evidence"
    assert payload["citations"] == []


def test_question_validation_rejects_too_short_input():
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422
