from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_invalid_structured_output_request_schema():
    r = client.post('/tickets/extract', json={"wrong_field": "hello"})
    assert r.status_code == 422


def test_empty_ticket_text_rejected():
    r = client.post('/tickets/extract', json={"text": ""})
    assert r.status_code == 422


def test_empty_rag_question_rejected():
    r = client.post('/rag/ask', json={"question": ""})
    assert r.status_code == 422
