from fastapi.testclient import TestClient
import pytest
from app.api.main import app

client = TestClient(app)


def test_unsupported_rag_question_has_controlled_no_answer_behavior():
    r = client.post('/rag/ask', json={'question': 'What is the internal policy for teleporting an order to Mars?'})
    if r.status_code == 501:
        pytest.skip('Starter contract: implement RAG before final submission')
    assert r.status_code == 200
    body = r.json()
    text = str(body).lower()
    assert any(term in text for term in ['no answer', 'not found', 'insufficient', 'cannot determine', 'unsupported'])
