import importlib.util
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'starter_code' / '10_api' / 'main.py'
spec = importlib.util.spec_from_file_location('real_api_main', PATH)
real_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_main)

client = TestClient(real_main.app)
client.__enter__()  # manually trigger startup event so the model/RAG index load


def test_unsupported_rag_question_has_controlled_no_answer_behavior():
    r = client.post('/ask/policy', json={'question': 'What is the internal policy for teleporting an order to Mars?'})
    assert r.status_code == 200
    body = r.json()
    text = str(body).lower()
    assert any(term in text for term in ['no answer', 'not found', 'insufficient', 'cannot determine', 'cannot confirm', 'unsupported'])


def test_rag_response_includes_sources_and_score():
    r = client.post('/ask/policy', json={'question': 'How many days do I have to return an item?'})
    assert r.status_code == 200
    body = r.json()
    assert 'answer' in body
    assert 'sources' in body
    assert 'top_retrieval_score' in body
    assert isinstance(body['sources'], list)
    assert len(body['sources']) > 0