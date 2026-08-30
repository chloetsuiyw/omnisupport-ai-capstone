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
 
 
def test_empty_rag_question_rejected():
    r = client.post('/ask/policy', json={"question": ""})
    assert r.status_code == 400
 
 
def test_missing_required_field_rejected():
    r = client.post('/ask/policy', json={"wrong_field": "hello"})
    assert r.status_code == 422
 