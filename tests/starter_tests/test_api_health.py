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
 
 
def test_api_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'
 