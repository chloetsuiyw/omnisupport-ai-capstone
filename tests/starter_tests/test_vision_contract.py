from fastapi.testclient import TestClient
import pytest
from app.api.main import app

client = TestClient(app)


def test_invalid_image_input_is_controlled():
    r = client.post('/vision/predict', json={'image_path': 'data/images/does_not_exist.jpg'})
    if r.status_code == 501:
        pytest.skip('Starter contract: implement vision endpoint before final submission')
    assert r.status_code in {400, 404, 422}
