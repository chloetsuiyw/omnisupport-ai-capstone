"""New tests for the escalation prediction endpoint.

The starter test suite never covered /predict/escalation, since this
endpoint does not exist on the stub API (app/api/main.py). These tests
were added to close that gap for the real implementation.
"""
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

VALID_PAYLOAD = {
    "support_channel": "web_chat",
    "customer_region": "London",
    "customer_age_band": "25-34",
    "preferred_language": "English",
    "customer_tenure_months": 12,
    "product_category": "electronics",
    "order_value_capped": 45.0,
    "order_value_was_capped": 0,
    "delivery_delay_days": 0,
    "previous_ticket_count": 1,
    "issue_category": "wrong_item",
    "priority": "low",
    "attachment_available": 0,
    "accessibility_support_flag": 0,
    "ticket_hour": 14,
    "ticket_day_of_week": 2,
    "ticket_month": 8,
    "issue_title_length": 20,
    "issue_description_length": 60,
}


def test_valid_escalation_request_returns_probability_in_range():
    r = client.post('/predict/escalation', json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 'escalation_probability' in body
    assert 0.0 <= body['escalation_probability'] <= 1.0
    assert isinstance(body['predicted_escalation'], bool)
    assert isinstance(body['requires_human_review'], bool)


def test_missing_required_field_rejected():
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != 'issue_category'}
    r = client.post('/predict/escalation', json=incomplete_payload)
    assert r.status_code == 422


def test_high_risk_profile_scores_higher_than_low_risk_profile():
    """Regression-style sanity check: a high-priority, high-previous-ticket-count
    profile should score at least as high as a low-priority, first-time profile,
    consistent with previous_ticket_count and priority being the model's
    strongest predictors (see 02_classification_notes.md)."""
    low_risk = dict(VALID_PAYLOAD, priority="low", previous_ticket_count=0)
    high_risk = dict(VALID_PAYLOAD, priority="urgent", previous_ticket_count=8, issue_category="lost_parcel")

    r_low = client.post('/predict/escalation', json=low_risk)
    r_high = client.post('/predict/escalation', json=high_risk)

    prob_low = r_low.json()['escalation_probability']
    prob_high = r_high.json()['escalation_probability']

    assert prob_high >= prob_low