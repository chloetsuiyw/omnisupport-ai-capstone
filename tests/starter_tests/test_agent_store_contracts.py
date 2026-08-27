import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_known_order_ids_exist():
    with (ROOT / 'data' / 'agent_store' / 'orders.csv').open(newline='', encoding='utf-8') as f:
        ids = {row['order_id'] for row in csv.DictReader(f)}
    assert {'ORD00001001', 'ORD00001002', 'ORD00001003', 'ORD00001007'} <= ids


def test_missing_order_case_requires_clarification_before_tools():
    cases = json.loads((ROOT / 'evaluation' / 'agent_test_cases.json').read_text())
    case = next(x for x in cases if x['id'] == 'A04')
    assert case['expected_tools'] == []
    assert 'missing order ID' in case['expected_outcome']
