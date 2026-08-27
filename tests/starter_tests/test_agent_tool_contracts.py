import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'starter_code' / '09_agents' / 'tools.py'
spec = importlib.util.spec_from_file_location('agent_tools', PATH)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


def _call_or_skip(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError:
        pytest.skip('Starter contract: implement this tool before final submission')


def test_missing_order_returns_controlled_not_found():
    result = _call_or_skip(tools.lookup_order, 'ORD99999999')
    assert isinstance(result, dict)
    assert result.get('found') is False or result.get('status') in {'not_found', 'missing'}


def test_refund_over_frontline_limit_requires_approval():
    result = _call_or_skip(tools.calculate_refund, 'ORD00001003', 'damaged')
    assert isinstance(result, dict)
    if float(result.get('proposed_refund', result.get('amount', 0))) > 100:
        assert result.get('requires_approval') is True
