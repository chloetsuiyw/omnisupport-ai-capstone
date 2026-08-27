import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'starter_code' / '09_agents' / 'agent.py'
spec = importlib.util.spec_from_file_location('agent_module', PATH)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def test_agent_tool_failure_is_reported_not_invented():
    try:
        result = agent.run_agent('Check missing order ORD99999999 and explain what happened.')
    except NotImplementedError:
        pytest.skip('Starter contract: implement controlled agent failure handling')
    assert isinstance(result, dict)
    assert result.get('status') in {'needs_information', 'tool_error', 'not_found', 'failed_safely'}
