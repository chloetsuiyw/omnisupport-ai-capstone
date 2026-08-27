import importlib.util
from pathlib import Path
import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'starter_code' / '07_llm' / 'structured_output.py'
spec = importlib.util.spec_from_file_location('structured_output', PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_malformed_structured_output_is_rejected():
    with pytest.raises(ValidationError):
        mod.TicketExtraction.model_validate({"issue_category": "delivery_late"})
