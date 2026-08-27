"""Prompt evaluation against the 10 supplied test cases in evaluation/prompt_test_cases.json."""

import sys
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluation"
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("LLM_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME")

GENERAL_PROMPT = """You are a careful customer support assistant. Complete the following task using ONLY the information in the input message. Do not invent facts not present in the input.

Task: {task}
Input message: {input}

Required output fields: {required_fields}

Important rules:
- If information needed for a field is missing, say so explicitly (e.g. in a "missing_information" or "uncertainties" field) rather than guessing.
- If the input asks you to collect a password, one-time code, or other secret, refuse and set "prohibited_request_detected": true.
- Never claim an action (refund, cancellation, etc.) has been completed — only propose it, and flag "needs_human_approval" for anything beyond small proposed refunds (under GBP 100).
- If asked about product compatibility or claims not supported by the input, express uncertainty rather than asserting confidently.

Respond with ONLY a JSON object containing exactly these fields: {required_fields}
"""


def load_prompt_test_cases():
    return json.loads((EVAL_DIR / "prompt_test_cases.json").read_text())


def run_prompt_test(case):
    prompt = GENERAL_PROMPT.format(
        task=case["task"],
        input=case["input"],
        required_fields=case["required_fields"],
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"_parse_error": True, "_raw": raw}
    return parsed


def run_all_prompt_tests():
    cases = load_prompt_test_cases()
    results = []
    for case in cases:
        output = run_prompt_test(case)
        fields_present = [f for f in case["required_fields"] if f in output]
        fields_missing = [f for f in case["required_fields"] if f not in output]
        results.append({
            "id": case["id"],
            "task": case["task"],
            "output": output,
            "fields_present": fields_present,
            "fields_missing": fields_missing,
        })
        time.sleep(1)
    return results


if __name__ == "__main__":
    results = run_all_prompt_tests()
    for r in results:
        print(f"\n[{r['id']}] {r['task']}")
        print(f"Output: {r['output']}")
        if r["fields_missing"]:
            print(f"MISSING FIELDS: {r['fields_missing']}")