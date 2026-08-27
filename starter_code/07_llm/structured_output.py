from typing import Optional, List
from pathlib import Path
import os
import json
import time
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("LLM_API_KEY"),
)
MODEL_NAME = os.getenv("MODEL_NAME")


class TicketExtraction(BaseModel):
    issue_category: str
    priority: str
    order_id: Optional[str] = None
    customer_intent: str
    needs_human_review: bool
    missing_information: List[str] = []


EXTRACTION_PROMPT = """You are a support ticket triage assistant. Given a customer message, extract structured information.

Return ONLY a JSON object with these exact fields:
- issue_category: one of [damaged_item, delivery_late, refund_status, wrong_item, return_request, payment_issue, account_access, product_question, warranty, lost_parcel]
- priority: one of [low, medium, high, urgent]
- order_id: the order ID if mentioned, otherwise null
- customer_intent: a short phrase describing what the customer wants
- needs_human_review: true if the message is ambiguous, emotionally charged, or involves a policy exception; otherwise false
- missing_information: a list of any information needed to resolve this ticket but not provided (e.g. "order_id")

Customer message: {message}

Respond with ONLY the JSON object, no other text."""

CLASSIFICATION_PROMPT = """Classify this customer support message into exactly one category.
Categories: damaged_item, delivery_late, refund_status, wrong_item, return_request, payment_issue, account_access, product_question, warranty, lost_parcel

Message: {message}

Respond with ONLY the category name, nothing else."""

SUMMARIZATION_PROMPT = """Summarize this customer support message in one short sentence, suitable for an agent dashboard.

Message: {message}

Respond with ONLY the summary sentence, nothing else."""


def call_with_retry(fn, *args, max_retries=5, base_delay=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def call_llm_for_structured_output(message: str) -> TicketExtraction:
    def _call():
        prompt = EXTRACTION_PROMPT.format(message=message)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw_text = response.choices[0].message.content.strip()
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw_text)
        return TicketExtraction(**data)
    return call_with_retry(_call)


def classify_ticket(message: str) -> str:
    def _call():
        prompt = CLASSIFICATION_PROMPT.format(message=message)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    return call_with_retry(_call)


def summarize_ticket(message: str) -> str:
    def _call():
        prompt = SUMMARIZATION_PROMPT.format(message=message)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    return call_with_retry(_call)


REGRESSION_TEST_CASES = [
    {
        "message": "My package never arrived and it's been 2 weeks",
        "expected_category": "lost_parcel",
    },
    {
        "message": "The screen on my new phone is cracked",
        "expected_category": "damaged_item",
    },
    {
        "message": "I can't log into my account, it says wrong password",
        "expected_category": "account_access",
    },
    {
        "message": "When will my order arrive? It says delayed",
        "expected_category": "delivery_late",
    },
    {
        "message": "I want to return this, it's not what I ordered",
        "expected_category": "wrong_item",
    },
]


def run_regression_tests():
    results = []
    for case in REGRESSION_TEST_CASES:
        predicted = classify_ticket(case["message"])
        passed = predicted == case["expected_category"]
        results.append({
            "message": case["message"][:50],
            "expected": case["expected_category"],
            "predicted": predicted,
            "passed": passed,
        })
        time.sleep(2)
    return results

def run_benchmark(messages, n_requests=20):
    results = []
    for i in range(n_requests):
        msg = messages[i % len(messages)]
        start = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(message=msg)}],
            temperature=0,
            max_tokens=300,
        )
        elapsed = time.time() - start
        usage = response.usage
        results.append({
            "request_num": i + 1,
            "latency_sec": round(elapsed, 3),
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        })
        time.sleep(1)
    return pd.DataFrame(results)

def estimate_cost(benchmark_df, input_price_per_million=0.30, output_price_per_million=2.50):
    total_input = benchmark_df["input_tokens"].sum()
    total_output = benchmark_df["output_tokens"].sum()
    cost = (total_input / 1_000_000 * input_price_per_million) + (total_output / 1_000_000 * output_price_per_million)
    return {
        "total_input_tokens": int(total_input),
        "total_output_tokens": int(total_output),
        "estimated_cost_usd": round(cost, 6),
    }

if __name__ == "__main__":
    test_message = "The box was crushed and the item inside is broken. Order ORD00001002. This is really frustrating, I need a refund ASAP."
    result = call_llm_for_structured_output(test_message)
    print(result)
    print("\nValidated Pydantic model:", result.model_dump())

    print("\n--- Regression Tests ---")
    test_results = run_regression_tests()
    for r in test_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['message']}... expected={r['expected']}, got={r['predicted']}")

    passed_count = sum(r["passed"] for r in test_results)
    print(f"\n{passed_count}/{len(test_results)} tests passed")

    print("\n--- LLM Benchmark (20 requests) ---")
    benchmark_messages = [case["message"] for case in REGRESSION_TEST_CASES] + [
        "I paid twice for the same order by mistake",
        "The warranty on my item expired but it's still broken",
        "Can I change my delivery address after ordering?",
        "My refund hasn't shown up in my account yet",
    ]
    benchmark_df = run_benchmark(benchmark_messages, n_requests=20)
    print(benchmark_df)
    print(f"\nMean latency: {benchmark_df['latency_sec'].mean():.3f}s")
    print(f"Total tokens used: {benchmark_df['total_tokens'].sum()}")
    print(f"Mean tokens per request: {benchmark_df['total_tokens'].mean():.1f}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(benchmark_df)

    cost_info = estimate_cost(benchmark_df)
    print("\n--- Cost Estimate ---")
    print(cost_info)