"""Agent starter.
TODO: define tool schemas, controlled loop, tool selection, error handling, human-approval gates, and traceable outputs.
"""
import sys
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_rag"))

import tools

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("LLM_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME")

ROUTING_PROMPT = """You are a support agent's intent router. Given a customer request, decide what action is needed.

Respond with ONLY a JSON object with these fields:
- intent: one of [check_order_status, calculate_refund, check_return_eligibility, search_policy, account_change, cancel_order, safety_issue, unclear]
- order_id: extract the order ID (format ORD########) if mentioned, else null
- requested_amount: extract a specific GBP amount the customer is demanding, if any, else null
- condition: extract item condition mentioned (e.g. "damaged", "packaging damage"), else null

Customer request: {request}

Respond with ONLY the JSON object."""


def route_intent(user_request: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": ROUTING_PROMPT.format(request=user_request)}],
        temperature=0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def run_agent(user_request: str) -> dict:
    routing = route_intent(user_request)
    intent = routing.get("intent")
    order_id = routing.get("order_id")
    requested_amount = routing.get("requested_amount")
    condition = routing.get("condition") or "unspecified"

    trace = {"user_request": user_request, "routing": routing, "tool_calls": []}

    # Sensitive intents that always require human approval, no tool call needed
    if intent == "account_change":
        trace.update({"status": "needs_human_approval", "message": "Account/identity changes require human handling and verification."})
        return trace

    if intent == "safety_issue":
        result = tools.search_policy(user_request)
        trace["tool_calls"].append({"tool": "search_policy", "result": result})
        trace.update({"status": "needs_human_approval", "message": "Safety issue detected. Relevant policy retrieved; escalating to a human immediately."})
        return trace

    if intent == "cancel_order":
        if not order_id:
            trace.update({"status": "needs_information", "message": "I need the order ID to look into a cancellation."})
            return trace
        order = tools.lookup_order(order_id)
        trace["tool_calls"].append({"tool": "lookup_order", "result": order})
        if not order["found"]:
            trace.update({"status": "not_found", "message": f"Order {order_id} was not found."})
            return trace
        eligibility = tools.check_return_eligibility(order_id, condition)
        trace["tool_calls"].append({"tool": "check_return_eligibility", "result": eligibility})
        trace.update({
            "status": "needs_human_approval",
            "message": f"Order {order_id} has dispatch_state='{order.get('dispatch_state')}'. Cancellation of a dispatched order cannot be confirmed automatically; routing to a human for approval.",
        })
        return trace

    if intent == "check_order_status":
        if not order_id:
            trace.update({"status": "needs_information", "message": "Could you provide the order ID so I can check its status?"})
            return trace
        order = tools.lookup_order(order_id)
        trace["tool_calls"].append({"tool": "lookup_order", "result": order})
        if not order["found"]:
            trace.update({"status": "not_found", "message": f"Order {order_id} was not found in our records."})
            return trace
        trace.update({"status": "ok", "message": f"Order {order_id}: {order.get('delivery_status')}, dispatched {order.get('days_since_delivery')} days ago."})
        return trace

    if intent == "calculate_refund":
        if not order_id:
            trace.update({"status": "needs_information", "message": "Could you provide the order ID so I can calculate a refund?"})
            return trace
        order = tools.lookup_order(order_id)
        trace["tool_calls"].append({"tool": "lookup_order", "result": order})
        if not order["found"]:
            trace.update({"status": "not_found", "message": f"Order {order_id} was not found."})
            return trace
        refund = tools.calculate_refund(order_id, condition)
        trace["tool_calls"].append({"tool": "calculate_refund", "result": refund})

        requires_approval = refund.get("requires_approval", False)
        if requested_amount is not None and float(requested_amount) > tools.FRONTLINE_REFUND_LIMIT:
            requires_approval = True

        trace.update({
            "status": "needs_human_approval" if requires_approval else "ok",
            "message": (
                f"Proposed refund for {order_id}: £{refund.get('proposed_refund')}. "
                + ("This exceeds frontline authority and requires human approval." if requires_approval
                   else "This is within frontline approval authority.")
            ),
        })
        return trace

    if intent == "check_return_eligibility":
        if not order_id:
            trace.update({"status": "needs_information", "message": "I need the order ID before I can check return eligibility."})
            return trace
        eligibility = tools.check_return_eligibility(order_id, condition)
        trace["tool_calls"].append({"tool": "check_return_eligibility", "result": eligibility})
        trace.update({"status": eligibility.get("status", "ok"), "message": eligibility.get("reason")})
        return trace

    if intent == "search_policy":
        result = tools.search_policy(user_request)
        trace["tool_calls"].append({"tool": "search_policy", "result": result})
        trace.update({"status": "ok", "message": result.get("answer", "No policy answer available.")})
        return trace

    trace.update({"status": "needs_information", "message": "I'm not sure what you need help with — could you clarify?"})
    return trace


if __name__ == "__main__":
    import json as json_module

    test_cases = json_module.loads(
        (Path(__file__).resolve().parents[2] / "evaluation" / "agent_test_cases.json").read_text()
    )

    for case in test_cases:
        print(f"\n=== {case['id']}: {case['request']} ===")
        result = run_agent(case["request"])
        print(f"Status: {result['status']}")
        print(f"Message: {result.get('message')}")
        print(f"Tools called: {[t['tool'] for t in result['tool_calls']]}")
        print(f"Expected tools: {case['expected_tools']}")
        print(f"Expected human approval: {case['human_approval_expected']}, Got: {result['status'] == 'needs_human_approval'}")