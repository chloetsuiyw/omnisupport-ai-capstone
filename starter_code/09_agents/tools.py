"""Safe local tool stubs for the OmniSupport agent.

Implement these against the supplied local CSV files in `data/agent_store/`.
Do not scan the million-row analytical dataset on every tool call and do not
perform any real external customer/order/payment action.
"""
from pathlib import Path
import csv
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_rag"))

AGENT_STORE = Path(__file__).resolve().parents[2] / "data" / "agent_store"
ORDERS_PATH = AGENT_STORE / "orders.csv"
CUSTOMERS_PATH = AGENT_STORE / "customers.csv"
RETURNS_PATH = AGENT_STORE / "returns.csv"


def _load_csv_as_dict(path, key_col):
    """Load a CSV once into a dict keyed by key_col, for fast repeated lookups."""
    result = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row[key_col]] = row
    return result


_ORDERS_CACHE = None
_CUSTOMERS_CACHE = None
_RETURNS_CACHE = None


def _orders():
    global _ORDERS_CACHE
    if _ORDERS_CACHE is None:
        _ORDERS_CACHE = _load_csv_as_dict(ORDERS_PATH, "order_id")
    return _ORDERS_CACHE


def _customers():
    global _CUSTOMERS_CACHE
    if _CUSTOMERS_CACHE is None:
        _CUSTOMERS_CACHE = _load_csv_as_dict(CUSTOMERS_PATH, "customer_id")
    return _CUSTOMERS_CACHE


def _returns():
    global _RETURNS_CACHE
    if _RETURNS_CACHE is None:
        _RETURNS_CACHE = _load_csv_as_dict(RETURNS_PATH, "order_id")
    return _RETURNS_CACHE


def lookup_order(order_id: str) -> dict:
    """Return one synthetic order or a controlled not-found result."""
    order = _orders().get(order_id)
    if order is None:
        return {"found": False, "status": "not_found", "order_id": order_id}
    return {"found": True, "status": "ok", **order}


def lookup_customer(customer_id: str) -> dict:
    """Return one synthetic customer or a controlled not-found result."""
    customer = _customers().get(customer_id)
    if customer is None:
        return {"found": False, "status": "not_found", "customer_id": customer_id}
    return {"found": True, "status": "ok", **customer}


def calculate_refund(order_id: str, condition: str) -> dict:
    """Return an advisory refund proposal, including whether human approval is required."""
    order = lookup_order(order_id)
    if not order["found"]:
        return {"status": "not_found", "proposed_refund": None, "requires_approval": None}

    return_record = _returns().get(order_id)
    if return_record is None:
        return {"status": "no_return_record", "proposed_refund": None, "requires_approval": None}

    proposed_refund = float(return_record.get("proposed_refund", 0))
    requires_approval = proposed_refund > FRONTLINE_REFUND_LIMIT

    return {
        "status": "ok",
        "order_id": order_id,
        "proposed_refund": proposed_refund,
        "requires_approval": requires_approval,
        "reason": (
            f"Proposed refund £{proposed_refund:.2f} exceeds the £{FRONTLINE_REFUND_LIMIT:.0f} frontline approval limit."
            if requires_approval else
            f"Proposed refund £{proposed_refund:.2f} is within frontline approval authority."
        ),
    }

_RAG_INDEX = None

def search_policy(query: str) -> dict:
    """Search the supplied local knowledge base and return source-aware evidence."""
    global _RAG_INDEX
    import rag_pipeline

    if _RAG_INDEX is None:
        _RAG_INDEX = rag_pipeline.build_index()

    result = rag_pipeline.answer_question(query, _RAG_INDEX, top_k=3)
    return {
        "status": "ok",
        "answer": result["answer"],
        "sources": result["retrieved_sources"],
        "retrieval_scores": result["retrieval_scores"],
    }

def check_return_eligibility(order_id: str, condition: str) -> dict:
    """Return eligible/ineligible/manual-review plus a reason based on local data/policy."""
    order = lookup_order(order_id)
    if not order["found"]:
        return {"status": "not_found", "eligible": None, "reason": f"Order {order_id} not found in local store."}

    return_record = _returns().get(order_id)
    if return_record is None:
        return {"status": "no_return_record", "eligible": None, "reason": f"No return record exists for order {order_id}."}

    eligible_flag = return_record.get("return_eligible") == "1"
    stored_condition = return_record.get("item_condition")

    if condition.lower().strip() != stored_condition.lower().strip():
        return {
            "status": "manual_review",
            "eligible": None,
            "reason": (
                f"Reported condition '{condition}' does not match recorded condition "
                f"'{stored_condition}' for this order; flagging for manual review rather than guessing."
            ),
        }

    return {
        "status": "ok",
        "eligible": eligible_flag,
        "reason": f"Return eligibility recorded as {'eligible' if eligible_flag else 'not eligible'} for condition '{stored_condition}'.",
        "days_since_delivery": order.get("days_since_delivery"),
    }


FRONTLINE_REFUND_LIMIT = 100.0


if __name__ == "__main__":
    print(lookup_order("ORD00001001"))
    print(lookup_order("ORD99999999"))
    print(lookup_customer("CUST0040150"))

    print("\n--- check_return_eligibility ---")
    print(check_return_eligibility("ORD00001004", "damaged_item"))

    print("\n--- calculate_refund ---")
    print(calculate_refund("ORD00001002", "damaged"))
    print(calculate_refund("ORD00001003", "damaged"))

    print("\n--- search_policy ---")
    print(check_return_eligibility("ORD00001004", "damaged_item"))
    print(calculate_refund("ORD00001002", "damaged"))
    print(calculate_refund("ORD00001003", "damaged"))

    print("\n--- search_policy ---")
    print(search_policy("What happens if my parcel is lost?"))