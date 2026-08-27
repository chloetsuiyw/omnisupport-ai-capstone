"""Streamlit stakeholder demo for OmniSupport AI."""

import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="OmniSupport AI", layout="centered")
st.title("OmniSupport AI — Stakeholder Demo")

tab1, tab2 = st.tabs(["Escalation Risk Predictor", "Policy Assistant"])

with tab1:
    st.subheader("Predict escalation risk for a new ticket")

    col1, col2 = st.columns(2)
    with col1:
        issue_category = st.selectbox("Issue category", [
            "lost_parcel", "delivery_late", "damaged_item", "wrong_item",
            "return_request", "payment_issue", "account_access",
            "product_question", "warranty", "refund_status"
        ])
        priority = st.selectbox("Priority", ["low", "medium", "high", "urgent"])
        customer_region = st.selectbox("Customer region", [
            "London", "South East", "North West", "West Midlands", "Yorkshire",
            "South West", "Scotland", "East Midlands", "Wales", "Northern Ireland", "Unknown"
        ])
        support_channel = st.selectbox("Support channel", ["web_chat", "email", "mobile_app", "phone"])
    with col2:
        order_value = st.number_input("Order value (£)", min_value=0.0, value=100.0)
        delivery_delay = st.number_input("Delivery delay (days)", min_value=0, value=0)
        previous_tickets = st.number_input("Previous ticket count", min_value=0, value=0)
        tenure = st.number_input("Customer tenure (months)", min_value=0, value=12)

    if st.button("Predict Escalation Risk"):
        payload = {
            "support_channel": support_channel,
            "customer_region": customer_region,
            "customer_age_band": "35-44",
            "preferred_language": "English",
            "customer_tenure_months": tenure,
            "product_category": "electronics",
            "order_value_capped": order_value,
            "order_value_was_capped": 0,
            "delivery_delay_days": delivery_delay,
            "previous_ticket_count": previous_tickets,
            "issue_category": issue_category,
            "priority": priority,
            "attachment_available": 0,
            "accessibility_support_flag": 0,
            "ticket_hour": 12,
            "ticket_day_of_week": 2,
            "ticket_month": 8,
            "issue_title_length": 20,
            "issue_description_length": 100,
        }
        try:
            response = requests.post(f"{API_BASE}/predict/escalation", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.metric("Escalation Probability", f"{result['escalation_probability']:.1%}")
            if result["requires_human_review"]:
                st.warning("⚠️ This ticket is flagged for human review (high escalation risk).")
            else:
                st.success("✅ Low escalation risk — standard handling recommended.")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API. Is it running? ({e})")

with tab2:
    st.subheader("Ask the policy assistant")
    question = st.text_input("Your question", placeholder="e.g. How many days do I have to return an item?")

    if st.button("Ask") and question:
        try:
            response = requests.post(f"{API_BASE}/ask/policy", json={"question": question}, timeout=15)
            response.raise_for_status()
            result = response.json()
            st.write("**Answer:**", result["answer"])
            st.caption(f"Sources: {', '.join(result['sources'])} | Top retrieval score: {result['top_retrieval_score']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API. Is it running? ({e})")