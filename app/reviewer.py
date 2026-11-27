"""Streamlit reviewer UI for a locally running ProductMatch AI FastAPI server.

Run with streamlit run app/reviewer.py. Decisions require a reviewer token.
No retraining or file uploads occur from this UI.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("PRODUCTMATCH_API_URL", "http://127.0.0.1:8000").rstrip("/")


def call_api(method: str, path: str, *, payload: dict | None = None) -> dict | list:
    headers = {"X-API-Key": st.session_state.api_key}
    if path.startswith("/v1/reviews"):
        headers["X-Review-Token"] = st.session_state.review_token
    response = requests.request(
        method, API_URL + path, headers=headers, json=payload, timeout=20
    )
    if not response.ok:
        raise ValueError(f"API {response.status_code}: {response.text[:350]}")
    return response.json()


st.set_page_config(page_title="ProductMatch Reviewer", layout="wide")
st.title("ProductMatch AI · Human Review")
st.caption("Private development UI. Reviews are stored in SQLite; models are never retrained.")
with st.sidebar:
    st.text_input("API endpoint", value=API_URL, disabled=True)
    st.text_input("API key", type="password", key="api_key")
    st.text_input("Reviewer token", type="password", key="review_token")
    st.text_input("Reviewer ID", value="reviewer", key="reviewer_id")

tabs = st.tabs(["Search & review", "Pairwise score", "Review history"])

with tabs[0]:
    title = st.text_input("Query product title", "Acme Shampoo 250 ml")
    brand = st.text_input("Brand", "Acme")
    method = st.selectbox("Ranking", ["bm25", "reranked"])
    top_k = st.slider("Candidates", min_value=1, max_value=20, value=5)
    if st.button("Find candidates", type="primary"):
        try:
            result = call_api(
                "POST",
                "/v1/search",
                payload={
                    "query": {"title": title, "brand": brand},
                    "method": method,
                    "top_k": top_k,
                },
            )
            st.session_state.candidates = result["candidates"]
            st.session_state.query = {"title": title, "brand": brand}
        except (ValueError, requests.RequestException) as exc:
            st.error(str(exc))
    candidates = st.session_state.get("candidates", [])
    if candidates:
        st.caption(f"Showing {len(candidates)} candidates")
        for i, item in enumerate(candidates):
            with st.expander(
                f"{i + 1}. {item['title']} — {item['score_type']}={item['score']:.3f}"
            ):
                st.write("Brand:", item["brand"], " · Offer ID:", item["offer_id"])
                decision = st.radio(
                    "Decision", ["match", "nonmatch", "unsure"],
                    horizontal=True, key=f"decision_{i}"
                )
                notes = st.text_area("Review notes", key=f"notes_{i}")
                if st.button("Save review", key=f"save_{i}"):
                    try:
                        record = call_api(
                            "POST", "/v1/reviews",
                            payload={
                                "query": st.session_state.query,
                                "candidate": {
                                    "offer_id": item["offer_id"],
                                    "title": item["title"],
                                    "brand": item["brand"],
                                },
                                "decision": decision,
                                "reviewer": st.session_state.reviewer_id,
                                "notes": notes,
                            },
                        )
                        st.success(f"Recorded review {record['review_id']}")
                    except (ValueError, requests.RequestException) as exc:
                        st.error(str(exc))

with tabs[1]:
    left_title = st.text_input("Left title", "Acme Shampoo 250 ml")
    right_title = st.text_input("Right title", "Acme Shampoo 500 ml")
    if st.button("Score product pair"):
        try:
            prediction = call_api(
                "POST", "/v1/match",
                payload={"left": {"title": left_title}, "right": {"title": right_title}},
            )
            st.metric("Match probability", f"{prediction['match_probability']:.3f}")
            st.write("Decision:", prediction["is_match"])
            st.caption(prediction["warning"])
        except (ValueError, requests.RequestException) as exc:
            st.error(str(exc))

with tabs[2]:
    if st.button("Load recent reviews"):
        try:
            records = call_api("GET", "/v1/reviews?limit=100")
            st.dataframe(
                [
                    {
                        "created_at": row["created_at"],
                        "reviewer": row["reviewer"],
                        "decision": row["decision"],
                        "query": row["query"]["title"],
                        "candidate": row["candidate"]["title"],
                        "notes": row["notes"],
                    }
                    for row in records
                ],
                use_container_width=True,
            )
        except (ValueError, requests.RequestException) as exc:
            st.error(str(exc))
