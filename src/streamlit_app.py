"""
CivicPulse - Streamlit Demo Frontend
========================================
A lightweight UI wrapper around the existing backend pipeline
(clustering.py -> rag_pipeline.py -> agent.py). No new logic is
introduced here -- this purely visualizes outputs that are already
computed by the backend, which keeps the core AI pipeline testable
independent of the UI (good practice, and easy to demo on video).

Run with:  streamlit run streamlit_app.py
"""

import json
import streamlit as st
import pandas as pd
import os

from rag_pipeline import build_knowledge_base, RAGRetriever, build_query_from_cluster
from agent import generate_policy_brief, priority_label

st.set_page_config(page_title="CivicPulse", page_icon="🏛️", layout="wide")

# Resolve data directory relative to THIS file's location, not the
# current working directory -- this makes the app work identically
# whether run locally (streamlit run streamlit_app.py from inside src/)
# or on Streamlit Cloud (which runs from the repo root).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")


@st.cache_resource
def load_pipeline():
    with open(os.path.join(DATA_DIR, "clusters_structured.json")) as f:
        clusters = json.load(f)
    chunks = build_knowledge_base()
    retriever = RAGRetriever(chunks)
    return clusters, retriever


clusters, retriever = load_pipeline()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("🏛️ CivicPulse")
st.sidebar.caption("AI Agent for Municipal Grievance-to-Policy Insight")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**SDG 11**: Sustainable Cities and Communities\n\n"
    "Pipeline: Entity Extraction → Clustering → RAG Retrieval → "
    "Grounded Policy Brief"
)
st.sidebar.markdown("---")

hotspot_only = st.sidebar.checkbox("Show recurring hotspots only", value=True)
filtered = [c for c in clusters if c["is_recurring_hotspot"]] if hotspot_only else clusters
filtered = sorted(filtered, key=lambda c: c["priority_score"], reverse=True)

options = [f"{c['ward']} — {c['issue_type'].replace('_',' ').title()} "
           f"({c['complaint_count']} complaints, priority {c['priority_score']})"
           for c in filtered]

if not options:
    st.warning("No clusters match this filter.")
    st.stop()

selected_idx = st.sidebar.selectbox("Select a ward/issue cluster", range(len(options)),
                                     format_func=lambda i: options[i])
selected_cluster = filtered[selected_idx]

# ---------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------
st.title("CivicPulse — Municipal Grievance Insight")
st.caption("Turning scattered citizen complaints into prioritized, policy-grounded action briefs.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ward", selected_cluster["ward"])
col2.metric("Issue Type", selected_cluster["issue_type"].replace("_", " ").title())
col3.metric("Total Complaints", selected_cluster["complaint_count"])
col4.metric("Priority", priority_label(selected_cluster["priority_score"]),
            f"score {selected_cluster['priority_score']}")

st.markdown("---")

# Retrieve + generate brief for the selected cluster
query = build_query_from_cluster(selected_cluster)
retrieved = retriever.retrieve(query, k=3)
brief = generate_policy_brief(selected_cluster, retrieved)

left, right = st.columns([3, 2])

with left:
    st.subheader("📋 Problem Summary")
    st.write(brief["problem_summary"])

    st.subheader("✅ Grounded Recommendation")
    st.info(brief["grounded_recommendation"])
    st.caption(brief["grounding_note"])

    st.subheader("🗣️ Evidence — Sample Citizen Complaints")
    for t in brief["sample_complaint_evidence"]:
        st.markdown(f"- _{t}_")

with right:
    st.subheader("🔍 Retrieved SOP Context")
    for r in retrieved:
        with st.expander(f"{r['source_doc']} — {r['header']}  (score: {r['relevance_score']})"):
            st.write(r["text"])

st.markdown("---")

# ---------------------------------------------------------------------
# Trend chart: complaint volume over time for the selected cluster
# ---------------------------------------------------------------------
st.subheader("📈 Complaint Trend Over Time")
st.caption(f"Weekly complaint volume for {selected_cluster['ward']} — "
           f"{selected_cluster['issue_type'].replace('_',' ').title()}")


@st.cache_data
def load_enriched():
    with open(os.path.join(DATA_DIR, "grievances_enriched.json")) as f:
        return pd.DataFrame(json.load(f))


enriched_df = load_enriched()
trend_df = enriched_df[
    (enriched_df["extracted_ward"] == selected_cluster["ward"]) &
    (enriched_df["extracted_issue"] == selected_cluster["issue_type"])
].copy()

if trend_df.empty:
    st.info("No time-series data available for this cluster.")
else:
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    weekly = trend_df.set_index("date").resample("W").size().rename("complaints").reset_index()
    st.line_chart(weekly.set_index("date")["complaints"])
    st.caption(
        f"Total: {len(trend_df)} complaints from "
        f"{trend_df['date'].min().strftime('%d %b')} to {trend_df['date'].max().strftime('%d %b %Y')}. "
        f"A rising or sustained trend (rather than a single spike) is what distinguishes a "
        f"genuine recurring infrastructure failure from a one-off event."
    )

st.markdown("---")
st.subheader("📝 Try It Yourself — Submit a New Grievance")
st.caption(
    "Type a complaint like a citizen would. The pipeline will extract "
    "entities, check if it matches a known hotspot, retrieve relevant "
    "SOP guidance, and generate a policy brief — live."
)

new_complaint = st.text_area(
    "Citizen complaint text",
    placeholder="e.g. Heavy waterlogging near Gandhi Chowk in Ward 12 since 5 days, water entering our shop.",
    height=80,
)

if st.button("🔎 Analyze this complaint"):
    if not new_complaint.strip():
        st.warning("Please type a complaint first.")
    else:
        from entity_extraction import extract_ward, extract_issue, extract_urgency, extract_duration

        ward = extract_ward(new_complaint)
        issue = extract_issue(new_complaint)
        urgency = extract_urgency(new_complaint)
        duration = extract_duration(new_complaint)

        st.markdown("#### 1️⃣ Extracted Entities")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Ward", ward)
        e2.metric("Issue Type", issue.replace("_", " ").title())
        e3.metric("Urgency", urgency.upper())
        e4.metric("Duration mentioned", duration)

        # Check whether this matches an existing recurring hotspot
        matching = [c for c in clusters if c["ward"] == ward and c["issue_type"] == issue]

        if matching:
            match = matching[0]
            st.markdown("#### 2️⃣ Hotspot Match")
            st.success(
                f"This matches an **existing pattern**: {ward} already has "
                f"{match['complaint_count']} logged {issue.replace('_',' ')} complaints "
                f"(priority score {match['priority_score']}, "
                f"{'flagged as a recurring hotspot' if match['is_recurring_hotspot'] else 'not yet a hotspot'})."
                f" This new complaint would be the {match['complaint_count']+1}th."
            )
            adhoc_cluster = dict(match)
            adhoc_cluster["complaint_count"] = match["complaint_count"] + 1
            adhoc_cluster["sample_texts"] = [new_complaint] + match.get("sample_texts", [])[:1]
        else:
            st.markdown("#### 2️⃣ Hotspot Match")
            st.info(f"No existing pattern found for {ward} + {issue.replace('_',' ')} — treating as a new, isolated report.")
            adhoc_cluster = {
                "ward": ward, "issue_type": issue, "complaint_count": 1,
                "priority_score": 0.15, "is_recurring_hotspot": False,
                "most_recent_complaint": "today", "sample_texts": [new_complaint],
            }

        st.markdown("#### 3️⃣ Retrieved SOP Context (RAG)")
        live_query = f"{issue.replace('_',' ')} {issue.replace('_',' ')} {new_complaint}"
        live_retrieved = retriever.retrieve(live_query, k=3)
        for r in live_retrieved:
            with st.expander(f"{r['source_doc']} — {r['header']}  (score: {r['relevance_score']})"):
                st.write(r["text"])

        st.markdown("#### 4️⃣ Generated Policy Brief")
        live_brief = generate_policy_brief(adhoc_cluster, live_retrieved)
        st.write(live_brief["problem_summary"])
        st.success(live_brief["grounded_recommendation"])
        st.caption(live_brief["grounding_note"])

st.markdown("---")
st.subheader("🗺️ All Recurring Hotspots — City Overview")
hotspots = [c for c in clusters if c["is_recurring_hotspot"]]
df = pd.DataFrame(hotspots)[["ward", "issue_type", "complaint_count",
                              "priority_score", "days_since_last_complaint"]]
df.columns = ["Ward", "Issue Type", "Complaints", "Priority Score", "Days Since Last Complaint"]
df = df.sort_values("Priority Score", ascending=False).reset_index(drop=True)
st.dataframe(df, use_container_width=True)

st.caption(
    "Responsible AI note: ward-level complaint volume can reflect reporting "
    "access as much as actual issue severity. Wards with historically low "
    "complaint counts should also receive periodic physical audits, per "
    "SOP equity guidance -- see sop_waste.txt, Section 5."
)
