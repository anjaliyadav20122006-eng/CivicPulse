# CivicPulse — AI Agent for Municipal Grievance-to-Policy Insight

**1M1B AI for Sustainability Virtual Internship (July–Sep 2026)**
*In Collaboration with IBM SkillsBuild & AICTE*

---

## Problem Statement

> How might we use AI to analyze unstructured citizen grievance data so
> that municipal authorities can identify and act on recurring civic
> infrastructure failures faster and more equitably?

Municipal grievance portals receive thousands of unstructured text
complaints daily — waterlogging, garbage collection, broken streetlights,
potholes, etc. Officers manually skim these, so recurring, high-impact
patterns often go unnoticed until they become crises. CivicPulse turns
raw complaint volume into prioritized, policy-grounded, actionable
insight.

## SDG Alignment

- **Primary:** SDG 11 — Sustainable Cities and Communities
- **Secondary:** SDG 16 (responsive governance), SDG 6 (where complaints involve water/sanitation)

## AI Solution Overview

CivicPulse is an end-to-end pipeline:

```
Raw Grievances → Entity Extraction → Clustering → RAG Retrieval → Agentic Routing → Policy Brief
```

| Stage | What it does | Tech |
|---|---|---|
| **Entity Extraction** | Extracts ward, issue type, urgency, duration from raw complaint text | Rule-based regex NLP (transparent, auditable) |
| **Privacy Redaction** | Strips phone numbers, emails, self-identified names before any processing | Regex-based PII detection |
| **Clustering** | Groups complaints into (ward, issue) hotspots + semantic text clusters; computes a transparent priority score (frequency + urgency + recency) | scikit-learn (TF-IDF, KMeans) |
| **RAG Retrieval** | Retrieves relevant sections from municipal SOP/policy documents to ground recommendations in real policy, not generic LLM guesses | TF-IDF + cosine similarity |
| **Agentic Routing** | Autonomously decides which department/action route a cluster should take (Health Escalation / Safety Corridor / Chief Engineer / Route Redesign / Routine Maintenance) based on evidence — a genuine branching decision, not a template fill | Custom rule-based decision agent |
| **Policy Brief Generation** | Combines all of the above into a structured, human-readable brief for municipal officers | Python |
| **Frontend** | Interactive dashboard: browse hotspots, view trend charts, submit a live complaint and watch the full pipeline run | Streamlit |

## Target Users

Municipal ward officers (primary), urban planners (secondary), citizens (indirect beneficiaries via faster resolution).

## Responsible AI Considerations

- **Fairness:** The synthetic dataset deliberately models uneven ward-level reporting (some wards under-report due to lower digital access). The dashboard and SOP guidance both flag that low complaint volume ≠ low actual need, and recommend periodic physical audits for such wards.
- **Transparency:** Every extraction, clustering, retrieval, and routing decision is rule-based and traceable — no black-box scoring. Retrieval relevance scores and routing reasoning are surfaced directly in the output.
- **Ethics:** The system supports, not replaces, human officer decision-making. It never auto-closes or auto-rejects a complaint.
- **Privacy:** A dedicated redaction module strips phone numbers, emails, and self-identified names from complaint text before it is stored, clustered, or shown in any policy brief (see `src/privacy_redaction.py`).

## Expected Impact

Faster identification of high-frequency civic issues, more equitable attention across wards, and policy-grounded, department-routed recommendations that reduce the manual triage burden on municipal officers.

## Project Structure

```
CivicPulse/
├── src/
│   ├── generate_dataset.py       # Synthetic grievance dataset generator
│   ├── entity_extraction.py      # Ward/issue/urgency extraction
│   ├── privacy_redaction.py      # PII detection & redaction
│   ├── clustering.py             # Hotspot detection & priority scoring
│   ├── rag_pipeline.py           # SOP knowledge base + retrieval
│   ├── agent.py                  # Agentic routing + policy brief generation
│   └── streamlit_app.py          # Interactive dashboard
└── data/
    ├── sop_drainage.txt          # Sample municipal SOP documents
    ├── sop_waste.txt
    ├── sop_infra.txt
    └── *.json / *.csv            # Generated data (created by running the scripts)
```

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn streamlit

# 2. Run the pipeline in order (from the src/ folder)
python generate_dataset.py
python entity_extraction.py
python privacy_redaction.py
python clustering.py
python rag_pipeline.py
python agent.py

# 3. Launch the interactive dashboard
streamlit run streamlit_app.py
```


---

*Built as part of the 1M1B AI for Sustainability Virtual Internship, Sep 2026.*
