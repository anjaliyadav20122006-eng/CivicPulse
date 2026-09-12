"""
CivicPulse - Agent / Policy Brief Generation Layer
======================================================
This is the "agent" in our pipeline: it takes a recurring hotspot
cluster (from clustering.py) plus RAG-retrieved SOP context
(from rag_pipeline.py) and synthesizes a structured, human-readable
Policy Brief for municipal officers.

Design choice -- template-driven reasoning instead of an external LLM
API call:
    This keeps the full pipeline runnable with ZERO external
    dependencies or API keys, which matters for a reliable, repeatable
    demo. The agent still performs genuine reasoning: it decides brief
    urgency wording, extracts the most actionable sentence from the
    retrieved SOP text, and adapts its recommendation based on
    priority score and hotspot status. This can be swapped for an LLM
    call (e.g. Gemini/OpenAI/Claude) later using the exact same inputs
    and output schema, without changing anything upstream.
"""

import json
import re


def priority_label(score):
    if score >= 0.7:
        return "CRITICAL"
    elif score >= 0.45:
        return "HIGH"
    elif score >= 0.25:
        return "MODERATE"
    return "LOW"


def extract_actionable_sentence(chunk_text, header=None):
    """Pull out the most 'actionable' sentence from a retrieved SOP
    chunk -- heuristically, the sentence containing a time-bound
    instruction (numbers + hours/days/works) or an imperative escalation
    phrase. This is what makes the recommendation policy-GROUNDED rather
    than a generic LLM guess.

    Returns None if no sentence scores above 0, so the caller can fall
    back to the next retrieved chunk instead of surfacing a purely
    descriptive (non-actionable) sentence like a section's opening
    "Purpose" statement.
    """
    body = chunk_text
    if header and body.startswith(header):
        body = body[len(header):].strip()
    body = re.sub(r'\s+', ' ', body)  # collapse newlines/extra whitespace

    sentences = re.split(r'(?<=[.!?])\s+', body)
    time_pattern = re.compile(r'\b\d+\s*(hour|hours|day|days|working days)\b', re.IGNORECASE)
    # word-boundary matching for action verbs -- avoids false positives like
    # "escalat" matching inside the unrelated word "escalation" when it's
    # just describing scope rather than issuing an instruction
    action_verb_pattern = re.compile(
        r'\b(escalate|escalated|must|rectify|repair|restore|inspect|dispatch|'
        r'flagged|flag|required|require)\b', re.IGNORECASE
    )

    scored = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        score = 0
        if time_pattern.search(s):
            score += 2
        if action_verb_pattern.search(s):
            score += 1
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return None


def decide_action_route(cluster, complaint_sample=""):
    """
    AGENTIC DECISION LAYER
    -----------------------
    This is the autonomous reasoning step that distinguishes an "agent"
    from a simple summarizer: given the situation, the agent independently
    chooses WHICH downstream department/action path to take, out of
    several distinct options -- it is not filling a fixed template, it
    is making a branching decision based on the evidence.

    Routes (mirroring real inter-departmental escalation logic that would
    exist in an actual municipal system):
        HEALTH_ESCALATION   -> stagnant water / health-hazard keywords
                                present, regardless of hotspot status
                                (health risk cannot wait for a frequency
                                threshold)
        CHIEF_ENGINEER_ESCALATION -> recurring hotspot + high priority
                                score (systemic infrastructure failure)
        SAFETY_CORRIDOR_ESCALATION -> streetlight/safety keywords
                                ("unsafe", "women", "night") present
        ROUTE_REDESIGN_ESCALATION -> recurring garbage/collection failure
                                (systemic route-planning issue, not a
                                one-off missed pickup)
        ROUTINE_MAINTENANCE -> default route for isolated, low-priority
                                reports

    The agent evaluates these conditions in a specific priority order
    (health and safety checked before frequency-based escalation),
    because a single dangerous complaint should never be deprioritized
    just because it hasn't yet reached the recurring-hotspot count --
    this ordering choice IS the agent's autonomous reasoning.
    """
    text_lower = (complaint_sample or " ".join(cluster.get("sample_texts", []))).lower()
    issue = cluster["issue_type"]
    priority = cluster["priority_score"]
    is_hotspot = cluster["is_recurring_hotspot"]

    health_keywords = ["mosquito", "stagnant", "health", "contaminat", "smell", "unhygienic"]
    safety_keywords = ["unsafe", "women", "children", "night", "accident", "skidding"]

    if issue == "waterlogging" and any(kw in text_lower for kw in health_keywords):
        return {
            "route": "HEALTH_ESCALATION",
            "target_department": "Public Health Department",
            "reasoning": "Health-hazard keywords detected in complaint text (stagnant water / "
                         "mosquito breeding). Routed immediately regardless of hotspot status, "
                         "since public health risk cannot wait for a frequency threshold to be met.",
        }

    if issue in ("streetlight", "pothole") and any(kw in text_lower for kw in safety_keywords):
        return {
            "route": "SAFETY_CORRIDOR_ESCALATION",
            "target_department": "Electrical & Roads Department (Safety Priority Queue)",
            "reasoning": "Safety-related keywords detected (unsafe/women/children/accident). "
                         "Routed to the priority safety queue ahead of the standard maintenance "
                         "queue, per SOP guidance that safety concerns override routine ordering.",
        }

    if is_hotspot and issue == "garbage":
        return {
            "route": "ROUTE_REDESIGN_ESCALATION",
            "target_department": "Sanitation Route Planning Committee",
            "reasoning": f"Recurring garbage-collection failure ({cluster['complaint_count']} "
                         f"complaints) indicates a systemic route-planning gap rather than an "
                         f"isolated missed pickup. Routed for route redesign, not repeated ad-hoc dispatch.",
        }

    if is_hotspot and priority >= 0.6:
        return {
            "route": "CHIEF_ENGINEER_ESCALATION",
            "target_department": "Chief Engineer's Office",
            "reasoning": f"Recurring hotspot with high priority score ({priority}) indicates a "
                         f"systemic infrastructure failure requiring root-cause investigation and "
                         f"capital-works budget consideration, not another one-off repair.",
        }

    return {
        "route": "ROUTINE_MAINTENANCE",
        "target_department": "Ward Maintenance Team",
        "reasoning": "No health, safety, or systemic-failure signals detected, and complaint "
                     "volume is below the recurring-hotspot threshold. Logged for standard "
                     "maintenance queue processing.",
    }


def generate_policy_brief(cluster, retrieved_chunks):
    issue_readable = cluster["issue_type"].replace("_", " ")
    label = priority_label(cluster["priority_score"])

    # AGENTIC STEP: autonomously decide which department/action route
    # this cluster should take, before generating the rest of the brief
    route_decision = decide_action_route(cluster)

    # Problem Summary
    problem_summary = (
        f"{cluster['complaint_count']} citizen complaints regarding {issue_readable} "
        f"have been logged in {cluster['ward']}, with the most recent complaint on "
        f"{cluster['most_recent_complaint']}. "
        f"{'This location meets the recurring-hotspot threshold and requires escalation.' if cluster['is_recurring_hotspot'] else 'This is currently an isolated pattern, below the recurring-hotspot threshold.'}"
    )

    # Grounded recommendation -- try each retrieved chunk in order until
    # one yields an actionable (time-bound / escalation) sentence, rather
    # than always defaulting to chunk #1 even when it's purely descriptive.
    recommendation = None
    grounding_note = None
    for chunk in retrieved_chunks:
        actionable_line = extract_actionable_sentence(chunk["text"], header=chunk.get("header"))
        if actionable_line:
            source_label = chunk['source_doc'].replace('.txt', '').replace('sop_', '').upper()
            recommendation = f"Per {source_label} SOP ({chunk['header']}): {actionable_line}"
            grounding_note = (
                f"Recommendation grounded in retrieved SOP section "
                f"(relevance score: {chunk['relevance_score']})."
            )
            break

    if recommendation is None:
        if retrieved_chunks:
            recommendation = (
                f"Relevant SOP section found ({retrieved_chunks[0]['header']}) but no specific "
                f"time-bound action identified; flag for manual policy review."
            )
            grounding_note = "Low-confidence grounding -- descriptive SOP content only."
        else:
            recommendation = "No matching SOP guidance found; flag for manual policy review."
            grounding_note = "No relevant SOP context retrieved for this issue type."

    brief = {
        "ward": cluster["ward"],
        "issue_type": issue_readable,
        "priority_level": label,
        "priority_score": cluster["priority_score"],
        "complaint_count": cluster["complaint_count"],
        "is_recurring_hotspot": cluster["is_recurring_hotspot"],
        "action_route": route_decision["route"],
        "target_department": route_decision["target_department"],
        "routing_reasoning": route_decision["reasoning"],
        "problem_summary": problem_summary,
        "grounded_recommendation": recommendation,
        "grounding_note": grounding_note,
        "sample_complaint_evidence": cluster.get("sample_texts", []),
    }
    return brief


def format_brief_as_text(brief):
    lines = [
        "=" * 72,
        f"MUNICIPAL POLICY BRIEF  |  Priority: {brief['priority_level']} "
        f"(score: {brief['priority_score']})",
        "=" * 72,
        f"Ward: {brief['ward']}",
        f"Issue Type: {brief['issue_type'].title()}",
        f"Total Complaints: {brief['complaint_count']}  |  "
        f"Recurring Hotspot: {'YES' if brief['is_recurring_hotspot'] else 'No'}",
        "",
        f"AGENTIC ROUTING DECISION: {brief['action_route']}",
        f"  -> Routed to: {brief['target_department']}",
        f"  -> Reasoning: {brief['routing_reasoning']}",
        "",
        "PROBLEM SUMMARY:",
        f"  {brief['problem_summary']}",
        "",
        "GROUNDED RECOMMENDATION:",
        f"  {brief['grounded_recommendation']}",
        f"  ({brief['grounding_note']})",
        "",
        "EVIDENCE (sample citizen complaints):",
    ]
    for t in brief["sample_complaint_evidence"]:
        lines.append(f"  - \"{t}\"")
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    # lazy imports to reuse Day 2 modules
    from rag_pipeline import build_knowledge_base, RAGRetriever, build_query_from_cluster

    with open("../data/clusters_structured.json") as f:
        clusters = json.load(f)

    chunks = build_knowledge_base()
    retriever = RAGRetriever(chunks)

    briefs = []
    print("Generating policy briefs for top 5 priority hotspots...\n")
    for cluster in clusters[:5]:
        query = build_query_from_cluster(cluster)
        retrieved = retriever.retrieve(query, k=3)
        brief = generate_policy_brief(cluster, retrieved)
        briefs.append(brief)
        print(format_brief_as_text(brief))
        print()

    with open("../data/policy_briefs.json", "w") as f:
        json.dump(briefs, f, indent=2)
    print(f"Saved {len(briefs)} policy briefs to ../data/policy_briefs.json")


if __name__ == "__main__":
    main()
