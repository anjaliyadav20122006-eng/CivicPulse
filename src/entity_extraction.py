"""
CivicPulse - Entity Extraction Pipeline
=========================================
Extracts structured signals from raw, unstructured grievance text:
    - ward / location
    - issue category
    - urgency level (low / medium / high)

Design choice: rule-based (regex + keyword lexicons) rather than a
black-box classifier. This is deliberate for the "Transparency" pillar
of Responsible AI -- every extraction decision can be traced back to
the exact keyword/pattern that triggered it, which we surface in the
final policy brief. A learned classifier could replace this module
later without changing the pipeline's interface.
"""

import re
import json
import csv
from collections import Counter

# ---------------------------------------------------------------------
# Lexicons (kept in sync conceptually with generate_dataset.py, but
# treated as INDEPENDENT knowledge here -- in a real deployment this
# pipeline would never see the "_true_*" labels)
# ---------------------------------------------------------------------

WARD_PATTERN = re.compile(r"\bWard\s?(\d{1,2})\b", re.IGNORECASE)

ISSUE_KEYWORDS = {
    "waterlogging": ["waterlog", "flood", "stagnant water", "paani bhar", "drain overflow", "water enter"],
    "garbage": ["garbage", "trash", "dump", "waste", "foul smell", "bin"],
    "streetlight": ["streetlight", "street light", "lamp", "dark at night", "unsafe at night"],
    "pothole": ["pothole", "road damaged", "skidding", "accidents"],
    "water_supply": ["no water supply", "water supply", "contaminated water", "tap water", "pipeline"],
    "sewage": ["sewage", "manhole", "drain broken", "dirty water flowing"],
}

URGENCY_HIGH_KEYWORDS = [
    "urgent", "urgently", "immediately", "emergency", "accidents",
    "unsafe", "unhygienic", "health hazard", "children affected",
]
URGENCY_MED_KEYWORDS = [
    "please help", "worried", "not fixed", "still not", "repeated",
    "again", "several months", "over a month",
]

DURATION_PATTERN = re.compile(
    r"(\d+\s*(?:day|days|week|weeks|month|months))|(?:a\s+week)|(?:several months)",
    re.IGNORECASE,
)


def extract_ward(text):
    match = WARD_PATTERN.search(text)
    if match:
        return f"Ward {int(match.group(1))}"
    return "Unknown"


def extract_issue(text):
    text_lower = text.lower()
    scores = {}
    for issue, keywords in ISSUE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[issue] = score
    if not scores:
        return "uncategorized"
    # pick the issue with the highest keyword hit count
    return max(scores, key=scores.get)


def extract_urgency(text):
    text_lower = text.lower()
    high_hits = sum(1 for kw in URGENCY_HIGH_KEYWORDS if kw in text_lower)
    med_hits = sum(1 for kw in URGENCY_MED_KEYWORDS if kw in text_lower)
    if high_hits > 0:
        return "high"
    elif med_hits > 0:
        return "medium"
    return "low"


def extract_duration(text):
    match = DURATION_PATTERN.search(text)
    return match.group(0) if match else "not specified"


def extract_entities(complaint):
    text = complaint["raw_text"]
    return {
        **complaint,
        "extracted_ward": extract_ward(text),
        "extracted_issue": extract_issue(text),
        "extracted_urgency": extract_urgency(text),
        "extracted_duration": extract_duration(text),
    }


def evaluate_accuracy(enriched):
    """Compare extracted labels against the synthetic ground truth
    (only possible because we generated the data ourselves -- used here
    purely to validate the pipeline, exactly as we'd validate against a
    small hand-labeled sample in a real deployment)."""
    ward_correct = sum(1 for c in enriched if c["extracted_ward"] == c["_true_ward"])
    issue_correct = sum(1 for c in enriched if c["extracted_issue"] == c["_true_issue"])
    n = len(enriched)
    print(f"Ward extraction accuracy:  {ward_correct}/{n} = {ward_correct/n:.1%}")
    print(f"Issue extraction accuracy: {issue_correct}/{n} = {issue_correct/n:.1%}")

    urgency_dist = Counter(c["extracted_urgency"] for c in enriched)
    print(f"Urgency distribution: {dict(urgency_dist)}")


def main(in_json="../data/grievances_raw.json",
         out_json="../data/grievances_enriched.json",
         out_csv="../data/grievances_enriched.csv"):
    with open(in_json) as f:
        complaints = json.load(f)

    enriched = [extract_entities(c) for c in complaints]

    with open(out_json, "w") as f:
        json.dump(enriched, f, indent=2)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=enriched[0].keys())
        writer.writeheader()
        writer.writerows(enriched)

    print(f"Enriched {len(enriched)} complaints with extracted entities.")
    print(f"Saved to {out_json} and {out_csv}\n")

    evaluate_accuracy(enriched)


if __name__ == "__main__":
    main()
