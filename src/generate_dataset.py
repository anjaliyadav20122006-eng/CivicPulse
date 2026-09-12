"""
CivicPulse - Synthetic Municipal Grievance Dataset Generator
================================================================
Generates a realistic synthetic dataset of citizen grievances submitted
to a Tier-2 Indian municipal corporation grievance portal (modeled after
public systems like Swachhata-App / PGRS).

Why synthetic data:
- Real portal data is not publicly accessible in a clean, licensed format.
- Synthetic generation lets us deliberately inject realistic patterns
  (recurring ward-level issues, seasonal spikes, uneven ward representation)
  that we can later verify our pipeline correctly detects.
- This is an accepted approach for a "conceptual or working prototype"
  per the internship project guidelines.
"""

import random
import json
import csv
from datetime import datetime, timedelta

random.seed(42)

# ---------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------

WARDS = [f"Ward {i}" for i in range(1, 16)]  # 15 wards

# Issue categories with sample complaint text templates.
# Templates include natural variation (formal/informal, typos, Hinglish
# flavor) to mimic real citizen-submitted text.
ISSUE_TEMPLATES = {
    "waterlogging": [
        "Heavy waterlogging near {landmark} in {ward} since {duration}. Road is completely flooded.",
        "Drain overflow causing water to enter houses near {landmark}, {ward}. Please help urgently.",
        "Every monsoon the road near {landmark} in {ward} gets waterlogged, no permanent solution done yet.",
        "Stagnant water near {landmark} for {duration}, mosquito breeding ground, {ward} residents worried.",
        "Sir humara area {ward} {landmark} ke paas paani bhar gaya hai, bahut dino se yehi problem hai.",
    ],
    "garbage": [
        "Garbage not collected for {duration} near {landmark}, {ward}. Foul smell spreading.",
        "Overflowing garbage bin at {landmark}, {ward}, please send collection truck.",
        "Illegal dumping of construction waste near {landmark} in {ward}, causing blockage.",
        "Municipal garbage van has skipped our street near {landmark} ({ward}) for {duration}.",
    ],
    "streetlight": [
        "Streetlight near {landmark}, {ward} not working for {duration}, safety issue at night.",
        "Multiple streetlights down on the stretch near {landmark} in {ward}, women feel unsafe.",
        "Complaint raised earlier too - streetlight at {landmark} ({ward}) still not fixed.",
    ],
    "pothole": [
        "Large pothole near {landmark}, {ward} causing accidents, two-wheelers keep skidding.",
        "Road near {landmark} in {ward} badly damaged after rains, potholes everywhere.",
        "Same pothole near {landmark} ({ward}) reported {duration} ago, still not repaired.",
    ],
    "water_supply": [
        "No water supply near {landmark}, {ward} for {duration}, please check pipeline.",
        "Contaminated water coming from tap near {landmark} in {ward}, smells bad.",
        "Irregular water supply timing in {ward} near {landmark}, affecting daily routine.",
    ],
    "sewage": [
        "Sewage line broken near {landmark}, {ward}, dirty water flowing on road for {duration}.",
        "Manhole overflow near {landmark} in {ward}, extremely unhygienic condition.",
    ],
}

LANDMARKS = [
    "Gandhi Chowk", "Railway Station Road", "Old Bus Stand", "Sector 4 Market",
    "Community Hall", "Shiv Mandir Road", "Govt School", "Housing Colony Gate",
    "Main Bazaar", "Ring Road Junction", "Petrol Pump Circle", "Riverside Lane",
    "Industrial Area Gate", "New Market Complex", "Ambedkar Park",
]

DURATIONS = ["2 days", "a week", "10 days", "3 weeks", "over a month", "5 days", "several months"]

URGENCY_KEYWORDS_HIGH = ["urgent", "urgently", "immediately", "emergency", "accidents",
                          "unsafe", "unhygienic", "children affected", "health hazard"]
URGENCY_KEYWORDS_MED = ["please help", "worried", "not fixed", "still not", "repeated"]

# Deliberately bias data generation: Ward 7 and Ward 12 get disproportionately
# MORE waterlogging complaints (simulates a genuine recurring infra failure),
# while a couple of wards (e.g. Ward 14, Ward 15) get very FEW complaints
# overall (simulates under-reporting / lower smartphone-literacy areas --
# useful later for the fairness/equity discussion in Responsible AI section).
WARD_WEIGHTS = {w: 1.0 for w in WARDS}
WARD_WEIGHTS["Ward 7"] = 3.5
WARD_WEIGHTS["Ward 12"] = 3.0
WARD_WEIGHTS["Ward 3"] = 2.0
WARD_WEIGHTS["Ward 14"] = 0.3
WARD_WEIGHTS["Ward 15"] = 0.3

ISSUE_WEIGHTS = {
    "waterlogging": 1.4,
    "garbage": 1.2,
    "streetlight": 0.9,
    "pothole": 1.0,
    "water_supply": 0.8,
    "sewage": 0.7,
}


def weighted_choice(weight_dict):
    items = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return random.choices(items, weights=weights, k=1)[0]


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_complaint(complaint_id, start_date, end_date):
    ward = weighted_choice(WARD_WEIGHTS)
    issue = weighted_choice(ISSUE_WEIGHTS)
    template = random.choice(ISSUE_TEMPLATES[issue])
    landmark = random.choice(LANDMARKS)
    duration = random.choice(DURATIONS)

    text = template.format(landmark=landmark, ward=ward, duration=duration)

    date = random_date(start_date, end_date)

    return {
        "complaint_id": f"CMP{complaint_id:05d}",
        "date": date.strftime("%Y-%m-%d"),
        "raw_text": text,
        # ground-truth labels kept ONLY for evaluation purposes later;
        # our extraction pipeline will re-derive these independently from raw_text
        "_true_ward": ward,
        "_true_issue": issue,
    }


def main(n_complaints=600, out_json="../data/grievances_raw.json", out_csv="../data/grievances_raw.csv"):
    start_date = datetime(2026, 6, 1)   # monsoon season start
    end_date = datetime(2026, 9, 1)

    complaints = [generate_complaint(i, start_date, end_date) for i in range(1, n_complaints + 1)]

    # sort by date to look like a real chronological feed
    complaints.sort(key=lambda c: c["date"])

    with open(out_json, "w") as f:
        json.dump(complaints, f, indent=2)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=complaints[0].keys())
        writer.writeheader()
        writer.writerows(complaints)

    print(f"Generated {len(complaints)} synthetic grievances.")
    print(f"Saved to {out_json} and {out_csv}")

    # quick sanity summary
    from collections import Counter
    ward_counts = Counter(c["_true_ward"] for c in complaints)
    issue_counts = Counter(c["_true_issue"] for c in complaints)
    print("\nWard distribution (top 5):", ward_counts.most_common(5))
    print("Issue distribution:", issue_counts.most_common())


if __name__ == "__main__":
    main()
