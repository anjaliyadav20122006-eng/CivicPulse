"""
CivicPulse - PII / Privacy Redaction Module
===============================================
Operationalizes the "Privacy" pillar of the internship's Responsible AI
requirements. Real citizens frequently include personal identifiers
(phone numbers, email addresses, sometimes their name) in free-text
grievance submissions when asking to be contacted about resolution.

Before any complaint text is stored, clustered, or shown to an officer
in a policy brief, this module strips such identifiers -- the pipeline
should reason about WHAT the problem is and WHERE it is, never WHO
specifically reported it (that mapping stays only in the original
grievance portal's secure database, outside this analysis pipeline).

This is deliberately regex/rule-based (not a black-box NER model) so
that every redaction decision is inspectable and auditable -- it also
reinforces the Transparency pillar alongside Privacy.
"""

import re
import json


PHONE_PATTERN = re.compile(
    r'(\+?91[\-\s]?)?[6-9]\d{9}\b'  # Indian mobile numbers, with/without +91
)

EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

# Heuristic name-mention patterns: "my name is X Y", "I am X Y",
# "this is X Y speaking" -- catches self-identification specifically,
# rather than attempting full named-entity recognition (which would be
# unreliable on short, informal text and risks both false positives on
# place names and false negatives on unusual name formats).
NAME_MENTION_PATTERN = re.compile(
    r'\b(?:my name is|i am|this is)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})',
    re.IGNORECASE
)


def redact_text(text):
    """Redact PII from a single complaint text.

    Returns:
        redacted_text (str)
        redactions (list of dicts): what was found and replaced, for
                                     the audit log
    """
    redactions = []
    redacted = text

    for match in PHONE_PATTERN.finditer(text):
        redactions.append({"type": "phone", "original": match.group()})
    redacted = PHONE_PATTERN.sub("[PHONE_REDACTED]", redacted)

    for match in EMAIL_PATTERN.finditer(text):
        redactions.append({"type": "email", "original": match.group()})
    redacted = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", redacted)

    for match in NAME_MENTION_PATTERN.finditer(text):
        redactions.append({"type": "name_mention", "original": match.group(1)})
    redacted = NAME_MENTION_PATTERN.sub(
        lambda m: m.group(0).replace(m.group(1), "[NAME_REDACTED]"), redacted
    )

    return redacted, redactions


def redact_dataset(complaints):
    """Apply redaction across a full list of complaint dicts (expects a
    'raw_text' field). Returns the redacted complaints plus a summary
    audit log."""
    redacted_complaints = []
    audit_log = []

    for c in complaints:
        redacted_text, redactions = redact_text(c["raw_text"])
        new_c = dict(c)
        new_c["raw_text_redacted"] = redacted_text
        new_c["pii_found"] = len(redactions) > 0
        redacted_complaints.append(new_c)
        if redactions:
            audit_log.append({
                "complaint_id": c["complaint_id"],
                "redactions": redactions,
            })

    return redacted_complaints, audit_log


# ---------------------------------------------------------------------
# Demo examples -- our synthetic dataset was generated WITHOUT injected
# PII (see generate_dataset.py), since real portal text is what would
# actually contain it. These examples simulate what real citizen
# submissions often look like, to prove the module correctly detects
# and redacts identifiers before they'd ever reach the clustering or
# agent stages.
# ---------------------------------------------------------------------
DEMO_EXAMPLES = [
    "Waterlogging near Gandhi Chowk, Ward 12 for a week. Please call me on 9876543210 to update.",
    "My name is Rakesh Sharma, garbage not collected near Community Hall, Ward 7. Email me at rakesh.sharma@gmail.com",
    "This is Priya Verma speaking, streetlight broken near Ambedkar Park Ward 3, contact +91-9123456780.",
    "Pothole near Old Bus Stand Ward 5, causing accidents. No contact info given here.",
]


def run_demo():
    print("=" * 72)
    print("PRIVACY REDACTION -- BEFORE / AFTER DEMO")
    print("=" * 72)
    for text in DEMO_EXAMPLES:
        redacted, redactions = redact_text(text)
        print(f"\nBEFORE: {text}")
        print(f"AFTER:  {redacted}")
        if redactions:
            print(f"Redacted: {[r['type'] for r in redactions]}")
        else:
            print("Redacted: (nothing found)")
    print("\n" + "=" * 72)


def main():
    run_demo()

    with open("../data/grievances_raw.json") as f:
        complaints = json.load(f)

    redacted_complaints, audit_log = redact_dataset(complaints)

    with open("../data/grievances_privacy_redacted.json", "w") as f:
        json.dump(redacted_complaints, f, indent=2)

    with open("../data/privacy_audit_log.json", "w") as f:
        json.dump(audit_log, f, indent=2)

    print(f"\nProcessed {len(complaints)} complaints from the main dataset.")
    print(f"Complaints containing PII: {len(audit_log)}")
    print("(Low/zero count is expected -- our synthetic generator does not "
          "inject PII into the main 600-complaint dataset by design, since "
          "PII presence depends on individual citizen behavior, not the "
          "issue-report templates. The DEMO_EXAMPLES above simulate "
          "realistic PII-bearing submissions to validate the module works.)")
    print(f"\nSaved: ../data/grievances_privacy_redacted.json")
    print(f"Saved: ../data/privacy_audit_log.json")


if __name__ == "__main__":
    main()
