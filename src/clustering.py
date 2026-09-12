"""
CivicPulse - Clustering & Hotspot Detection
==============================================
Groups extracted complaints into recurring problem clusters and assigns
a transparent, explainable priority score to each cluster.

Two complementary clustering strategies are used:

1. STRUCTURED CLUSTERING (primary, explainable):
   Group by (ward, issue_type). This directly answers the officer's
   real question: "which ward + issue combos keep coming back?"
   Priority score is a simple weighted formula over three factors:
       - frequency   (how many complaints)
       - urgency     (share of high/medium urgency complaints)
       - recency     (how recently complaints are still coming in)
   Every score is fully traceable -- no black box.

2. SEMANTIC CLUSTERING (secondary, exploratory):
   TF-IDF + KMeans over raw complaint text, to surface cross-ward
   patterns that a strict (ward, issue) grouping might miss (e.g. a
   city-wide drainage design flaw showing up in the language of
   complaints from multiple wards).
"""

import json
from collections import defaultdict
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


URGENCY_SCORE = {"high": 1.0, "medium": 0.5, "low": 0.1}

# Minimum number of complaints for a (ward, issue) combo to be treated
# as a "recurring hotspot" worth flagging to officers.
HOTSPOT_THRESHOLD = 5


def structured_clusters(enriched):
    groups = defaultdict(list)
    for c in enriched:
        key = (c["extracted_ward"], c["extracted_issue"])
        groups[key].append(c)

    clusters = []
    max_date = max(datetime.strptime(c["date"], "%Y-%m-%d") for c in enriched)

    for (ward, issue), items in groups.items():
        n = len(items)
        avg_urgency = sum(URGENCY_SCORE[c["extracted_urgency"]] for c in items) / n

        dates = [datetime.strptime(c["date"], "%Y-%m-%d") for c in items]
        most_recent = max(dates)
        days_since_last = (max_date - most_recent).days
        # recency score: 1.0 if complaint came in very recently, decays over time
        recency_score = max(0.0, 1 - (days_since_last / 60))

        # Transparent, documented weighting -- easy to justify to a reviewer:
        #   frequency matters most (40%), urgency next (35%), recency least (25%)
        freq_score = min(1.0, n / 20)  # normalize; 20+ complaints = max frequency score
        priority_score = round(0.40 * freq_score + 0.35 * avg_urgency + 0.25 * recency_score, 3)

        clusters.append({
            "ward": ward,
            "issue_type": issue,
            "complaint_count": n,
            "avg_urgency_score": round(avg_urgency, 3),
            "most_recent_complaint": most_recent.strftime("%Y-%m-%d"),
            "days_since_last_complaint": days_since_last,
            "priority_score": priority_score,
            "is_recurring_hotspot": n >= HOTSPOT_THRESHOLD,
            "sample_complaint_ids": [c["complaint_id"] for c in items[:3]],
            "sample_texts": [c["raw_text"] for c in items[:2]],
        })

    clusters.sort(key=lambda x: x["priority_score"], reverse=True)
    return clusters


def semantic_clusters(enriched, n_clusters=8):
    texts = [c["raw_text"] for c in enriched]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(texts)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # top terms per semantic cluster, for interpretability
    terms = vectorizer.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    semantic_summary = []
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :6]]
        member_idxs = [idx for idx, lbl in enumerate(labels) if lbl == i]
        wards_in_cluster = set(enriched[idx]["extracted_ward"] for idx in member_idxs)
        semantic_summary.append({
            "semantic_cluster_id": i,
            "size": len(member_idxs),
            "top_terms": top_terms,
            "distinct_wards_involved": len(wards_in_cluster),
            "sample_text": enriched[member_idxs[0]]["raw_text"] if member_idxs else None,
        })
    return semantic_summary


def main(in_json="../data/grievances_enriched.json",
         out_structured="../data/clusters_structured.json",
         out_semantic="../data/clusters_semantic.json"):
    with open(in_json) as f:
        enriched = json.load(f)

    structured = structured_clusters(enriched)
    with open(out_structured, "w") as f:
        json.dump(structured, f, indent=2)

    hotspots = [c for c in structured if c["is_recurring_hotspot"]]
    print(f"Total (ward, issue) clusters: {len(structured)}")
    print(f"Recurring hotspots (>= {HOTSPOT_THRESHOLD} complaints): {len(hotspots)}\n")

    print("TOP 5 PRIORITY CLUSTERS:")
    for c in structured[:5]:
        print(f"  {c['ward']} | {c['issue_type']:15s} | count={c['complaint_count']:3d} "
              f"| priority={c['priority_score']} | hotspot={c['is_recurring_hotspot']}")

    semantic = semantic_clusters(enriched)
    with open(out_semantic, "w") as f:
        json.dump(semantic, f, indent=2)

    print("\nSEMANTIC CLUSTERS (cross-ward text patterns):")
    for s in semantic:
        print(f"  Cluster {s['semantic_cluster_id']}: size={s['size']:3d}, "
              f"wards_involved={s['distinct_wards_involved']}, top_terms={s['top_terms']}")

    print(f"\nSaved structured clusters to {out_structured}")
    print(f"Saved semantic clusters to {out_semantic}")


if __name__ == "__main__":
    main()
