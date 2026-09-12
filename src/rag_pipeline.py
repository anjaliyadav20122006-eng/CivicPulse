"""
CivicPulse - RAG Knowledge Base & Retrieval Pipeline
========================================================
Implements Retrieval-Augmented Generation (RAG) grounding for policy
brief generation.

Design choice -- TF-IDF retrieval instead of neural embeddings:
    Dense neural embeddings (e.g. sentence-transformers, OpenAI
    embeddings) typically require downloading large model weights from
    the internet, which is unreliable in constrained/offline environments
    and adds a heavy dependency for a prototype. TF-IDF + cosine
    similarity is:
        - Fully local, no model download needed
        - Transparent (retrieval score is traceable to specific matching
          keywords -- reinforces our Transparency principle)
        - Sufficient for a small, focused knowledge base like ours
    This can be swapped for dense embeddings later without changing the
    pipeline's interface (retrieve(query, k) -> list of chunks).

Pipeline:
    1. Load raw SOP documents (.txt files)
    2. Chunk each document into paragraph-level sections
    3. Build a TF-IDF index over all chunks
    4. Given a query (derived from a hotspot cluster), retrieve the
       top-k most relevant chunks to ground the policy brief
"""

import os
import glob
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")


def load_documents(doc_dir=None):
    """Load all SOP .txt documents from the data directory."""
    if doc_dir is None:
        doc_dir = DEFAULT_DATA_DIR
    docs = {}
    for path in glob.glob(os.path.join(doc_dir, "sop_*.txt")):
        name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            docs[name] = f.read()
    return docs


def chunk_document(text, doc_name):
    """Split a document into chunks at double-newline (paragraph/section)
    boundaries. Each chunk keeps its source document name and a
    section header (first line) for traceability in the final brief."""
    raw_sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    chunks = []
    for i, section in enumerate(raw_sections):
        lines = section.split("\n")
        header = lines[0] if lines else f"Section {i}"
        chunks.append({
            "chunk_id": f"{doc_name}::chunk{i}",
            "source_doc": doc_name,
            "header": header,
            "text": section,
        })
    return chunks


def build_knowledge_base(doc_dir=None):
    docs = load_documents(doc_dir)
    all_chunks = []
    for name, text in docs.items():
        all_chunks.extend(chunk_document(text, name))
    return all_chunks


class RAGRetriever:
    """A minimal, transparent TF-IDF retriever over SOP document chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_idxs = scores.argsort()[::-1][:k]
        results = []
        for idx in top_idxs:
            if scores[idx] <= 0:
                continue  # skip zero-relevance matches
            results.append({
                **self.chunks[idx],
                "relevance_score": round(float(scores[idx]), 3),
            })
        return results


def build_query_from_cluster(cluster):
    """Turn a hotspot cluster (from clustering.py output) into a natural
    language query for retrieval.

    NOTE: we deliberately EXCLUDE generic administrative words like
    "complaint", "recurring", "hotspot" from the query. These words
    appear across almost every SOP section (headers, purpose statements,
    reporting clauses), so including them dilutes the TF-IDF signal and
    causes the retriever to match on boilerplate instead of the
    issue-specific content. Repeating the issue-type keyword instead
    sharpens the match toward the right section.
    """
    issue = cluster["issue_type"].replace("_", " ")
    sample_texts = " ".join(cluster.get("sample_texts", []))
    # repeat issue keyword to boost its TF-IDF weight relative to
    # incidental words in the sample complaint text
    return f"{issue} {issue} {sample_texts}"


def main():
    chunks = build_knowledge_base()
    print(f"Loaded {len(chunks)} chunks from SOP documents.\n")

    retriever = RAGRetriever(chunks)

    # Save the knowledge base chunks for inspection
    with open("../data/kb_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    # Test retrieval against our actual top hotspot clusters
    with open("../data/clusters_structured.json") as f:
        clusters = json.load(f)

    print("=" * 70)
    print("TESTING RETRIEVAL ON TOP 5 PRIORITY HOTSPOTS")
    print("=" * 70)
    for cluster in clusters[:5]:
        query = build_query_from_cluster(cluster)
        results = retriever.retrieve(query, k=2)

        print(f"\nHOTSPOT: {cluster['ward']} | {cluster['issue_type']} "
              f"(priority={cluster['priority_score']})")
        print(f"Query: {query[:90]}...")
        print("Retrieved SOP context:")
        for r in results:
            print(f"  [{r['relevance_score']}] {r['source_doc']} -> {r['header']}")


if __name__ == "__main__":
    main()
