"""Collaborative-usage statistics (docs/TAG_MODEL.md §7, §11).

Everything here is computed from the tag-assignment log alone - the
(document, tag) pairs users have actually accepted. No embeddings, no model,
no network. That is deliberate: these are the *empirical* edges (R_emp) and
must stay separable from anything the ontology or the encoder asserts.

Input convention: `assign_df` is the DataFrame behind user_tag_assignments.csv
with at least the columns the app writes:
    subreddit_key, corpus_path, row_id, tag_label
A document is identified by the (subreddit_key, corpus_path, row_id) triple.
"""
from collections import Counter
from itertools import combinations
import math

import pandas as pd

DOC_KEY_COLS = ("subreddit_key", "corpus_path", "row_id")


def doc_key(df: pd.DataFrame) -> pd.Series:
    """Stable per-document key from the three identifying columns."""
    return (
        df["subreddit_key"].astype(str)
        + "||"
        + df["corpus_path"].astype(str)
        + "||"
        + df["row_id"].astype(str)
    )


def doc_tag_sets(assign_df: pd.DataFrame) -> dict[str, set[str]]:
    """{doc_key: set(tag labels)} - the Y_dt indicator matrix in sparse form."""
    if assign_df is None or assign_df.empty:
        return {}
    tmp = assign_df.dropna(subset=["tag_label"]).copy()
    tmp["tag_label"] = tmp["tag_label"].astype(str).str.strip()
    tmp = tmp[tmp["tag_label"] != ""]
    tmp["_doc"] = doc_key(tmp)
    return {k: set(g["tag_label"]) for k, g in tmp.groupby("_doc")}


def tag_counts(doc_tags: dict[str, set[str]]) -> Counter:
    """N_i = number of documents carrying tag i."""
    c: Counter = Counter()
    for tags in doc_tags.values():
        c.update(tags)
    return c


def couse_counts(doc_tags: dict[str, set[str]]) -> Counter:
    """N_ij = sum_d Y_di Y_dj over unordered pairs (i < j lexically)."""
    c: Counter = Counter()
    for tags in doc_tags.values():
        if len(tags) < 2:
            continue
        for a, b in combinations(sorted(tags), 2):
            c[(a, b)] += 1
    return c


def jaccard(n_ij: int, n_i: int, n_j: int) -> float:
    """J(i,j) = N_ij / (N_i + N_j - N_ij). 0 when the union is empty."""
    union = n_i + n_j - n_ij
    return n_ij / union if union > 0 else 0.0


def pmi(n_ij: int, n_i: int, n_j: int, n_docs: int) -> float:
    """PMI(i,j) = log P(i,j) / (P(i) P(j)), with P(.) = N(.)/n_docs.
    Returns 0.0 when any count is zero rather than -inf, so it can sit in a
    CSV column without poisoning downstream sums.
    """
    if n_ij == 0 or n_i == 0 or n_j == 0 or n_docs == 0:
        return 0.0
    return math.log((n_ij * n_docs) / (n_i * n_j))


def behavioral_subsumption(n_ij: int, n_i: int, n_j: int) -> float:
    """B(i,j) = P(j|i) - P(i|j)  (TAG_MODEL §11).

    B >> 0 : whenever i appears, j usually does too, but not vice versa -
             evidence that i is *narrower* than j.
    B << 0 : the reverse.
    Evidence, not proof - the model says so and so does this docstring.
    """
    p_j_given_i = n_ij / n_i if n_i else 0.0
    p_i_given_j = n_ij / n_j if n_j else 0.0
    return p_j_given_i - p_i_given_j


def couse_edges(assign_df: pd.DataFrame, min_count: int = 1) -> pd.DataFrame:
    """All empirical coUsedWith edges with their association statistics.

    Columns: src_label, dst_label, relation, n_ij, n_i, n_j, jaccard, pmi,
    subsumption. One row per unordered pair with N_ij >= min_count.
    `relation` is always "coUsedWith" - the R_emp name from TAG_MODEL §2 -
    which relations.relation_bucket maps to "cooccurs_with".
    """
    doc_tags = doc_tag_sets(assign_df)
    n_docs = len(doc_tags)
    counts = tag_counts(doc_tags)
    pairs = couse_counts(doc_tags)

    rows = []
    for (a, b), n_ij in pairs.items():
        if n_ij < min_count:
            continue
        n_i, n_j = counts[a], counts[b]
        rows.append(
            {
                "src_label": a,
                "dst_label": b,
                "relation": "coUsedWith",
                "n_ij": int(n_ij),
                "n_i": int(n_i),
                "n_j": int(n_j),
                "jaccard": jaccard(n_ij, n_i, n_j),
                "pmi": pmi(n_ij, n_i, n_j, n_docs),
                "subsumption": behavioral_subsumption(n_ij, n_i, n_j),
            }
        )
    cols = ["src_label", "dst_label", "relation", "n_ij", "n_i", "n_j", "jaccard", "pmi", "subsumption"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["n_ij", "jaccard"], ascending=[False, False])
        .reset_index(drop=True)
    )


def subsumption_candidates(edges: pd.DataFrame, threshold: float = 0.5, min_count: int = 2) -> pd.DataFrame:
    """Pairs whose |B| clears `threshold`, oriented so that `narrower` is the
    tag with the higher conditional. These are *proposals* for a human (or
    for the relation classifier in TAG_MODEL §10) - never written into the
    ontology directly.
    """
    if edges.empty:
        return pd.DataFrame(columns=["narrower", "broader", "subsumption", "n_ij"])
    strong = edges[(edges["subsumption"].abs() >= threshold) & (edges["n_ij"] >= min_count)]
    rows = []
    for _, r in strong.iterrows():
        if r["subsumption"] > 0:
            rows.append({"narrower": r["src_label"], "broader": r["dst_label"], "subsumption": r["subsumption"], "n_ij": r["n_ij"]})
        else:
            rows.append({"narrower": r["dst_label"], "broader": r["src_label"], "subsumption": -r["subsumption"], "n_ij": r["n_ij"]})
    return pd.DataFrame(rows, columns=["narrower", "broader", "subsumption", "n_ij"]).sort_values(
        "subsumption", ascending=False
    ).reset_index(drop=True)
