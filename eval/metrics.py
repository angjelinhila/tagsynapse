"""Metrics, split by whether they need data we don't have yet.

Study-dependent (below, `NotImplementedError`): need the controlled study
(M0 / M1 / M2 conditions, BLUEPRINT §14) or relevance judgments. Left failing
loudly on purpose rather than returning a fake number - see
docs/EVALUATION.md for the full breakdown.

Vocabulary-state metrics (below that): computable from the tagging CSVs
alone, no study needed. These run today on every snapshot - see snapshot.py.
"""
import pandas as pd

from tagsynapse import usage as U


# --- Study-dependent: need condition labels or relevance judgments ---------

def tag_acceptance_rate(suggestion_log: pd.DataFrame) -> float:
    """% of suggested tags a user accepted, per condition. Needs the app to
    log *suggestions shown*, not just tags saved - today it logs only the
    latter (user_tag_assignments.csv), so the denominator doesn't exist yet."""
    raise NotImplementedError


def semantic_alignment(assign_df: pd.DataFrame, vocab_emb, user_emb) -> float:
    """Mean cosine between a user's chosen tags and the concept(s) a
    curator would have chosen. Needs gold labels for a sample of documents."""
    raise NotImplementedError


def retrieval_precision_recall(ranked_doc_ids: list, relevant_doc_ids: set, k: int) -> tuple[float, float]:
    """P@k / R@k for the relational retrieval path (TAG_MODEL §20). Needs
    relevance judgments per query - none collected yet."""
    raise NotImplementedError


def perceived_usefulness_trust_control(survey_df: pd.DataFrame) -> dict:
    """Likert aggregates from the post-task questionnaire. Survey instrument
    not finalised; see docs/EVALUATION.md §4."""
    raise NotImplementedError


# --- Computable today: vocabulary-state metrics, from the CSVs alone --------

def vocabulary_size(user_tags_df: pd.DataFrame) -> int:
    """|F_t| - distinct user-created tag labels."""
    if user_tags_df is None or user_tags_df.empty:
        return 0
    return int(user_tags_df["tag_label"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())


def controlled_vocab_coverage(assign_df: pd.DataFrame, vocab_labels: set[str]) -> float:
    """Fraction of *assignments* whose label is a controlled concept.
    1.0 = everyone tags straight from the controlled vocabulary;
    0.0 = pure folksonomy. The whole point of TagSynapse is that this number
    should neither be forced to 1 nor allowed to collapse to 0."""
    if assign_df is None or assign_df.empty:
        return 0.0
    labels = assign_df["tag_label"].astype(str).str.strip()
    labels = labels[labels != ""]
    if labels.empty:
        return 0.0
    return float(labels.isin(vocab_labels).mean())


def custom_tag_rate(assign_df: pd.DataFrame, vocab_labels: set[str]) -> float:
    """1 - controlled_vocab_coverage. Named separately because BLUEPRINT §14
    lists 'custom tag creation' as its own outcome."""
    if assign_df is None or assign_df.empty:
        return 0.0
    return 1.0 - controlled_vocab_coverage(assign_df, vocab_labels)


def tags_per_doc(assign_df: pd.DataFrame) -> float:
    """Mean |T_d| over tagged documents."""
    doc_tags = U.doc_tag_sets(assign_df)
    if not doc_tags:
        return 0.0
    return sum(len(t) for t in doc_tags.values()) / len(doc_tags)


def hierarchy_depth_stats(user_tags_df: pd.DataFrame) -> dict:
    """How much of the folksonomy has been given parents by users, and how
    deep the user-built hierarchy goes. Cycles are reported, not followed."""
    if user_tags_df is None or user_tags_df.empty:
        return {"n_with_parent": 0, "max_depth": 0, "n_cycles": 0}
    parent = {
        str(r["tag_label"]).strip(): str(r["parent_tag_label"]).strip()
        for _, r in user_tags_df.iterrows()
        if str(r.get("parent_tag_label", "")).strip()
    }
    max_depth, n_cycles = 0, 0
    for start in parent:
        seen, node, depth = {start}, start, 0
        while node in parent and parent[node]:
            node = parent[node]
            depth += 1
            if node in seen:
                n_cycles += 1
                break
            seen.add(node)
        max_depth = max(max_depth, depth)
    return {"n_with_parent": len(parent), "max_depth": max_depth, "n_cycles": n_cycles}


def couse_graph_stats(assign_df: pd.DataFrame, min_count: int = 1) -> dict:
    """Size and density of the empirical coUsedWith graph (TAG_MODEL §7) and
    how many pairs show strong behavioral subsumption (§11) - i.e. how much
    *directional* hierarchy evidence usage has produced so far."""
    edges = U.couse_edges(assign_df, min_count=min_count)
    if edges.empty:
        return {"n_couse_edges": 0, "mean_jaccard": 0.0, "n_subsumption_candidates": 0}
    cands = U.subsumption_candidates(edges)
    return {
        "n_couse_edges": int(len(edges)),
        "mean_jaccard": float(edges["jaccard"].mean()),
        "n_subsumption_candidates": int(len(cands)),
    }


def canonicalization_rate(canon_df: pd.DataFrame, assign_df: pd.DataFrame) -> float:
    """Of the labels users typed, what fraction got mapped onto an existing
    concept (TAG_MODEL §15, the 'existing concept / alternate label' branch).
    Denominator is total assignments, so this is a lower bound."""
    if canon_df is None or canon_df.empty or assign_df is None or assign_df.empty:
        return 0.0
    changed = canon_df[canon_df["original_label"].astype(str) != canon_df["canonical_label"].astype(str)]
    return float(len(changed) / len(assign_df)) if len(assign_df) else 0.0
