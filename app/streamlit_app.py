"""TagSynapse Streamlit interface.

Thin UI layer: corpus browsing, semantic + lexical search, tagging with
controlled-vocabulary and KG-mediated suggestions, and vocabulary analytics.
Every path and knob comes from tagsynapse.config; every bit of logic that
doesn't need Streamlit lives in the tagsynapse package so it can be tested
without a corpus or a GPU.

    streamlit run app/streamlit_app.py
"""
import re
import sys
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo
# root - make the package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tagsynapse import config, store
from tagsynapse.relations import relation_bucket as _relation_bucket
from tagsynapse.text import STOPWORDS, normalize_rows as _normalize_matrix, tokenize as tokenize_query

# ============================================================
# CONFIG - all values live in tagsynapse/config.py; aliased here so the
# rest of this file reads the same as before the extraction.
# ============================================================

FAISS_API_URL = config.FAISS_API_URL
SUBREDDIT_CONFIG = config.SUBREDDIT_CONFIG
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME
TFIDF_MAX_FEATURES = config.TFIDF_MAX_FEATURES
TFIDF_MAX_DOCS = config.TFIDF_MAX_DOCS
BM25_MAX_DOCS = config.BM25_MAX_DOCS

INDEX_DIR = config.INDEX_DIR
TAG_DATA_DIR = config.TAG_DATA_DIR
TAG_DATA_DIR.mkdir(parents=True, exist_ok=True)

USER_TAGS_PATH = config.USER_TAGS_PATH
TAG_ASSIGN_PATH = config.TAG_ASSIGN_PATH
TAG_VOCAB_PATH = config.TAG_VOCAB_PATH
TAG_VOCAB_INDEX_PATH = config.TAG_VOCAB_INDEX_PATH
TAG_VOCAB_EMB_PATH = config.TAG_VOCAB_EMB_PATH
KG_EDGES_INDEX_PATH = config.KG_EDGES_INDEX_PATH
TAG_EDGES_CAT_PATH = config.TAG_EDGES_CAT_PATH
TAG_CANON_LOG_PATH = config.TAG_CANON_LOG_PATH
TAG_COOCCUR_PATH = config.TAG_COOCCUR_PATH
USED_NODES_PATH = config.USED_NODES_PATH
USED_EDGES_PATH = config.USED_EDGES_PATH
TAG_GNN_PRED_PATH = config.TAG_GNN_PRED_PATH
USER_TAG_EMB_PATH = config.USER_TAG_EMB_PATH
TAG_CENTRALITY_PATH = config.TAG_CENTRALITY_PATH
HYBRID_GRAPH_PATH = config.HYBRID_GRAPH_PATH

# ============================================================
# TAG STORAGE
# ============================================================

load_user_tags = store.load_user_tags
save_user_tags = store.save_user_tags
load_tag_assignments = store.load_tag_assignments
save_tag_assignments = store.save_tag_assignments
_next_tag_id = store.next_tag_id


def get_tags_for_doc(subreddit_key: str, corpus_path: str, row_id: int):
    assign_df = load_tag_assignments()
    if assign_df.empty:
        return []
    mask = (
        (assign_df["subreddit_key"] == subreddit_key)
        & (assign_df["corpus_path"] == corpus_path)
        & (assign_df["row_id"] == int(row_id))
    )
    if not mask.any():
        return []
    tags = (
        assign_df.loc[mask, "tag_label"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    tags = [t for t in tags if t]
    return sorted(tags)


def add_tags_to_doc(
    subreddit_key: str,
    corpus_path: str,
    row: pd.Series,
    tag_text: str,
    parent_tag_text: str | None = None,
):
    tags_df = load_user_tags()
    assign_df = load_tag_assignments()

    if not isinstance(tag_text, str):
        return False, "No tags given."

    raw_tags = [t.strip() for t in tag_text.split(",") if t.strip()]
    if not raw_tags:
        return False, "No non-empty tags given."

    parent_label = ""
    if isinstance(parent_tag_text, str):
        parent_label = parent_tag_text.strip()

    if parent_label:
        existing_parent = tags_df[tags_df["tag_label"] == parent_label]
        if existing_parent.empty:
            new_id = _next_tag_id(tags_df)
            tags_df = pd.concat(
                [
                    tags_df,
                    pd.DataFrame(
                        [{"tag_id": new_id, "tag_label": parent_label, "parent_tag_label": ""}]
                    ),
                ],
                ignore_index=True,
            )

    for label in raw_tags:
        existing = tags_df[tags_df["tag_label"] == label]
        if existing.empty:
            new_id = _next_tag_id(tags_df)
            tags_df = pd.concat(
                [
                    tags_df,
                    pd.DataFrame(
                        [{"tag_id": new_id, "tag_label": label, "parent_tag_label": parent_label}]
                    ),
                ],
                ignore_index=True,
            )
        else:
            if parent_label and existing["parent_tag_label"].iloc[0] == "":
                tags_df.loc[existing.index, "parent_tag_label"] = parent_label

        doc_id = row.get("doc_id") if "doc_id" in row.index else None
        this_row_id = int(row["row_id"])
        mask = (
            (assign_df["subreddit_key"] == subreddit_key)
            & (assign_df["corpus_path"] == corpus_path)
            & (assign_df["row_id"] == this_row_id)
            & (assign_df["tag_label"] == label)
        )
        if not mask.any():
            assign_df = pd.concat(
                [
                    assign_df,
                    pd.DataFrame(
                        [{
                            "subreddit_key": subreddit_key,
                            "corpus_path": corpus_path,
                            "doc_id": doc_id if doc_id is not None else "",
                            "row_id": this_row_id,
                            "tag_label": label,
                        }]
                    ),
                ],
                ignore_index=True,
            )

    save_user_tags(tags_df)
    save_tag_assignments(assign_df)
    return True, f"Added/updated {len(raw_tags)} tag(s)."

# ============================================================
# CONTROLLED VOCAB / KG
# ============================================================

@st.cache_resource
def load_controlled_vocab_and_kg():
    if not TAG_VOCAB_PATH.exists() or not TAG_VOCAB_EMB_PATH.exists():
        return None, None, None

    try:
        vocab_df = pd.read_csv(TAG_VOCAB_PATH)
    except Exception:
        return None, None, None

    label_candidates = [
        c for c in vocab_df.columns
        if c.lower() in ("tag_label", "label", "tag", "canonical_label", "name")
    ]
    if not label_candidates:
        return None, None, None

    if label_candidates[0] != "tag_label":
        vocab_df = vocab_df.rename(columns={label_candidates[0]: "tag_label"})
    vocab_df["tag_label"] = vocab_df["tag_label"].astype(str).str.strip()

    try:
        emb = np.load(TAG_VOCAB_EMB_PATH).astype("float32")
    except Exception:
        return vocab_df, None, None

    idx_to_label = {}
    idx_to_tag_id = {}

    if TAG_VOCAB_INDEX_PATH.exists():
        try:
            idx_df = pd.read_csv(TAG_VOCAB_INDEX_PATH)
            label_candidates = [
                c for c in idx_df.columns
                if c.lower() in ("tag_label", "label", "tag", "canonical_label", "name")
            ]
            idx_candidates = [
                c for c in idx_df.columns
                if c.lower() in ("tag_idx", "emb_idx", "index", "idx", "row", "row_id")
            ]
            id_candidates = [
                c for c in idx_df.columns
                if c.lower() in ("tag_id", "id", "uri")
            ]
            if label_candidates and idx_candidates:
                lab_col = label_candidates[0]
                idx_col = idx_candidates[0]
                idx_to_label = {
                    int(row[idx_col]): str(row[lab_col]).strip()
                    for _, row in idx_df.iterrows()
                }
                if id_candidates:
                    id_col = id_candidates[0]
                    idx_to_tag_id = {
                        int(row[idx_col]): str(row[id_col]).strip()
                        for _, row in idx_df.iterrows()
                    }
        except Exception:
            idx_to_label = {}
            idx_to_tag_id = {}

    if not idx_to_label:
        idx_to_label = {
            i: str(vocab_df.iloc[i]["tag_label"]).strip()
            for i in range(min(len(vocab_df), emb.shape[0]))
        }

    n = min(len(vocab_df), emb.shape[0])
    vocab_df = vocab_df.iloc[:n].reset_index(drop=True)
    vocab_emb = emb[:n].astype("float32")

    tag_id_to_label = {}
    if "tag_id" in vocab_df.columns:
        tag_id_to_label = {
            str(row["tag_id"]).strip(): str(row["tag_label"]).strip()
            for _, row in vocab_df.iterrows()
        }

    kg_edges_df = None

    if KG_EDGES_INDEX_PATH.exists():
        try:
            raw = pd.read_csv(KG_EDGES_INDEX_PATH)
            src_candidates = [c for c in raw.columns if c.lower() in ("src_idx", "source_idx", "src", "source")]
            dst_candidates = [c for c in raw.columns if c.lower() in ("dst_idx", "target_idx", "dst", "target")]
            rel_candidates = [c for c in raw.columns if c.lower() in ("relation", "rel", "edge_type")]
            wt_candidates = [c for c in raw.columns if c.lower() in ("weight", "score")]

            if src_candidates and dst_candidates:
                src_col = src_candidates[0]
                dst_col = dst_candidates[0]
                rel_col = rel_candidates[0] if rel_candidates else None
                wt_col = wt_candidates[0] if wt_candidates else None

                out = raw.copy()
                out["src_label"] = out[src_col].map(idx_to_label)
                out["dst_label"] = out[dst_col].map(idx_to_label)
                out = out.dropna(subset=["src_label", "dst_label"]).copy()
                out["relation"] = out[rel_col].astype(str) if rel_col else "related_to"
                out["weight"] = pd.to_numeric(out[wt_col], errors="coerce").fillna(1.0) if wt_col else 1.0
                kg_edges_df = out[["src_label", "dst_label", "relation", "weight"]]
        except Exception:
            kg_edges_df = None

    if kg_edges_df is None and TAG_EDGES_CAT_PATH.exists():
        try:
            raw = pd.read_csv(TAG_EDGES_CAT_PATH)
            cols_lower = [c.lower() for c in raw.columns]

            src_label_col = None
            dst_label_col = None
            src_id_col = None
            dst_id_col = None

            for cand in ("src_label", "source_label", "src_tag", "source_tag", "src"):
                if cand in cols_lower:
                    src_label_col = raw.columns[cols_lower.index(cand)]
                    break
            for cand in ("dst_label", "target_label", "dst_tag", "target_tag", "dst"):
                if cand in cols_lower:
                    dst_label_col = raw.columns[cols_lower.index(cand)]
                    break

            for cand in ("src_tag_id", "source_tag_id", "src_id", "source_id"):
                if cand in cols_lower:
                    src_id_col = raw.columns[cols_lower.index(cand)]
                    break
            for cand in ("dst_tag_id", "target_tag_id", "dst_id", "target_id"):
                if cand in cols_lower:
                    dst_id_col = raw.columns[cols_lower.index(cand)]
                    break

            rel_col = raw.columns[cols_lower.index("relation")] if "relation" in cols_lower else None
            wt_col = raw.columns[cols_lower.index("weight")] if "weight" in cols_lower else None

            out = raw.copy()

            if src_label_col and dst_label_col:
                out["src_label"] = out[src_label_col].astype(str).str.strip()
                out["dst_label"] = out[dst_label_col].astype(str).str.strip()
            elif src_id_col and dst_id_col and tag_id_to_label:
                out["src_label"] = out[src_id_col].astype(str).str.strip().map(tag_id_to_label)
                out["dst_label"] = out[dst_id_col].astype(str).str.strip().map(tag_id_to_label)
            else:
                out["src_label"] = np.nan
                out["dst_label"] = np.nan

            out = out.dropna(subset=["src_label", "dst_label"]).copy()
            out["relation"] = out[rel_col].astype(str) if rel_col else "related_to"
            out["weight"] = pd.to_numeric(out[wt_col], errors="coerce").fillna(1.0) if wt_col else 1.0
            kg_edges_df = out[["src_label", "dst_label", "relation", "weight"]]
        except Exception:
            kg_edges_df = None

    return vocab_df, vocab_emb, kg_edges_df


def suggest_tags_for_text(
    text: str,
    model,
    vocab_df: pd.DataFrame | None,
    vocab_emb: np.ndarray | None,
    top_k: int = 5,
    sim_threshold: float = 0.4,
):
    if not text or vocab_df is None or vocab_emb is None or vocab_df.empty:
        return []

    q = model.encode(text, convert_to_numpy=True).astype("float32")
    q_norm = q / (np.linalg.norm(q) + 1e-8)

    emb_mat = _normalize_matrix(vocab_emb)
    sims = emb_mat @ q_norm

    top_idx = np.argsort(sims)[::-1][:top_k]
    suggestions = []
    for idx in top_idx:
        score = float(sims[idx])
        if score < sim_threshold:
            continue
        label = vocab_df.loc[idx, "tag_label"]
        facet = vocab_df.loc[idx, "facet"] if "facet" in vocab_df.columns else None
        categories = vocab_df.loc[idx, "categories"] if "categories" in vocab_df.columns else None
        suggestions.append(
            {
                "tag_label": label,
                "score": score,
                "facet": facet,
                "categories": categories,
            }
        )
    return suggestions


def get_kg_neighbors(tag_label: str, kg_edges_df: pd.DataFrame | None):
    if kg_edges_df is None or kg_edges_df.empty:
        return {}

    neighbors = defaultdict(set)

    for _, row in kg_edges_df.iterrows():
        src = str(row["src_label"]).strip()
        tgt = str(row["dst_label"]).strip()
        if src == tag_label:
            other = tgt
        elif tgt == tag_label:
            other = src
        else:
            continue

        rel = str(row["relation"]) if "relation" in row.index else "related_to"
        neighbors[rel].add(other)

    return {k: sorted(list(v)) for k, v in neighbors.items()}

# ============================================================
# CANONICALIZATION / RELATION LOGGING
# ============================================================

load_canonicalization_events = store.load_canonicalization_events
append_canonicalization_events = store.append_canonicalization_events


def canonicalize_input_labels_detailed(
    tag_text: str,
    vocab_df: pd.DataFrame | None,
    vocab_emb: np.ndarray | None,
    model,
    sim_threshold: float = 0.7,
):
    raw_labels = [t.strip() for t in str(tag_text).split(",") if t.strip()]
    if not raw_labels:
        return [], [], ""

    if vocab_df is None or vocab_emb is None or vocab_df.empty:
        details = [
            {
                "original_label": lab,
                "canonical_label": lab,
                "similarity": np.nan,
                "was_canonicalized": False,
            }
            for lab in raw_labels
        ]
        return raw_labels, details, ""

    emb_mat = _normalize_matrix(vocab_emb)
    queries = model.encode(raw_labels, convert_to_numpy=True).astype("float32")
    queries = _normalize_matrix(queries)

    canonical = []
    details = []
    messages = []

    sims = queries @ emb_mat.T

    for i, lab in enumerate(raw_labels):
        best_idx = int(np.argmax(sims[i]))
        best_score = float(sims[i, best_idx])
        best_label = str(vocab_df.loc[best_idx, "tag_label"]).strip()

        if best_score >= sim_threshold:
            stored_label = best_label
            was_canonicalized = best_label.lower() != lab.lower()
            if was_canonicalized:
                messages.append(f'"{lab}" → "{best_label}" (sim={best_score:.2f})')
        else:
            stored_label = lab
            was_canonicalized = False

        canonical.append(stored_label)
        details.append(
            {
                "original_label": lab,
                "canonical_label": stored_label,
                "similarity": best_score,
                "was_canonicalized": was_canonicalized,
            }
        )

    msg = "Suggested anchors: " + "; ".join(messages) if messages else ""
    return canonical, details, msg

# ============================================================
# GNN PREDICTIONS
# ============================================================

def load_gnn_predictions() -> pd.DataFrame:
    if TAG_GNN_PRED_PATH.exists():
        try:
            return pd.read_csv(TAG_GNN_PRED_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["raw_tag_label", "predicted_control_tag", "score", "rank"])


def get_gnn_predictions_for_label(label: str, gnn_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if gnn_df.empty:
        return pd.DataFrame(columns=["predicted_control_tag", "score", "rank"])
    out = gnn_df[gnn_df["raw_tag_label"].astype(str).str.strip().str.lower() == str(label).strip().lower()].copy()
    if out.empty:
        return out
    return out.sort_values(["rank", "score"], ascending=[True, False]).head(top_k)

# ============================================================
# USER-TAG SUGGESTIONS
# ============================================================

@st.cache_data
def load_user_tags_for_suggestions():
    tags_df = load_user_tags()
    assign_df = load_tag_assignments()

    if tags_df.empty:
        return pd.DataFrame(columns=["tag_label", "parent_tag_label", "n_assignments"])

    tmp = tags_df.copy()
    tmp["tag_label"] = tmp["tag_label"].astype(str).str.strip()
    tmp["parent_tag_label"] = tmp["parent_tag_label"].fillna("").astype(str).str.strip()

    assign_counts = (
        assign_df.groupby("tag_label").size().to_dict()
        if not assign_df.empty else {}
    )

    tmp["n_assignments"] = tmp["tag_label"].map(assign_counts).fillna(0).astype(int)
    tmp = tmp[tmp["tag_label"] != ""].drop_duplicates(subset=["tag_label"]).reset_index(drop=True)
    return tmp


@st.cache_resource
def build_user_tag_embedding_index(user_tags_df: pd.DataFrame, _model):
    if user_tags_df is None or user_tags_df.empty:
        return None, None
    labels = user_tags_df["tag_label"].astype(str).tolist()
    emb = _model.encode(labels, convert_to_numpy=True).astype("float32")
    emb = _normalize_matrix(emb)
    return labels, emb


def suggest_user_tags_for_text(
    text: str,
    model,
    user_tags_df: pd.DataFrame,
    user_tag_labels,
    user_tag_emb: np.ndarray | None,
    top_k: int = 5,
    sim_threshold: float = 0.35,
):
    if not text or user_tags_df is None or user_tags_df.empty or user_tag_emb is None:
        return []

    q = model.encode(text, convert_to_numpy=True).astype("float32")
    q = q / (np.linalg.norm(q) + 1e-8)

    sims = user_tag_emb @ q
    top_idx = np.argsort(sims)[::-1][:top_k]

    suggestions = []
    for idx in top_idx:
        score = float(sims[idx])
        if score < sim_threshold:
            continue
        row = user_tags_df.iloc[int(idx)]
        suggestions.append(
            {
                "tag_label": row["tag_label"],
                "score": score,
                "n_assignments": int(row.get("n_assignments", 0)),
                "parent_tag_label": row.get("parent_tag_label", ""),
            }
        )
    return suggestions


def suggest_tags_for_typed_label(
    typed_label: str,
    model,
    user_tags_df: pd.DataFrame,
    user_tag_labels,
    user_tag_emb: np.ndarray | None,
    vocab_df: pd.DataFrame | None,
    vocab_emb: np.ndarray | None,
    top_k_user: int = 5,
    top_k_controlled: int = 5,
    sim_threshold_user: float = 0.35,
    sim_threshold_controlled: float = 0.45,
):
    if not typed_label or not typed_label.strip():
        return {"user": [], "controlled": []}

    q = model.encode(typed_label.strip(), convert_to_numpy=True).astype("float32")
    q = q / (np.linalg.norm(q) + 1e-8)

    out = {"user": [], "controlled": []}

    if user_tags_df is not None and not user_tags_df.empty and user_tag_emb is not None:
        sims_user = user_tag_emb @ q
        top_idx = np.argsort(sims_user)[::-1][:top_k_user]
        for idx in top_idx:
            score = float(sims_user[idx])
            if score < sim_threshold_user:
                continue
            row = user_tags_df.iloc[int(idx)]
            out["user"].append(
                {
                    "tag_label": row["tag_label"],
                    "score": score,
                    "n_assignments": int(row.get("n_assignments", 0)),
                }
            )

    if vocab_df is not None and vocab_emb is not None and not vocab_df.empty:
        emb_mat = _normalize_matrix(vocab_emb)
        sims_ctrl = emb_mat @ q
        top_idx = np.argsort(sims_ctrl)[::-1][:top_k_controlled]
        for idx in top_idx:
            score = float(sims_ctrl[idx])
            if score < sim_threshold_controlled:
                continue
            row = vocab_df.iloc[int(idx)]
            out["controlled"].append(
                {
                    "tag_label": row["tag_label"],
                    "score": score,
                    "facet": row["facet"] if "facet" in row.index else None,
                    "categories": row["categories"] if "categories" in row.index else None,
                }
            )

    return out

# ============================================================
# HYBRID GRAPH HELPERS
# ============================================================

def expand_tags_via_kg(selected_tags: list[str], kg_edges_df: pd.DataFrame, allowed_buckets: list[str]) -> tuple[set[str], dict[str, list[str]]]:
    if kg_edges_df is None or kg_edges_df.empty or not selected_tags:
        return set(selected_tags), {}

    expanded = set(selected_tags)
    mapping = defaultdict(list)

    for tag in selected_tags:
        tmp = kg_edges_df[
            (kg_edges_df["src_label"] == tag) | (kg_edges_df["dst_label"] == tag)
        ].copy()

        if tmp.empty:
            continue

        for _, row in tmp.iterrows():
            neigh = row["dst_label"] if row["src_label"] == tag else row["src_label"]
            bucket = _relation_bucket(row.get("relation", "related"))
            if bucket in allowed_buckets:
                expanded.add(neigh)
                mapping[tag].append(f"{neigh} [{bucket}]")

    return expanded, dict(mapping)


def build_used_tag_subgraph(
    assign_df: pd.DataFrame,
    user_tags_df: pd.DataFrame,
    vocab_df: pd.DataFrame | None,
    kg_edges_df: pd.DataFrame | None,
    canon_df: pd.DataFrame | None,
):
    used_labels = set()
    if not assign_df.empty:
        used_labels |= set(assign_df["tag_label"].dropna().astype(str).str.strip().tolist())
    if canon_df is not None and not canon_df.empty:
        used_labels |= set(canon_df["original_label"].dropna().astype(str).str.strip().tolist())
        used_labels |= set(canon_df["canonical_label"].dropna().astype(str).str.strip().tolist())

    if not used_labels:
        return pd.DataFrame(), pd.DataFrame()

    assign_counts = (
        assign_df.groupby("tag_label").size().to_dict()
        if not assign_df.empty else {}
    )
    entry_counts = (
        canon_df.groupby("original_label").size().to_dict()
        if canon_df is not None and not canon_df.empty else {}
    )

    user_label_set = set(user_tags_df["tag_label"].astype(str).str.strip().tolist()) if not user_tags_df.empty else set()
    vocab_label_set = set(vocab_df["tag_label"].astype(str).str.strip().tolist()) if vocab_df is not None and not vocab_df.empty else set()

    nodes = []
    for label in sorted(used_labels):
        if label in vocab_label_set and label in user_label_set:
            node_type = "hybrid"
        elif label in vocab_label_set:
            node_type = "controlled"
        elif label in user_label_set:
            node_type = "user"
        else:
            node_type = "other"

        nodes.append(
            {
                "label": label,
                "node_type": node_type,
                "n_assignments": int(assign_counts.get(label, 0)),
                "n_entries": int(entry_counts.get(label, 0)),
            }
        )

    nodes_df = pd.DataFrame(nodes)
    nodes_df = nodes_df.reset_index(drop=True)
    nodes_df.insert(0, "node_id", nodes_df.index.astype(int))

    edges = []

    if not user_tags_df.empty:
        tmp = user_tags_df.copy()
        tmp["tag_label"] = tmp["tag_label"].astype(str).str.strip()
        tmp["parent_tag_label"] = tmp["parent_tag_label"].fillna("").astype(str).str.strip()
        for _, row in tmp.iterrows():
            child = row["tag_label"]
            parent = row["parent_tag_label"]
            if child in used_labels and parent and parent in used_labels:
                edges.append(
                    {"src_label": parent, "dst_label": child, "relation": "user_parent", "weight": 1.0}
                )

    if kg_edges_df is not None and not kg_edges_df.empty:
        kg_sub = kg_edges_df[
            kg_edges_df["src_label"].isin(used_labels) & kg_edges_df["dst_label"].isin(used_labels)
        ].copy()
        if not kg_sub.empty:
            edges.extend(kg_sub.to_dict("records"))

    if canon_df is not None and not canon_df.empty:
        tmp = (
            canon_df.groupby(["original_label", "canonical_label"]).size().reset_index(name="weight")
        )
        for _, row in tmp.iterrows():
            a = str(row["original_label"]).strip()
            b = str(row["canonical_label"]).strip()
            if a in used_labels and b in used_labels and a != b:
                edges.append(
                    {"src_label": a, "dst_label": b, "relation": "canonicalized_to", "weight": float(row["weight"])}
                )

    if not assign_df.empty:
        tmp = assign_df.copy()
        tmp["doc_key"] = (
            tmp["subreddit_key"].astype(str) + "||" +
            tmp["corpus_path"].astype(str) + "||" +
            tmp["row_id"].astype(str)
        )
        pair_counts = Counter()
        for _, g in tmp.groupby("doc_key"):
            labels = sorted(set(g["tag_label"].dropna().astype(str).str.strip()))
            if len(labels) < 2:
                continue
            for a, b in combinations(labels, 2):
                if a in used_labels and b in used_labels:
                    pair_counts[(a, b)] += 1
        for (a, b), wt in pair_counts.items():
            edges.append(
                {"src_label": a, "dst_label": b, "relation": "cooccurs_with", "weight": float(wt)}
            )

    edges_df = pd.DataFrame(edges)
    if edges_df.empty:
        return nodes_df, edges_df

    edges_df["weight"] = pd.to_numeric(edges_df["weight"], errors="coerce").fillna(1.0)
    edges_df = (
        edges_df.groupby(["src_label", "dst_label", "relation"], as_index=False)["weight"]
        .sum()
        .sort_values(["relation", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return nodes_df, edges_df


def add_semantic_neighbor_edges(nodes_df: pd.DataFrame, model, top_k: int = 3, sim_threshold: float = 0.55):
    if nodes_df.empty or len(nodes_df) < 2:
        return pd.DataFrame(columns=["src_label", "dst_label", "relation", "weight"])

    labels = nodes_df["label"].astype(str).tolist()
    emb = model.encode(labels, convert_to_numpy=True).astype("float32")
    emb = _normalize_matrix(emb)
    sims = emb @ emb.T

    edges = []
    for i, src in enumerate(labels):
        ranked = np.argsort(sims[i])[::-1]
        kept = 0
        for j in ranked:
            if i == j:
                continue
            score = float(sims[i, j])
            if score < sim_threshold:
                continue
            dst = labels[j]
            a, b = sorted([src, dst])
            edges.append(
                {"src_label": a, "dst_label": b, "relation": "semantic_neighbor", "weight": score}
            )
            kept += 1
            if kept >= top_k:
                break

    if not edges:
        return pd.DataFrame(columns=["src_label", "dst_label", "relation", "weight"])

    out = pd.DataFrame(edges).drop_duplicates(subset=["src_label", "dst_label", "relation"])
    return out.reset_index(drop=True)


def compute_tag_centrality_scores(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    if nodes_df.empty:
        return pd.DataFrame(columns=["label"])

    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        G.add_node(
            row["label"],
            node_type=row.get("node_type", "other"),
            n_assignments=float(row.get("n_assignments", 0)),
            n_entries=float(row.get("n_entries", 0)),
        )

    if not edges_df.empty:
        for _, row in edges_df.iterrows():
            G.add_edge(
                row["src_label"],
                row["dst_label"],
                relation=row.get("relation", "related"),
                weight=float(row.get("weight", 1.0)),
            )

    degree_c = nx.degree_centrality(G) if G.number_of_nodes() > 0 else {}
    betweenness_c = nx.betweenness_centrality(G, weight="weight") if G.number_of_nodes() > 0 else {}

    try:
        eigen_c = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigen_c = {n: 0.0 for n in G.nodes()}

    rows = []
    for n, attrs in G.nodes(data=True):
        n_assign = float(attrs.get("n_assignments", 0))
        n_entries = float(attrs.get("n_entries", 0))

        bridge_score = float(betweenness_c.get(n, 0.0))
        anchor_score = (
            0.35 * float(eigen_c.get(n, 0.0)) +
            0.25 * float(degree_c.get(n, 0.0)) +
            0.25 * np.log1p(n_assign) +
            0.15 * np.log1p(n_entries)
        )

        rows.append(
            {
                "label": n,
                "node_type": attrs.get("node_type", "other"),
                "degree_centrality": float(degree_c.get(n, 0.0)),
                "betweenness_centrality": float(betweenness_c.get(n, 0.0)),
                "eigenvector_centrality": float(eigen_c.get(n, 0.0)),
                "n_assignments": n_assign,
                "n_entries": n_entries,
                "anchor_score": float(anchor_score),
                "bridge_score": float(bridge_score),
            }
        )

    return pd.DataFrame(rows).sort_values("anchor_score", ascending=False).reset_index(drop=True)


def get_graph_neighborhood(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, focus_label: str | None, hops: int = 1):
    if nodes_df.empty:
        return nodes_df, edges_df

    if not focus_label or focus_label not in set(nodes_df["label"].astype(str)):
        return nodes_df, edges_df

    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["label"])
    for _, row in edges_df.iterrows():
        G.add_edge(row["src_label"], row["dst_label"])

    keep = {focus_label}
    frontier = {focus_label}
    for _ in range(max(1, hops)):
        next_frontier = set()
        for n in frontier:
            if n in G:
                next_frontier |= set(G.neighbors(n))
        keep |= next_frontier
        frontier = next_frontier

    sub_nodes = nodes_df[nodes_df["label"].isin(keep)].copy()
    sub_edges = edges_df[
        edges_df["src_label"].isin(keep) & edges_df["dst_label"].isin(keep)
    ].copy()

    return sub_nodes.reset_index(drop=True), sub_edges.reset_index(drop=True)

# ============================================================
# CYTOSCAPE RENDERER
# ============================================================

def render_cytoscape_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, centrality_df: pd.DataFrame | None = None, height: int = 700):
    if nodes_df.empty:
        st.caption("No graph to render.")
        return

    nodes_df = nodes_df.copy()
    if centrality_df is not None and not centrality_df.empty:
        nodes_df = nodes_df.merge(
            centrality_df[["label", "anchor_score", "bridge_score", "degree_centrality"]],
            on="label",
            how="left",
        )

    elements = []

    for _, row in nodes_df.iterrows():
        node_type = row.get("node_type", "other")
        anchor_score = float(row.get("anchor_score", 0.0) or 0.0)
        bridge_score = float(row.get("bridge_score", 0.0) or 0.0)
        size = 28 + 55 * min(anchor_score, 1.0)

        elements.append(
            {
                "data": {
                    "id": row["label"],
                    "label": row["label"],
                    "node_type": node_type,
                    "n_assignments": int(row.get("n_assignments", 0)),
                    "n_entries": int(row.get("n_entries", 0)),
                    "anchor_score": round(anchor_score, 4),
                    "bridge_score": round(bridge_score, 4),
                    "size": size,
                }
            }
        )

    for i, row in edges_df.reset_index(drop=True).iterrows():
        rel = str(row.get("relation", "related"))
        elements.append(
            {
                "data": {
                    "id": f"e{i}",
                    "source": row["src_label"],
                    "target": row["dst_label"],
                    "relation": rel,
                    "weight": float(row.get("weight", 1.0)),
                }
            }
        )

    html = f"""
    <html>
    <head>
      <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
      <style>
        html, body, #cy {{
          width: 100%;
          height: {height}px;
          margin: 0;
          padding: 0;
          background: white;
          font-family: Arial, sans-serif;
        }}
      </style>
    </head>
    <body>
      <div id="cy"></div>
      <script>
        const elements = {json.dumps(elements)};
        const cy = cytoscape({{
          container: document.getElementById('cy'),
          elements: elements,
          style: [
            {{
              selector: 'node',
              style: {{
                'label': 'data(label)',
                'width': 'data(size)',
                'height': 'data(size)',
                'font-size': '10px',
                'text-wrap': 'wrap',
                'text-max-width': '110px',
                'text-valign': 'center',
                'text-halign': 'center',
                'color': '#111',
                'background-color': '#999',
                'border-width': 2,
                'border-color': '#555'
              }}
            }},
            {{
              selector: 'node[node_type = "controlled"]',
              style: {{
                'background-color': '#1f77b4'
              }}
            }},
            {{
              selector: 'node[node_type = "user"]',
              style: {{
                'background-color': '#ff7f0e'
              }}
            }},
            {{
              selector: 'node[node_type = "hybrid"]',
              style: {{
                'background-color': '#17a2b8'
              }}
            }},
            {{
              selector: 'node[node_type = "other"]',
              style: {{
                'background-color': '#2ca02c'
              }}
            }},
            {{
              selector: 'edge',
              style: {{
                'curve-style': 'bezier',
                'width': 'mapData(weight, 0, 10, 1, 6)',
                'line-color': '#bbb',
                'opacity': 0.75,
                'label': 'data(relation)',
                'font-size': '8px',
                'text-rotation': 'autorotate',
                'text-background-opacity': 1,
                'text-background-color': '#fff',
                'text-background-padding': '1px'
              }}
            }},
            {{
              selector: 'edge[relation = "semantic_neighbor"]',
              style: {{
                'line-color': '#9467bd'
              }}
            }},
            {{
              selector: 'edge[relation = "cooccurs_with"]',
              style: {{
                'line-color': '#2ca02c'
              }}
            }},
            {{
              selector: 'edge[relation = "canonicalized_to"]',
              style: {{
                'line-color': '#d62728'
              }}
            }},
            {{
              selector: 'edge[relation = "user_parent"]',
              style: {{
                'line-color': '#8c564b'
              }}
            }}
          ],
          layout: {{
            name: 'cose',
            animate: false,
            fit: true,
            padding: 20
          }}
        }});
      </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=False)

# ============================================================
# CACHED LOADERS
# ============================================================

def _resolve_corpus_path(candidates):
    """First existing corpus file. Candidates are relative to config.INDEX_DIR;
    absolute paths and legacy CWD-relative paths are honoured too."""
    for p in candidates:
        for cand in (INDEX_DIR / p, Path(p)):
            if cand.exists():
                return str(cand)
    raise FileNotFoundError(
        f"None of the corpus paths exist under {INDEX_DIR}: {candidates}"
    )


@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data
def load_corpus(corpus_path: str):
    df = pd.read_csv(corpus_path)

    if "text" not in df.columns:
        if "proposition" in df.columns:
            df["text"] = df["proposition"].astype(str)
        elif "body" in df.columns:
            df["text"] = df["body"].astype(str)
        else:
            df["text"] = ""

    if "author" not in df.columns and "user" in df.columns:
        df["author"] = df["user"]
    if "author" not in df.columns:
        df["author"] = "[unknown]"

    df["text"] = df["text"].fillna("")
    df["text_len"] = df["text"].astype(str).str.len().astype("int32")
    df["text_lower"] = df["text"].astype(str).str.lower()

    time_col = None
    for cand in ["created_utc", "created_utc_x", "created_utc_y", "created", "timestamp"]:
        if cand in df.columns:
            time_col = cand
            break

    created_dt = None
    if time_col is not None:
        col = df[time_col]
        numeric_col = pd.to_numeric(col, errors="coerce")
        num_non_na = numeric_col.notna().sum()

        if num_non_na > 0:
            max_val = numeric_col.max()
            unit = "s"
            if max_val > 1e11:
                unit = "ms"
            created_dt = pd.to_datetime(numeric_col, unit=unit, errors="coerce", utc=True)
        else:
            created_dt = pd.to_datetime(col.astype(str), errors="coerce", utc=True)

    if created_dt is not None:
        df["created_dt"] = created_dt
        df["year"] = df["created_dt"].dt.year
        df["year_month"] = df["created_dt"].dt.to_period("M").astype(str)
    else:
        df["created_dt"] = pd.NaT
        df["year"] = np.nan
        df["year_month"] = ""

    if "row_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_id"})

    return df


@st.cache_resource
def build_tfidf_index(df: pd.DataFrame):
    n_docs = len(df)
    if n_docs == 0 or n_docs > TFIDF_MAX_DOCS:
        return None, None

    texts = df["text"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


@st.cache_resource
def build_bm25_index(df: pd.DataFrame):
    n_docs = len(df)
    if n_docs == 0 or n_docs > BM25_MAX_DOCS:
        return None

    docs_tokens = []
    doc_lens = []
    df_counts = Counter()

    for text in df["text_lower"].astype(str).tolist():
        toks = re.findall(r"[a-z0-9_]+", text)
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
        docs_tokens.append(toks)
        doc_lens.append(len(toks))
        for t in set(toks):
            df_counts[t] += 1

    N = len(docs_tokens)
    avgdl = np.mean(doc_lens) if N > 0 else 0.0

    idf = {}
    for term, df_t in df_counts.items():
        idf_val = np.log(1 + (N - df_t + 0.5) / (df_t + 0.5))
        idf[term] = float(idf_val)

    doc_term_freqs = []
    for toks in docs_tokens:
        doc_term_freqs.append(Counter(toks))

    return {
        "docs_tokens": docs_tokens,
        "doc_lens": np.array(doc_lens, dtype="float32"),
        "avgdl": float(avgdl),
        "idf": idf,
        "doc_term_freqs": doc_term_freqs,
        "N": N,
    }


@st.cache_data
def get_overview_stats(df: pd.DataFrame):
    df = df.copy()

    docs_per_year = (
        df.dropna(subset=["created_dt"])
        .groupby("year")["row_id"]
        .count()
        .reset_index()
    )
    docs_per_year.columns = ["year", "n_docs"]

    docs_per_month = (
        df.dropna(subset=["created_dt"])
        .groupby("year_month")["row_id"]
        .count()
        .reset_index()
    )
    docs_per_month.columns = ["year_month", "n_docs"]

    score_col = "score" if "score" in df.columns else None
    if score_col:
        score_df = df[[score_col]].copy()
        score_df = score_df[score_df[score_col].notna()]
    else:
        score_df = pd.DataFrame(columns=["score"])

    vc = df["author"].fillna("[unknown]").value_counts().head(25)
    top_authors = pd.DataFrame({"author": vc.index, "n_docs": vc.values})

    topic_counts = pd.DataFrame()
    topic_time = pd.DataFrame()

    return docs_per_year, docs_per_month, score_df, top_authors, topic_counts, topic_time


def get_topic_stats(df: pd.DataFrame, topic_id_col: str, topic_label_col: str):
    if topic_id_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    tmp = df.copy()

    topic_counts = (
        tmp.groupby(topic_id_col)["row_id"]
        .count()
        .reset_index(name="n_docs")
    )

    if topic_label_col in tmp.columns:
        topic_counts = topic_counts.merge(
            tmp[[topic_id_col, topic_label_col]].drop_duplicates(),
            on=topic_id_col,
            how="left",
        )
    else:
        topic_counts[topic_label_col] = topic_counts[topic_id_col].apply(
            lambda x: f"{topic_id_col}_{x}"
        )

    if tmp["created_dt"].notna().any():
        topic_time = (
            tmp.dropna(subset=["created_dt"])
            .groupby(["year_month", topic_id_col])["row_id"]
            .count()
            .reset_index(name="n_docs")
        )
        if topic_label_col in tmp.columns:
            topic_time = topic_time.merge(
                tmp[[topic_id_col, topic_label_col]].drop_duplicates(),
                on=topic_id_col,
                how="left",
            )
        else:
            topic_time[topic_label_col] = topic_time[topic_id_col].apply(
                lambda x: f"{topic_id_col}_{x}"
            )
    else:
        topic_time = pd.DataFrame()

    return topic_counts, topic_time


@st.cache_data
def build_user_network(df: pd.DataFrame, max_nodes: int = 50):
    if "author" not in df.columns or "thread_id" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    G = nx.Graph()

    df_small = df.copy()
    if len(df_small) > 20_000:
        df_small = df_small.sample(20_000, random_state=42)

    grouped = df_small.groupby("thread_id")
    for _, g in grouped:
        users = g["author"].dropna().unique()
        if len(users) < 2:
            continue
        if len(users) > 20:
            users = users[:20]

        for u in users:
            if not G.has_node(u):
                G.add_node(u)
        for u, v in combinations(users, 2):
            if u == v:
                continue
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

    if G.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()

    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    deg = nx.degree_centrality(G)

    nodes_df = pd.DataFrame(
        [
            {
                "user": n,
                "eigenvector": eig.get(n, 0.0),
                "degree": deg.get(n, 0.0),
                "weighted_degree": sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True)),
            }
            for n in G.nodes()
        ]
    )

    nodes_df = nodes_df.sort_values("eigenvector", ascending=False).head(max_nodes)
    keep_users = set(nodes_df["user"])

    edges = []
    for u, v, d in G.edges(data=True):
        if u in keep_users and v in keep_users:
            edges.append({"source": u, "target": v, "weight": d.get("weight", 1.0)})

    edges_df = pd.DataFrame(edges)
    return nodes_df, edges_df


@st.cache_data
def build_topic_graph(
    df: pd.DataFrame,
    topic_id_col: str,
    topic_label_col: str,
    max_topics: int = 30,
):
    if topic_id_col not in df.columns:
        return None, None, None

    topic_counts = (
        df.groupby(topic_id_col)["row_id"].count().reset_index(name="n_docs")
    ).sort_values("n_docs", ascending=False)
    top_topics = topic_counts.head(max_topics)[topic_id_col].tolist()
    top_topics_set = set(top_topics)

    if topic_label_col in df.columns:
        topic_labels = (
            df[[topic_id_col, topic_label_col]]
            .drop_duplicates()
            .set_index(topic_id_col)[topic_label_col]
            .to_dict()
        )
    else:
        topic_labels = {tid: f"{topic_id_col}_{tid}" for tid in top_topics}

    author_topics = (
        df[df[topic_id_col].isin(top_topics_set)]
        .groupby("author")[topic_id_col]
        .apply(lambda x: list(set(x)))
    )

    G = nx.Graph()
    for tid in top_topics:
        size = int(topic_counts.loc[topic_counts[topic_id_col] == tid, "n_docs"].iloc[0])
        G.add_node(tid, size=size)

    for _, tlist in author_topics.items():
        if len(tlist) < 2:
            continue
        for a, b in combinations(tlist, 2):
            if a == b:
                continue
            if not (a in top_topics_set and b in top_topics_set):
                continue
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    if G.number_of_nodes() == 0:
        return None, None, None

    return G, topic_counts, topic_labels

# ============================================================
# FAISS + CONTEXT
# ============================================================

def embed_query(model, text: str) -> np.ndarray:
    emb = model.encode(text, convert_to_numpy=True)
    return emb.astype("float32")


def faiss_search(vector: np.ndarray, k: int = 5):
    payload = {"vector": vector.tolist(), "k": int(k)}
    resp = requests.post(FAISS_API_URL, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    indices = data.get("results") or data.get("indices")
    if indices is None:
        raise KeyError(f"FAISS response missing 'indices'/'results' keys: {data.keys()}")
    return indices[0], data["distances"][0]


def get_thread_context(df: pd.DataFrame, thread_id, exclude_row_id=None, limit=10):
    if "thread_id" not in df.columns:
        return pd.DataFrame()

    ctx = df[df["thread_id"] == thread_id].copy()
    if exclude_row_id is not None:
        ctx = ctx[ctx["row_id"] != exclude_row_id]

    if "created_dt" in ctx.columns:
        try:
            if "score" in ctx.columns:
                ctx = ctx.sort_values(["created_dt", "score"], ascending=[True, False])
            else:
                ctx = ctx.sort_values(["created_dt"], ascending=[True])
        except Exception:
            if "score" in ctx.columns:
                ctx = ctx.sort_values("score", ascending=False)
    elif "score" in ctx.columns:
        ctx = ctx.sort_values("score", ascending=False)

    return ctx.head(limit)


def build_rag_context(df: pd.DataFrame, indices, max_chars=3000):
    snippets = []
    total_len = 0
    for idx in indices:
        row = df.iloc[int(idx)]
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        meta = []
        if "author" in df.columns:
            meta.append(f"by {row.get('author', '[unknown]')}")
        if "subreddit" in df.columns:
            meta.append(f"r/{row.get('subreddit', '')}")
        if "year" in df.columns and not pd.isna(row.get("year")):
            meta.append(str(int(row.get("year"))))
        meta_str = " · ".join([m for m in meta if m])

        chunk = text
        if meta_str:
            chunk = f"[{meta_str}]\n{chunk}"

        if total_len + len(chunk) > max_chars:
            break
        snippets.append(chunk)
        total_len += len(chunk)

    return "\n\n---\n\n".join(snippets)

# ============================================================
# LEXICAL HELPERS
# ============================================================

def bm25_search(query: str, bm25_index, k: int = 20):
    if bm25_index is None:
        return []

    q_terms = tokenize_query(query)
    if not q_terms:
        return []

    idf = bm25_index["idf"]
    doc_term_freqs = bm25_index["doc_term_freqs"]
    doc_lens = bm25_index["doc_lens"]
    avgdl = bm25_index["avgdl"]

    k1 = 1.5
    b = 0.75

    scores = np.zeros(len(doc_term_freqs), dtype="float32")

    for term in set(q_terms):
        term_idf = idf.get(term)
        if term_idf is None:
            continue
        for doc_idx, tf_map in enumerate(doc_term_freqs):
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            dl = doc_lens[doc_idx]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            scores[doc_idx] += term_idf * ((tf * (k1 + 1)) / denom)

    nonzero_idx = np.nonzero(scores)[0]
    if len(nonzero_idx) == 0:
        return []

    ranked = sorted(
        [(int(i), float(scores[i])) for i in nonzero_idx],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:k]


def boolean_and_search(query: str, df: pd.DataFrame, k: int = 50):
    q_terms = tokenize_query(query)
    if not q_terms:
        return []

    hits = []
    texts = df["text_lower"].astype(str).tolist()
    for idx, doc in enumerate(texts):
        ok = True
        for t in q_terms:
            if t not in doc:
                ok = False
                break
        if not ok:
            continue
        hits.append(idx)
        if len(hits) >= k * 5:
            break

    return hits

# ============================================================
# CONCEPT NETWORK
# ============================================================

def build_concept_network_figure(texts, max_nodes=30, min_edge_weight=1):
    tokens_per_doc = []
    global_counts = Counter()

    for t in texts:
        if not isinstance(t, str):
            continue
        toks = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_]+", t)]
        toks = [w for w in toks if w not in STOPWORDS and len(w) > 2]
        if not toks:
            continue
        toks = list(set(toks))
        tokens_per_doc.append(toks)
        global_counts.update(toks)

    if not global_counts:
        return None

    top_terms = {w for w, _ in global_counts.most_common(max_nodes)}

    G = nx.Graph()
    for w in top_terms:
        G.add_node(w, freq=global_counts[w])

    for toks in tokens_per_doc:
        toks = [w for w in toks if w in top_terms]
        for a, b in combinations(toks, 2):
            if a == b:
                continue
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    to_drop = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("weight", 0) < min_edge_weight
    ]
    G.remove_edges_from(to_drop)

    if len(G.nodes) == 0:
        return None

    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)

    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    degrees = dict(G.degree())
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} (freq={global_counts[node]}, deg={degrees[node]})")
        node_size.append(10 + 2 * np.log1p(global_counts[node]))
        node_color.append(degrees[node])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(width=1),
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        hovertext=node_text,
        hoverinfo="text",
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Degree"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=30, b=10),
        title="Concept co-occurrence network (current results)",
    )
    return fig

# ============================================================
# UI TABS
# ============================================================

def tab_semantic_search(
    df: pd.DataFrame,
    model,
    tfidf_vec,
    tfidf_mat,
    bm25_index,
    subreddit_key: str,
    corpus_path: str,
    vocab_df,
    vocab_emb,
    kg_edges_df,
):
    st.subheader("Semantic & Lexical Search")

    if "search_triggered" not in st.session_state:
        st.session_state["search_triggered"] = False
    if "selected_graph_tag" not in st.session_state:
        st.session_state["selected_graph_tag"] = None
    if "tag_search_from_graph" not in st.session_state:
        st.session_state["tag_search_from_graph"] = None

    user_tags_df = load_user_tags()
    tag_assign_df = load_tag_assignments()
    canon_df = load_canonicalization_events()

    user_tags_suggest_df = load_user_tags_for_suggestions()
    user_tag_labels, user_tag_emb = build_user_tag_embedding_index(user_tags_suggest_df, model)

    with st.expander("Browse by tags", expanded=False):
        available_tags = set()
        if not user_tags_df.empty:
            available_tags |= set(user_tags_df["tag_label"].dropna().astype(str).str.strip().tolist())
        if vocab_df is not None and not vocab_df.empty:
            available_tags |= set(vocab_df["tag_label"].dropna().astype(str).str.strip().tolist())

        all_tags = sorted([t for t in available_tags if t])

        graph_seed = st.session_state.get("tag_search_from_graph")
        if graph_seed and graph_seed not in all_tags:
            all_tags = sorted(all_tags + [graph_seed])

        default_selected = [graph_seed] if graph_seed else []

        selected_tags = st.multiselect(
            "Filter documents by selected tag(s)",
            all_tags,
            default=default_selected,
            key=f"tag_search_multiselect_{subreddit_key}",
        )

        expand_with_kg = st.checkbox(
            "Expand selected tags via logical KG relations",
            value=False,
            key=f"expand_with_kg_{subreddit_key}",
        )

        allowed_buckets = st.multiselect(
            "KG relations to expand through",
            ["equivalence", "broader", "narrower", "related", "cooccurs_with"],
            default=["equivalence", "broader", "narrower", "related"],
            key=f"kg_expand_rel_{subreddit_key}",
        )

        if selected_tags:
            effective_tags = set(selected_tags)
            kg_map = {}

            if expand_with_kg and kg_edges_df is not None and not kg_edges_df.empty:
                effective_tags, kg_map = expand_tags_via_kg(selected_tags, kg_edges_df, allowed_buckets)

            if kg_map:
                st.caption(
                    "KG expansion: " +
                    " | ".join(f"{k}: {', '.join(v)}" for k, v in kg_map.items())
                )

            mask = (
                (tag_assign_df["subreddit_key"] == subreddit_key)
                & (tag_assign_df["corpus_path"] == corpus_path)
                & (tag_assign_df["tag_label"].isin(list(effective_tags)))
            )
            matched_assign = tag_assign_df[mask]
            if matched_assign.empty:
                st.info("No documents currently have those tags.")
            else:
                row_ids = matched_assign["row_id"].unique().tolist()
                df_tag = df[df["row_id"].isin(row_ids)].copy()
                st.write(
                    f"Found **{len(df_tag):,}** documents with effective tags: "
                    + ", ".join(sorted(effective_tags))
                )
                for _, r in df_tag.head(50).iterrows():
                    st.markdown("---")
                    hdr = f"`row_id={int(r['row_id'])}` · `{r.get('author','[unknown]')}`"
                    if "created_dt" in r and not pd.isna(r["created_dt"]):
                        hdr += f" · `{r['created_dt']}`"
                    st.markdown(hdr)
                    st.markdown(r.get("text", ""))

    with st.expander("Browse tag graph", expanded=False):
        nodes_df, edges_df = build_used_tag_subgraph(
            tag_assign_df, user_tags_df, vocab_df, kg_edges_df, canon_df
        )

        if nodes_df.empty:
            st.caption("No used tag graph yet.")
        else:
            include_semantic_neighbors = st.checkbox(
                "Include semantic-neighbor edges",
                value=True,
                key=f"include_semantic_neighbors_{subreddit_key}",
            )

            if include_semantic_neighbors:
                semantic_edges_df = add_semantic_neighbor_edges(nodes_df, model, top_k=3, sim_threshold=0.55)
                if not semantic_edges_df.empty:
                    edges_df = pd.concat([edges_df, semantic_edges_df], ignore_index=True)
                    edges_df = (
                        edges_df.groupby(["src_label", "dst_label", "relation"], as_index=False)["weight"]
                        .sum()
                        .reset_index(drop=True)
                    )

            all_relations = sorted(edges_df["relation"].astype(str).unique().tolist()) if not edges_df.empty else []
            relation_filter = st.multiselect(
                "Relations to show",
                all_relations,
                default=all_relations,
                key=f"graph_relation_filter_{subreddit_key}",
            )

            if relation_filter:
                edges_df = edges_df[edges_df["relation"].astype(str).isin(relation_filter)].copy()

            centrality_df = compute_tag_centrality_scores(nodes_df, edges_df)

            focus_options = sorted(nodes_df["label"].astype(str).tolist())
            current_focus = st.session_state.get("selected_graph_tag")
            if current_focus not in focus_options:
                current_focus = focus_options[0] if focus_options else None

            col_graph_a, col_graph_b, col_graph_c = st.columns([2, 1, 1])
            with col_graph_a:
                selected_graph_tag = st.selectbox(
                    "Focus tag",
                    focus_options,
                    index=focus_options.index(current_focus) if current_focus in focus_options else 0,
                    key=f"selected_graph_tag_{subreddit_key}",
                )
            with col_graph_b:
                hops = st.selectbox("Neighborhood hops", [1, 2], index=0, key=f"graph_hops_{subreddit_key}")
            with col_graph_c:
                if st.button("Use focused tag in retrieval", key=f"use_graph_tag_{subreddit_key}"):
                    st.session_state["tag_search_from_graph"] = selected_graph_tag

            st.session_state["selected_graph_tag"] = selected_graph_tag

            sub_nodes_df, sub_edges_df = get_graph_neighborhood(nodes_df, edges_df, selected_graph_tag, hops=hops)
            sub_centrality_df = compute_tag_centrality_scores(sub_nodes_df, sub_edges_df)

            st.caption(f"Nodes: {len(sub_nodes_df):,} · Edges: {len(sub_edges_df):,}")
            render_cytoscape_graph(sub_nodes_df, sub_edges_df, centrality_df=sub_centrality_df, height=700)

            st.markdown("**Top tags by anchor score**")
            st.dataframe(
                sub_centrality_df[[
                    "label",
                    "node_type",
                    "anchor_score",
                    "bridge_score",
                    "degree_centrality",
                    "betweenness_centrality",
                    "eigenvector_centrality",
                    "n_assignments",
                    "n_entries",
                ]].head(25),
                use_container_width=True,
            )

    with st.expander("Browse controlled vocabulary & KG", expanded=False):
        if vocab_df is None or vocab_emb is None or vocab_df.empty:
            st.caption(
                "Controlled vocabulary / KG not loaded. "
                "Make sure tags_vocab_cat.csv and tags_vocab_embeddings.npy exist in faiss/tagging."
            )
        else:
            query_tag = st.text_input(
                "Search controlled tags",
                key=f"kg_tag_search_{subreddit_key}",
                placeholder="e.g. AI & Reasoning, Action Theory",
            )
            if query_tag:
                mask = vocab_df["tag_label"].str.contains(query_tag, case=False, na=False)
                subset = vocab_df[mask].head(100)
            else:
                subset = vocab_df.head(100)

            options = subset["tag_label"].tolist()
            if options:
                selected_label = st.selectbox(
                    "Choose a controlled tag",
                    options,
                    key=f"kg_tag_select_{subreddit_key}",
                )
                if selected_label:
                    st.caption(f"Selected tag: `{selected_label}`")
                    if "facet" in vocab_df.columns:
                        facet_val = vocab_df.loc[
                            vocab_df["tag_label"] == selected_label, "facet"
                        ].iloc[0]
                        st.caption(f"Facet: `{facet_val}`")
                    if "categories" in vocab_df.columns:
                        cat_val = vocab_df.loc[
                            vocab_df["tag_label"] == selected_label, "categories"
                        ].iloc[0]
                        if pd.notna(cat_val):
                            st.caption(f"Categories: `{cat_val}`")

                    if kg_edges_df is not None and not kg_edges_df.empty:
                        neighbors = get_kg_neighbors(selected_label, kg_edges_df)
                        if neighbors:
                            st.markdown("**KG neighbors**")
                            for rel, neigh_list in neighbors.items():
                                st.markdown(f"- **{rel}**: " + ", ".join(neigh_list))
                        else:
                            st.caption("No KG edges stored for this tag yet.")
                    else:
                        st.caption("KG edge file not found; only vocab browsing is available.")
            else:
                st.info("No controlled tags match your search.")

    col_main, col_opts = st.columns([3, 1])

    with col_opts:
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            [
                "Dense (FAISS)",
                "Lexical (TF-IDF ranked)",
                "Lexical (Boolean AND filter)",
                "Lexical (BM25 probabilistic)",
                "Hybrid (FAISS ∩ TF-IDF)",
            ],
        )
        k_faiss = st.slider("k (FAISS neighbors)", min_value=1, max_value=50, value=10)
        k_lex = st.slider("k (lexical neighbors / max hits)", min_value=1, max_value=100, value=20)

        min_len = int(df["text_len"].min())
        max_len = int(df["text_len"].max())
        prop_len_range = st.slider(
            "Proposition length (characters)",
            min_value=min_len,
            max_value=max_len,
            value=(min_len, min(min_len + 1000, max_len)),
        )

        show_thread_ctx = st.checkbox("Show thread context", value=False)
        show_raw = st.checkbox("Show raw similarity/ids", value=False)
        min_sim = st.slider(
            "Min similarity (FAISS score)",
            min_value=-1.0,
            max_value=1.0,
            value=-1.0,
            step=0.05,
            help="Inner-product / cosine-ish: higher = more similar. Only used for FAISS/hybrid.",
        )

        show_concept_net = st.checkbox("Show concept network for these results", value=True)

    with col_main:
        query = st.text_area("Enter your query (semantic + lexical)", height=80)
        run = st.button("Search", type="primary")

        if run:
            st.session_state["search_triggered"] = True

    if not (st.session_state["search_triggered"] and query.strip()):
        st.info("Enter a query and click **Search** to run retrieval.")
        return

    query_str = query.strip()
    rows = []

    faiss_rows = []
    if retrieval_mode in ("Dense (FAISS)", "Hybrid (FAISS ∩ TF-IDF)"):
        with st.spinner("Embedding query and searching FAISS..."):
            try:
                q_vec = embed_query(model, query_str)
                indices, distances = faiss_search(q_vec, k=k_faiss)
            except Exception as e:
                st.error(f"Error during FAISS search: {e}")
                return

        for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
            row = df.iloc[int(idx)]
            if not (prop_len_range[0] <= row["text_len"] <= prop_len_range[1]):
                continue
            try:
                if dist < min_sim:
                    continue
            except Exception:
                pass
            faiss_rows.append((rank, idx, dist, row))

    tfidf_rows = []
    if retrieval_mode in ("Lexical (TF-IDF ranked)", "Hybrid (FAISS ∩ TF-IDF)"):
        if tfidf_vec is None or tfidf_mat is None:
            st.warning("TF-IDF index not available (corpus too large or not built).")
        else:
            with st.spinner("Running lexical retrieval (TF-IDF)..."):
                q_vec = tfidf_vec.transform([query_str])
                sims = cosine_similarity(q_vec, tfidf_mat).ravel()
                top_idx = np.argsort(sims)[::-1][:k_lex]
                for rank, idx in enumerate(top_idx, start=1):
                    score = sims[idx]
                    row = df.iloc[int(idx)]
                    if not (prop_len_range[0] <= row["text_len"] <= prop_len_range[1]):
                        continue
                    tfidf_rows.append((rank, idx, score, row))

    bool_rows = []
    if retrieval_mode == "Lexical (Boolean AND filter)":
        with st.spinner("Running Boolean AND filter over corpus..."):
            hit_indices = boolean_and_search(query_str, df, k=k_lex)
        for rank, idx in enumerate(hit_indices, start=1):
            row = df.iloc[int(idx)]
            if not (prop_len_range[0] <= row["text_len"] <= prop_len_range[1]):
                continue
            bool_rows.append((rank, idx, 1.0, row))
        bool_rows = bool_rows[:k_lex]

    bm25_rows = []
    if retrieval_mode == "Lexical (BM25 probabilistic)":
        if bm25_index is None:
            st.warning("BM25 index not available (corpus too large or not built).")
        else:
            with st.spinner("Running BM25 scoring over corpus..."):
                ranked = bm25_search(query_str, bm25_index, k=k_lex)
            for rank, (idx, score) in enumerate(ranked, start=1):
                row = df.iloc[int(idx)]
                if not (prop_len_range[0] <= row["text_len"] <= prop_len_range[1]):
                    continue
                bm25_rows.append((rank, idx, score, row))

    if retrieval_mode == "Dense (FAISS)":
        rows = faiss_rows
    elif retrieval_mode == "Lexical (TF-IDF ranked)":
        rows = tfidf_rows
    elif retrieval_mode == "Lexical (Boolean AND filter)":
        rows = bool_rows
    elif retrieval_mode == "Lexical (BM25 probabilistic)":
        rows = bm25_rows
    elif retrieval_mode == "Hybrid (FAISS ∩ TF-IDF)":
        faiss_ids = {int(r[1]) for r in faiss_rows}
        tfidf_ids = {int(r[1]) for r in tfidf_rows}
        common = faiss_ids.intersection(tfidf_ids)
        id_to_faiss = {int(r[1]): r for r in faiss_rows}
        id_to_tfidf = {int(r[1]): r for r in tfidf_rows}

        combined = []
        for idx in common:
            fr = id_to_faiss[idx]
            tr = id_to_tfidf[idx]
            dist = fr[2]
            tfidf_score = tr[2]
            combined_score = (dist + tfidf_score) / 2.0
            combined.append((idx, combined_score, fr[3]))

        combined.sort(key=lambda x: x[1], reverse=True)
        rows = []
        for rank, (idx, cscore, row) in enumerate(combined, start=1):
            rows.append((rank, idx, cscore, row))

    if not rows:
        st.warning("No results after applying filters/mode.")
        return

    try:
        st.session_state["last_search_ids"] = [int(r[3]["row_id"]) for r in rows]
        st.session_state["last_search_query"] = query_str
        st.session_state["last_search_mode"] = retrieval_mode
    except Exception:
        pass

    texts_for_concept_net = [r[3].get("text", "") for r in rows]
    if show_concept_net:
        fig = build_concept_network_figure(texts_for_concept_net)
        if fig is not None:
            st.markdown("### Concept Network for Current Results")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough lexical signal to build a concept network from these results.")

    gnn_df = load_gnn_predictions()

    for rank, idx, score_val, row in rows:
        st.markdown("---")
        header = f"**#{rank}** – by `{row.get('author', '[unknown]')}`"
        if "created_dt" in row and not pd.isna(row["created_dt"]):
            header += f" · `{row['created_dt']}`"
        if "subreddit" in df.columns:
            header += f" · r/{row.get('subreddit', '')}"
        st.markdown(header)

        st.markdown(row.get("text", ""))

        current_tags = get_tags_for_doc(subreddit_key, corpus_path, row["row_id"])
        if current_tags:
            st.caption("Tags: " + ", ".join(current_tags))
        else:
            st.caption("Tags: (none yet)")

        base_key = f"{subreddit_key}_{int(row['row_id'])}"

        with st.expander("Add tags to this proposition", expanded=False):
            tag_input_key = f"tag_input_{base_key}"
            parent_input_key = f"parent_tag_input_{base_key}"

            new_tags = st.text_input(
                "New tag(s), comma-separated",
                key=tag_input_key,
                placeholder="e.g. alignment, agency, epistemology",
            )
            parent_tag = st.text_input(
                "Parent tag (optional, for hierarchy)",
                key=parent_input_key,
                placeholder="e.g. AI ethics",
            )

            if new_tags and new_tags.strip():
                raw_preview = [t.strip() for t in new_tags.split(",") if t.strip()]
                for raw_lab in raw_preview:
                    bridge = suggest_tags_for_typed_label(
                        raw_lab,
                        model,
                        user_tags_suggest_df,
                        user_tag_labels,
                        user_tag_emb,
                        vocab_df,
                        vocab_emb,
                        top_k_user=3,
                        top_k_controlled=3,
                    )

                    if bridge["user"]:
                        st.caption(
                            f"Related user tags for '{raw_lab}': " +
                            "; ".join(f"{r['tag_label']} ({r['score']:.2f})" for r in bridge["user"])
                        )

                    if bridge["controlled"]:
                        st.caption(
                            f"Controlled anchors for '{raw_lab}': " +
                            "; ".join(f"{r['tag_label']} ({r['score']:.2f})" for r in bridge["controlled"])
                        )

                    gnn_preview = get_gnn_predictions_for_label(raw_lab, gnn_df, top_k=3)
                    if not gnn_preview.empty:
                        st.caption(
                            f"GNN anchor candidates for '{raw_lab}': " +
                            "; ".join(
                                f"{r['predicted_control_tag']} ({r['score']:.2f})"
                                for _, r in gnn_preview.iterrows()
                            )
                        )

            if st.button("Add tag(s)", key=f"add_tags_btn_{base_key}"):
                raw_tag_labels = [t.strip() for t in str(new_tags).split(",") if t.strip()]
                if not raw_tag_labels:
                    st.warning("No non-empty tags given.")
                else:
                    ok, msg = add_tags_to_doc(
                        subreddit_key,
                        corpus_path,
                        row,
                        ", ".join(raw_tag_labels),
                        parent_tag,
                    )

                    if ok:
                        st.success(msg)

                        if vocab_df is not None and vocab_emb is not None:
                            canon_rows = []
                            _, canon_details, canon_msg = canonicalize_input_labels_detailed(
                                ", ".join(raw_tag_labels),
                                vocab_df,
                                vocab_emb,
                                model,
                                sim_threshold=0.7,
                            )

                            for d in canon_details:
                                canon_rows.append(
                                    {
                                        "original_label": d["original_label"],
                                        "canonical_label": d["canonical_label"],
                                        "similarity": d["similarity"],
                                        "subreddit_key": subreddit_key,
                                        "corpus_path": corpus_path,
                                        "row_id": int(row["row_id"]),
                                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                                    }
                                )

                            if canon_rows:
                                append_canonicalization_events(canon_rows)

                            if canon_msg:
                                st.caption("Anchor suggestions logged: " + canon_msg)
                    else:
                        st.warning(msg)

        if vocab_df is not None and vocab_emb is not None and not vocab_df.empty:
            with st.expander("Suggested controlled tags (anchors)", expanded=False):
                suggestions = suggest_tags_for_text(
                    row.get("text", ""),
                    model,
                    vocab_df,
                    vocab_emb,
                    top_k=5,
                    sim_threshold=0.4,
                )
                if not suggestions:
                    st.caption("No strong controlled-vocab suggestions for this proposition.")
                else:
                    plain_labels = [s["tag_label"] for s in suggestions]
                    label_with_scores = [
                        f"{s['tag_label']} (sim={s['score']:.2f}"
                        + (f", facet={s['facet']}" if s["facet"] is not None and pd.notna(s["facet"]) else "")
                        + ")"
                        for s in suggestions
                    ]
                    selected = st.multiselect(
                        "Select controlled tags to apply",
                        options=plain_labels,
                        default=[],
                        key=f"suggested_tags_{base_key}",
                    )
                    st.caption("Top suggestions: " + "; ".join(label_with_scores))
                    if st.button("Apply selected controlled tags", key=f"apply_suggested_{base_key}"):
                        if selected:
                            ok2, msg2 = add_tags_to_doc(
                                subreddit_key,
                                corpus_path,
                                row,
                                ", ".join(selected),
                                parent_tag_text=None,
                            )
                            if ok2:
                                st.success(msg2)
                            else:
                                st.warning(msg2)
                        else:
                            st.info("No suggestions selected.")

        with st.expander("Suggested user tags (emergent / folksonomic)", expanded=False):
            user_suggestions = suggest_user_tags_for_text(
                row.get("text", ""),
                model,
                user_tags_suggest_df,
                user_tag_labels,
                user_tag_emb,
                top_k=5,
                sim_threshold=0.35,
            )

            if not user_suggestions:
                st.caption("No strong user-tag suggestions for this proposition yet.")
            else:
                plain_labels = [s["tag_label"] for s in user_suggestions]
                label_with_scores = [
                    f"{s['tag_label']} (sim={s['score']:.2f}, n={s['n_assignments']})"
                    for s in user_suggestions
                ]

                selected_user = st.multiselect(
                    "Select user tags to apply",
                    options=plain_labels,
                    default=[],
                    key=f"user_suggested_tags_{base_key}",
                )
                st.caption("Top user-tag suggestions: " + "; ".join(label_with_scores))

                if st.button("Apply selected user tags", key=f"apply_user_suggested_{base_key}"):
                    if selected_user:
                        ok3, msg3 = add_tags_to_doc(
                            subreddit_key,
                            corpus_path,
                            row,
                            ", ".join(selected_user),
                            parent_tag_text=None,
                        )
                        if ok3:
                            st.success(msg3)
                        else:
                            st.warning(msg3)
                    else:
                        st.info("No user-tag suggestions selected.")

        if show_raw and score_val is not None:
            st.caption(f"Similarity/score: {float(score_val):.4f} · row_id: {int(row['row_id'])}")

        if show_thread_ctx and "thread_id" in df.columns:
            thread_id = row.get("thread_id")
            if pd.notna(thread_id):
                ctx = get_thread_context(df, thread_id, exclude_row_id=row.get("row_id"))
                if not ctx.empty:
                    with st.expander("Thread context"):
                        for _, r2 in ctx.iterrows():
                            h2 = f"- `{r2.get('author', '[unknown]')}`"
                            if "created_dt" in r2 and not pd.isna(r2["created_dt"]):
                                h2 += f" · `{r2['created_dt']}`"
                            st.markdown(h2)
                            st.markdown(r2.get("text", ""))

    st.markdown("---")


def tab_rag(df, model):
    st.subheader("RAG-style Context Builder")

    st.markdown(
        "This tab retrieves context via FAISS and constructs a consolidated context block "
        "that you can feed into any LLM. The app stays lean and only handles retrieval + context construction."
    )

    k = st.slider("k (neighbors for context)", min_value=3, max_value=50, value=15)
    query = st.text_area("Retrieval query (for FAISS)", height=80)
    question = st.text_area("Question to ask about the retrieved context", height=80)
    max_chars = st.slider("Max context length (characters)", 1000, 8000, 4000, step=500)

    run = st.button("Build RAG Context", type="primary")

    if not (run and query.strip() and question.strip()):
        st.info("Fill both fields and click **Build RAG Context**.")
        return

    with st.spinner("Retrieving documents from FAISS..."):
        try:
            q_vec = embed_query(model, query.strip())
            indices, distances = faiss_search(q_vec, k=k)
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            return

    context_text = build_rag_context(df, indices, max_chars=max_chars)

    if len(context_text.strip()) < 50:
        st.warning("Retrieved context is too short; try increasing k or adjusting your query.")
        return

    st.markdown("### Retrieved context (copy-paste into your LLM)")
    st.code(context_text, language="markdown")

    st.markdown("### Question")
    st.write(question.strip())


def tab_analytics(df: pd.DataFrame):
    st.subheader("Analytics & Visualizations")

    (
        docs_per_year,
        docs_per_month,
        score_df,
        top_authors,
        _topic_counts_unused,
        _topic_time_unused,
    ) = get_overview_stats(df)

    nodes_df, edges_df = build_user_network(df)
    has_sentiment = "sentiment_vader" in df.columns

    topic_model_options = {}
    if "lda_topic_id" in df.columns:
        topic_model_options["LDA"] = ("lda_topic_id", "lda_topic_label")
    if "kmeans_topic_id" in df.columns:
        topic_model_options["K-Means"] = ("kmeans_topic_id", "kmeans_topic_label")
    if "bertopic_topic_id" in df.columns:
        topic_model_options["BERTopic"] = ("bertopic_topic_id", "bertopic_topic_label")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Volume over time")

        if not docs_per_month.empty:
            fig_month = px.line(docs_per_month, x="year_month", y="n_docs", title="Documents per month")
            fig_month.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_month, use_container_width=True)
        elif not docs_per_year.empty:
            fig_year = px.bar(
                docs_per_year,
                x="year",
                y="n_docs",
                title="Documents per year",
                color="n_docs",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("No timestamp information to plot docs over time.")

        st.markdown("#### Global sentiment over time")
        if has_sentiment and df["created_dt"].notna().any():
            sent_time = (
                df.dropna(subset=["created_dt", "sentiment_vader"])
                .groupby("year_month")["sentiment_vader"]
                .mean()
                .reset_index()
            )
            if not sent_time.empty:
                fig_sent = px.line(sent_time, x="year_month", y="sentiment_vader", title="Mean sentiment_vader per month")
                fig_sent.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.info("Not enough sentiment/time data to plot.")
        else:
            st.info("No sentiment_vader column or timestamps; cannot plot sentiment over time.")

        st.markdown("#### Top authors")
        if not top_authors.empty:
            fig_auth = px.bar(
                top_authors,
                x="author",
                y="n_docs",
                title="Top authors by document count",
                color="n_docs",
                color_continuous_scale="Viridis",
            )
            fig_auth.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_auth, use_container_width=True)
        else:
            st.info("No author data available.")

        st.markdown("#### User network (eigenvector + degree centrality)")
        if not nodes_df.empty and not edges_df.empty:
            G = nx.Graph()
            for _, row in nodes_df.iterrows():
                G.add_node(
                    row["user"],
                    eigen=row["eigenvector"],
                    degree=row["degree"],
                    wdeg=row["weighted_degree"],
                )
            for _, row in edges_df.iterrows():
                G.add_edge(row["source"], row["target"], weight=row["weight"])

            pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

            edge_x, edge_y = [], []
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
            for n, (x, y) in pos.items():
                node_x.append(x)
                node_y.append(y)
                data = G.nodes[n]
                eig = data.get("eigen", 0.0)
                deg = data.get("degree", 0.0)
                wdeg = data.get("wdeg", 0.0)
                node_text.append(f"{n}\nEigen={eig:.4f}, Degree={deg:.4f}, WDeg={wdeg:.1f}")
                node_size.append(10 + 30 * eig)
                node_color.append(wdeg)

            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line=dict(width=1))
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=list(G.nodes()),
                hovertext=node_text,
                hoverinfo="text",
                textposition="top center",
                marker=dict(
                    size=node_size,
                    color=node_color,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Weighted degree"),
                ),
            )

            fig_user = go.Figure(data=[edge_trace, node_trace])
            fig_user.update_layout(
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=30, b=10),
                title="User co-participation network (sampled)",
            )
            st.plotly_chart(fig_user, use_container_width=True)
        else:
            st.info("User network could not be built (missing thread_id or author data).")

    with col_right:
        st.markdown("#### Topic analytics")

        if not topic_model_options:
            st.info(
                "No topic-model columns found. "
                "Once you use the *_with_topics_labeled.csv corpora, topic analytics will show up here."
            )
            topic_counts = pd.DataFrame()
            topic_time = pd.DataFrame()
            topic_graph = None
            topic_counts_full = None
            topic_labels = {}
        else:
            model_names = list(topic_model_options.keys())
            chosen_model = st.selectbox("Topic model", model_names, index=0, key="topic_model_select")
            topic_id_col, topic_label_col = topic_model_options[chosen_model]

            topic_counts, topic_time = get_topic_stats(df, topic_id_col, topic_label_col)

            st.markdown(f"**Topic distribution – {chosen_model}**")
            if not topic_counts.empty:
                fig_topic = px.bar(
                    topic_counts.sort_values("n_docs", ascending=False),
                    x=topic_label_col,
                    y="n_docs",
                    title=f"{chosen_model} topic sizes",
                    color=topic_label_col,
                )
                fig_topic.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_topic, use_container_width=True)
            else:
                st.info(f"No topics found for {chosen_model}.")

            st.markdown(f"**Topics over time – {chosen_model}**")
            if not topic_time.empty:
                fig_ttime = px.line(
                    topic_time,
                    x="year_month",
                    y="n_docs",
                    color=topic_label_col,
                    title=f"{chosen_model} topic volume over time",
                )
                fig_ttime.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_ttime, use_container_width=True)
            else:
                st.info("No topic-time data (missing topics and/or timestamps).")

            if has_sentiment:
                st.markdown(f"**Topic sentiment (VADER) – {chosen_model}**")

                tmp_sent = df[[topic_id_col, topic_label_col, "sentiment_vader"]].dropna(
                    subset=[topic_id_col, "sentiment_vader"]
                ).copy()

                if not topic_counts.empty:
                    top_ids = topic_counts[topic_id_col].tolist()
                    tmp_sent = tmp_sent[tmp_sent[topic_id_col].isin(top_ids)]

                topic_sent = (
                    tmp_sent.groupby(topic_label_col)["sentiment_vader"]
                    .mean()
                    .reset_index()
                )

                if not topic_sent.empty:
                    fig_sent_topics = px.bar(
                        topic_sent,
                        x=topic_label_col,
                        y="sentiment_vader",
                        title=f"Average sentiment per topic ({chosen_model})",
                        color="sentiment_vader",
                        color_continuous_scale="RdBu",
                    )
                    fig_sent_topics.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig_sent_topics, use_container_width=True)
                else:
                    st.info("No sentiment data available for topics.")

                st.markdown(f"**Topic sentiment over time – {chosen_model}**")
                if df["created_dt"].notna().any() and not topic_counts.empty:
                    topic_counts_sorted = topic_counts.sort_values("n_docs", ascending=False)
                    labels = topic_counts_sorted[topic_label_col].tolist()
                    ids = topic_counts_sorted[topic_id_col].tolist()
                    selected_label = st.selectbox(
                        "Choose topic for sentiment-over-time",
                        labels,
                        index=0,
                        key="topic_sent_over_time_select",
                    )
                    selected_id = ids[labels.index(selected_label)]

                    df_topic = df[
                        (df[topic_id_col] == selected_id)
                        & df["created_dt"].notna()
                        & df["sentiment_vader"].notna()
                    ].copy()

                    if not df_topic.empty:
                        sent_topic_time = (
                            df_topic.groupby("year_month")["sentiment_vader"]
                            .mean()
                            .reset_index()
                        )
                        if not sent_topic_time.empty:
                            fig_tsent = px.line(
                                sent_topic_time,
                                x="year_month",
                                y="sentiment_vader",
                                title=f"Mean sentiment_vader for topic: {selected_label}",
                            )
                            fig_tsent.update_xaxes(tickangle=-45)
                            st.plotly_chart(fig_tsent, use_container_width=True)
                        else:
                            st.info("No sentiment/time data for this topic.")
                    else:
                        st.info("No documents with sentiment/time for this topic.")
                else:
                    st.info("Topic sentiment-over-time requires timestamps + sentiment_vader + topics.")
            else:
                st.info("No 'sentiment_vader' column found.")

            st.markdown("**Topic force graph (co-occurrence by author)**")
            topic_graph, topic_counts_full, topic_labels = build_topic_graph(
                df,
                topic_id_col=topic_id_col,
                topic_label_col=topic_label_col,
                max_topics=30,
            )

            if topic_graph is not None and topic_counts_full is not None:
                pos = nx.spring_layout(topic_graph, k=0.8, iterations=50, seed=42)

                edge_x, edge_y = [], []
                for u, v, d in topic_graph.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
                for n, (x, y) in pos.items():
                    node_x.append(x)
                    node_y.append(y)
                    size = int(topic_graph.nodes[n].get("size", 1))
                    label = topic_labels.get(n, f"{topic_id_col}_{n}")
                    node_text.append(f"{label} (n={size})")
                    node_size.append(10 + 2 * np.log1p(size))
                    node_color.append(size)

                edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line=dict(width=1))
                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=[topic_labels.get(n, f"{topic_id_col}_{n}") for n in topic_graph.nodes()],
                    hovertext=node_text,
                    hoverinfo="text",
                    textposition="top center",
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        colorscale="Plasma",
                        showscale=True,
                        colorbar=dict(title="Topic size"),
                    ),
                )

                fig_topic_net = go.Figure(data=[edge_trace, node_trace])
                fig_topic_net.update_layout(
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=f"{chosen_model} topic co-occurrence force graph",
                )
                st.plotly_chart(fig_topic_net, use_container_width=True)
            else:
                st.info("Topic graph not available yet.")

        st.markdown("#### Dynamic topics × sentiment – last semantic search")

        last_ids = st.session_state.get("last_search_ids")
        if last_ids is None:
            st.info("Run a search in the Semantic & Lexical Search tab to see a result snapshot.")
        else:
            df_sub = df[df["row_id"].isin(last_ids)].copy()
            q = st.session_state.get("last_search_query", "")
            mode = st.session_state.get("last_search_mode", "")

            st.caption(f"Last search: `{q}` (mode: {mode}) · {len(df_sub)} results")

            if topic_model_options and has_sentiment:
                if chosen_model in topic_model_options:
                    tid_col, tlabel_col = topic_model_options[chosen_model]
                else:
                    tid_col, tlabel_col = next(iter(topic_model_options.values()))

                if tid_col in df_sub.columns:
                    agg = (
                        df_sub.dropna(subset=["sentiment_vader"])
                        .groupby(tid_col)
                        .agg(
                            n_docs=("row_id", "count"),
                            mean_sent=("sentiment_vader", "mean"),
                        )
                        .reset_index()
                    )

                    if tlabel_col in df_sub.columns:
                        agg = agg.merge(
                            df_sub[[tid_col, tlabel_col]].drop_duplicates(),
                            on=tid_col,
                            how="left",
                        )
                    else:
                        agg[tlabel_col] = agg[tid_col].apply(lambda x: f"{tid_col}_{x}")

                    agg = agg[agg["n_docs"] >= 3]

                    if not agg.empty:
                        fig_dyn = px.scatter(
                            agg,
                            x="mean_sent",
                            y="n_docs",
                            size="n_docs",
                            color="mean_sent",
                            color_continuous_scale="RdBu",
                            hover_name=tlabel_col,
                            title=f"Topics in last search – sentiment vs frequency ({chosen_model})",
                            labels={"mean_sent": "mean sentiment_vader", "n_docs": "doc count"},
                        )
                        st.plotly_chart(fig_dyn, use_container_width=True)
                    else:
                        st.info("Last search results don't have enough docs per topic.")
                else:
                    st.info("Current topic model columns not present in last search subset.")
            else:
                st.info("Need topic model + sentiment_vader to build the dynamic topic × sentiment graph.")

# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="TagSynapse",
        layout="wide",
    )

    st.title("TagSynapse")
    st.caption("Structure without vocabulary homogenization - semantic + graph-mediated tagging over filtered Reddit corpora.")

    with st.sidebar:
        st.markdown("### Corpus selection")

        sub_keys = list(SUBREDDIT_CONFIG.keys())
        default_key = "artificial_filtered" if "artificial_filtered" in sub_keys else sub_keys[0]
        chosen_key = st.selectbox(
            "Choose subreddit corpus",
            options=sub_keys,
            format_func=lambda k: SUBREDDIT_CONFIG[k]["label"],
            index=sub_keys.index(default_key),
            key="subreddit_select",
        )

        cfg = SUBREDDIT_CONFIG[chosen_key]

        try:
            corpus_path = _resolve_corpus_path(cfg["corpus_candidates"])
            st.caption(f"Using corpus file:\n`{corpus_path}`")
        except FileNotFoundError:
            st.error(
                f"Corpus files for **{cfg['label']}** not found on server.\n\n"
                f"Expected one of (under `{INDEX_DIR}`, set TAGSYNAPSE_DATA_DIR to change):\n"
                + "\n".join(f"- `{p}`" for p in cfg["corpus_candidates"])
            )
            st.stop()

        st.markdown("---")

    df = load_corpus(corpus_path)
    model = load_model()
    tfidf_vec, tfidf_mat = build_tfidf_index(df)
    bm25_index = build_bm25_index(df)

    vocab_df, vocab_emb, kg_edges_df = load_controlled_vocab_and_kg()

    st.markdown(
        f"Corpus size: **{len(df):,}** propositions · "
        f"Authors: **{df['author'].nunique():,}**"
    )

    ts_coverage = df["created_dt"].notna().mean() if "created_dt" in df.columns else 0.0
    st.caption(f"Timestamps present for ~{ts_coverage*100:.1f}% of documents.")

    tab1, tab2, tab3 = st.tabs(
        ["Semantic & Lexical Search", "RAG Context", "Analytics"]
    )

    with tab1:
        tab_semantic_search(
            df,
            model,
            tfidf_vec,
            tfidf_mat,
            bm25_index,
            chosen_key,
            corpus_path,
            vocab_df,
            vocab_emb,
            kg_edges_df,
        )

    with tab2:
        tab_rag(df, model)

    with tab3:
        tab_analytics(df)


if __name__ == "__main__":
    main()