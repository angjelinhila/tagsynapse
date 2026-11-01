#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tag_kg.py

Construct a tag-centric Knowledge Graph from:

  - Controlled vocabulary:
      tags_vocab_cat.csv
      tags_vocab_embeddings.npy
      tags_edges_cat.csv (optional, if parseable)
  - User tags:
      user_tags.csv
      user_tag_assignments.csv

Outputs (written to config.TAG_DATA_DIR, default faiss/tagging/):

  - kg_nodes.csv:
        node_id (int)
        node_type (str: TAG_CONTROL, TAG_USER, PROP)
        label (str)
        is_controlled (0/1)
        source (str)         # e.g., 'FAST_subset', 'user'
        extra_id (str)       # original ids if available

  - kg_edges.csv:
        src_id (int)
        dst_id (int)
        relation (str)       # CONTROL_REL, USER_PARENT, USER_TO_CONTROL
        weight (float)
        source (str)         # 'controlled', 'user_hierarchy', 'embedding_map'

  - kg_prop_edges.csv:
        tag_id (int)
        prop_id (int)
        relation (str)       # TAGGED
        subreddit_key (str)
        corpus_path (str)
        row_id (int)

This is intended as the backing KG for TagSynapse + later GNNs.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> `tagsynapse` importable
from tagsynapse import config


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

# All paths come from tagsynapse/config.py (TAGSYNAPSE_DATA_DIR overrides).
TAG_DATA_DIR = config.TAG_DATA_DIR

CONTROL_VOCAB_PATH = TAG_DATA_DIR / "tags_vocab_cat.csv"
CONTROL_EMBED_PATH = TAG_DATA_DIR / "tags_vocab_embeddings.npy"
CONTROL_EDGES_PATH = TAG_DATA_DIR / "tags_edges_cat.csv"
KG_EDGES_INDEX_PATH = TAG_DATA_DIR / "kg_edges_index.csv"  # optional, may or may not be used

USER_TAGS_PATH = TAG_DATA_DIR / "user_tags.csv"
USER_ASSIGN_PATH = TAG_DATA_DIR / "user_tag_assignments.csv"

KG_NODES_OUT = TAG_DATA_DIR / "kg_nodes.csv"
KG_EDGES_OUT = TAG_DATA_DIR / "kg_edges.csv"
KG_PROP_EDGES_OUT = TAG_DATA_DIR / "kg_prop_edges.csv"

# Model for embedding USER tags; should match whatever you used for CONTROL embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------

def load_control_vocab() -> Tuple[pd.DataFrame, np.ndarray]:
    if not CONTROL_VOCAB_PATH.exists() or not CONTROL_EMBED_PATH.exists():
        raise FileNotFoundError(
            f"Missing controlled vocab or embeddings: "
            f"{CONTROL_VOCAB_PATH} or {CONTROL_EMBED_PATH}"
        )

    vocab_df = pd.read_csv(CONTROL_VOCAB_PATH)
    emb = np.load(CONTROL_EMBED_PATH).astype("float32")

    if len(vocab_df) != emb.shape[0]:
        print(
            f"[WARN] Controlled vocab mismatch: {len(vocab_df)} rows vs "
            f"{emb.shape[0]} embedding vectors. Truncating to min()."
        )
        n = min(len(vocab_df), emb.shape[0])
        vocab_df = vocab_df.iloc[:n].reset_index(drop=True)
        emb = emb[:n]

    # Heuristically pick a label column; normalize to 'label'
    label_candidates = [
        c for c in vocab_df.columns
        if c.lower() in ("label", "pref_label", "tag_label", "term", "name")
    ]
    if label_candidates:
        label_col = label_candidates[0]
    else:
        label_col = vocab_df.columns[0]

    vocab_df = vocab_df.copy()
    vocab_df["label"] = vocab_df[label_col].astype(str)

    # Optional external ID column if present
    id_candidates = [
        c for c in vocab_df.columns
        if c.lower() in ("id", "tag_id", "fast_id", "uri")
    ]
    extra_id_col = id_candidates[0] if id_candidates else None
    if extra_id_col is not None:
        vocab_df["extra_id"] = vocab_df[extra_id_col].astype(str)
    else:
        vocab_df["extra_id"] = ""

    return vocab_df, emb


def load_control_edges(control_nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Try to parse tags_edges_cat.csv into a simple edge list on labels.

    Expected to be forgiving: if we can't identify columns, return empty DataFrame.
    """
    if not CONTROL_EDGES_PATH.exists():
        print("[INFO] No tags_edges_cat.csv found; skipping CONTROL_REL edges.")
        return pd.DataFrame(columns=["src_label", "dst_label", "relation", "weight"])

    edges_raw = pd.read_csv(CONTROL_EDGES_PATH)
    if edges_raw.empty:
        return pd.DataFrame(columns=["src_label", "dst_label", "relation", "weight"])

    cols_lower = {c.lower(): c for c in edges_raw.columns}

    # Try label-based edges first (easiest to align)
    label_src_col = None
    label_dst_col = None
    for key in ["source_label", "src_label", "label_source", "from_label"]:
        if key in cols_lower:
            label_src_col = cols_lower[key]
            break
    for key in ["target_label", "dst_label", "label_target", "to_label"]:
        if key in cols_lower:
            label_dst_col = cols_lower[key]
            break

    # If no obvious label columns, bail out gracefully
    if label_src_col is None or label_dst_col is None:
        print(
            "[WARN] Unable to find label-based columns in tags_edges_cat.csv; "
            "CONTROL_REL edges will be skipped for now."
        )
        return pd.DataFrame(columns=["src_label", "dst_label", "relation", "weight"])

    rel_col = None
    for key in ["relation", "rel_type", "edge_type", "type"]:
        if key in cols_lower:
            rel_col = cols_lower[key]
            break

    weight_col = None
    for key in ["weight", "score", "w"]:
        if key in cols_lower:
            weight_col = cols_lower[key]
            break

    edges = pd.DataFrame()
    edges["src_label"] = edges_raw[label_src_col].astype(str)
    edges["dst_label"] = edges_raw[label_dst_col].astype(str)
    edges["relation"] = "CONTROL_REL"
    if rel_col is not None:
        # keep original relation type in a suffix if needed
        edges["relation"] = edges_raw[rel_col].astype(str)

    if weight_col is not None:
        edges["weight"] = pd.to_numeric(
            edges_raw[weight_col], errors="coerce"
        ).fillna(1.0)
    else:
        edges["weight"] = 1.0

    # Filter to edges whose labels exist in our controlled vocab
    labels = set(control_nodes["label"].tolist())
    mask = edges["src_label"].isin(labels) & edges["dst_label"].isin(labels)
    edges = edges[mask].reset_index(drop=True)

    print(f"[INFO] Loaded {len(edges)} CONTROL_REL edges from tags_edges_cat.csv")
    return edges


def load_user_tags() -> pd.DataFrame:
    if not USER_TAGS_PATH.exists():
        print("[INFO] No user_tags.csv found yet; treating as empty.")
        return pd.DataFrame(columns=["tag_id", "tag_label", "parent_tag_label"])

    df = pd.read_csv(USER_TAGS_PATH)
    df = df.copy()

    if "tag_label" not in df.columns:
        raise ValueError("user_tags.csv missing 'tag_label' column.")

    if "tag_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "tag_id"})

    if "parent_tag_label" not in df.columns:
        df["parent_tag_label"] = ""

    df["tag_id"] = pd.to_numeric(df["tag_id"], errors="coerce").fillna(0).astype(int)
    df["tag_label"] = df["tag_label"].astype(str)
    df["parent_tag_label"] = df["parent_tag_label"].fillna("").astype(str)

    return df


def load_user_assignments() -> pd.DataFrame:
    """
    Columns (from your Streamlit app):

      subreddit_key, corpus_path, doc_id, row_id, tag_label
    """
    if not USER_ASSIGN_PATH.exists():
        print("[INFO] No user_tag_assignments.csv found; treating as empty.")
        return pd.DataFrame(
            columns=[
                "subreddit_key",
                "corpus_path",
                "doc_id",
                "row_id",
                "tag_label",
            ]
        )
    df = pd.read_csv(USER_ASSIGN_PATH)
    df = df.copy()
    for col in ["subreddit_key", "corpus_path", "tag_label"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "row_id" in df.columns:
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype(int)
    else:
        df["row_id"] = -1
    if "doc_id" not in df.columns:
        df["doc_id"] = ""
    return df


def build_node_table(
    control_df: pd.DataFrame,
    user_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    include_props: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Build unified node table and label->node_id maps for control + user tags.

    Returns:
        nodes_df     – all nodes
        control_map  – {control_label -> node_id}
        user_map     – {user_tag_label -> node_id}
    """
    nodes = []

    # Controlled tags
    for i, row in control_df.reset_index(drop=True).iterrows():
        nodes.append(
            {
                "node_id": len(nodes),
                "node_type": "TAG_CONTROL",
                "label": str(row["label"]),
                "is_controlled": 1,
                "source": "FAST_subset",
                "extra_id": str(row.get("extra_id", "")),
            }
        )
    control_start = 0
    control_end = len(nodes)
    control_labels = [n["label"] for n in nodes[control_start:control_end]]
    control_map = {lab: control_start + idx for idx, lab in enumerate(control_labels)}

    # User tags
    for _, row in user_df.iterrows():
        nodes.append(
            {
                "node_id": len(nodes),
                "node_type": "TAG_USER",
                "label": str(row["tag_label"]),
                "is_controlled": 0,
                "source": "user",
                "extra_id": str(row.get("tag_id", "")),
            }
        )
    user_start = control_end
    user_end = len(nodes)
    user_labels = [n["label"] for n in nodes[user_start:user_end]]
    user_map = {lab: user_start + idx for idx, lab in enumerate(user_labels)}

    # Proposition nodes (optional)
    prop_map = {}
    if include_props and not assignments_df.empty:
        # Unique propositions keyed by (subreddit_key, corpus_path, row_id)
        assignments_df = assignments_df.copy()
        assignments_df["prop_key"] = (
            assignments_df["subreddit_key"].astype(str)
            + "||"
            + assignments_df["corpus_path"].astype(str)
            + "||"
            + assignments_df["row_id"].astype(str)
        )
        unique_props = assignments_df["prop_key"].unique().tolist()
        for pk in unique_props:
            sub, corpus, rid = pk.split("||")
            nodes.append(
                {
                    "node_id": len(nodes),
                    "node_type": "PROP",
                    "label": f"{sub}|{rid}",
                    "is_controlled": 0,
                    "source": corpus,
                    "extra_id": pk,
                }
            )
            prop_map[pk] = nodes[-1]["node_id"]

    nodes_df = pd.DataFrame(nodes)
    print(f"[INFO] Built node table with {len(nodes_df)} nodes "
          f"({len(control_map)} control, {len(user_map)} user, "
          f"{len(nodes_df) - len(control_map) - len(user_map)} props).")

    return nodes_df, control_map, user_map


def build_control_edges(
    control_edges_df: pd.DataFrame,
    control_map: Dict[str, int],
) -> pd.DataFrame:
    if control_edges_df is None or control_edges_df.empty:
        return pd.DataFrame(columns=["src_id", "dst_id", "relation", "weight", "source"])

    rows = []
    for _, row in control_edges_df.iterrows():
        s_lab = row["src_label"]
        d_lab = row["dst_label"]
        if s_lab not in control_map or d_lab not in control_map:
            continue
        rows.append(
            {
                "src_id": control_map[s_lab],
                "dst_id": control_map[d_lab],
                "relation": str(row.get("relation", "CONTROL_REL")),
                "weight": float(row.get("weight", 1.0)),
                "source": "controlled",
            }
        )
    edges = pd.DataFrame(rows)
    print(f"[INFO] Built {len(edges)} CONTROL_REL edges.")
    return edges


def build_user_parent_edges(
    user_df: pd.DataFrame,
    control_map: Dict[str, int],
    user_map: Dict[str, int],
) -> pd.DataFrame:
    rows = []
    for _, row in user_df.iterrows():
        child_label = str(row["tag_label"])
        parent_label = str(row.get("parent_tag_label", "")).strip()
        if not parent_label:
            continue

        # Resolve parent: could be another user tag OR a controlled tag
        if parent_label in user_map:
            parent_id = user_map[parent_label]
        elif parent_label in control_map:
            parent_id = control_map[parent_label]
        else:
            # Parent not found; skip (or could create implicit node, but let's stay conservative)
            continue

        child_id = user_map.get(child_label)
        if child_id is None:
            continue

        rows.append(
            {
                "src_id": parent_id,   # parent -> child
                "dst_id": child_id,
                "relation": "USER_PARENT",
                "weight": 1.0,
                "source": "user_hierarchy",
            }
        )

    edges = pd.DataFrame(rows)
    print(f"[INFO] Built {len(edges)} USER_PARENT edges from parent_tag_label.")
    return edges


def embed_user_tags(
    user_df: pd.DataFrame,
    model: SentenceTransformer,
) -> np.ndarray:
    if user_df.empty:
        return np.zeros((0, 0), dtype="float32")

    texts = user_df["tag_label"].astype(str).tolist()
    emb = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")


def build_user_to_control_edges(
    user_df: pd.DataFrame,
    user_emb: np.ndarray,
    control_df: pd.DataFrame,
    control_emb: np.ndarray,
    user_map: Dict[str, int],
    control_map: Dict[str, int],
    top_k: int = 5,
    sim_threshold: float = 0.6,
) -> pd.DataFrame:
    if user_df.empty or user_emb.size == 0:
        return pd.DataFrame(columns=["src_id", "dst_id", "relation", "weight", "source"])

    # Normalize embeddings for cosine similarity
    u = user_emb / (np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-9)
    c = control_emb / (np.linalg.norm(control_emb, axis=1, keepdims=True) + 1e-9)

    rows = []
    for i, row in user_df.reset_index(drop=True).iterrows():
        u_vec = u[i]  # [dim]
        sims = c @ u_vec  # [n_control]
        # Top-K indices
        k = min(int(top_k), sims.shape[0])
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        label_user = row["tag_label"]
        src_id = user_map.get(label_user)
        if src_id is None:
            continue

        for j in top_idx:
            sim = float(sims[j])
            if sim < sim_threshold:
                continue
            label_control = control_df.iloc[int(j)]["label"]
            dst_id = control_map.get(label_control)
            if dst_id is None:
                continue
            rows.append(
                {
                    "src_id": src_id,        # user -> controlled
                    "dst_id": dst_id,
                    "relation": "USER_TO_CONTROL",
                    "weight": sim,
                    "source": "embedding_map",
                }
            )

    edges = pd.DataFrame(rows)
    print(
        f"[INFO] Built {len(edges)} USER_TO_CONTROL edges "
        f"(top_k={top_k}, sim_threshold={sim_threshold})."
    )
    return edges


def build_tag_prop_edges(
    assignments_df: pd.DataFrame,
    control_map: Dict[str, int],
    user_map: Dict[str, int],
    nodes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build TAGGED edges from tag nodes to PROP nodes.
    """
    if assignments_df.empty:
        return pd.DataFrame(
            columns=[
                "tag_id",
                "prop_id",
                "relation",
                "subreddit_key",
                "corpus_path",
                "row_id",
            ]
        )

    # Build prop_key -> node_id map from nodes_df
    prop_nodes = nodes_df[nodes_df["node_type"] == "PROP"].copy()
    prop_nodes["prop_key"] = prop_nodes["extra_id"].astype(str)
    prop_map = {
        pk: nid
        for pk, nid in zip(prop_nodes["prop_key"], prop_nodes["node_id"])
    }

    assignments_df = assignments_df.copy()
    assignments_df["prop_key"] = (
        assignments_df["subreddit_key"].astype(str)
        + "||"
        + assignments_df["corpus_path"].astype(str)
        + "||"
        + assignments_df["row_id"].astype(str)
    )

    rows = []
    for _, r in assignments_df.iterrows():
        tag_label = str(r["tag_label"])
        # Resolve tag id: prefer user map, then control map
        if tag_label in user_map:
            tag_id = user_map[tag_label]
        elif tag_label in control_map:
            tag_id = control_map[tag_label]
        else:
            # Tag label not found in node maps; skip
            continue

        pk = r["prop_key"]
        prop_id = prop_map.get(pk)
        if prop_id is None:
            continue

        rows.append(
            {
                "tag_id": tag_id,
                "prop_id": prop_id,
                "relation": "TAGGED",
                "subreddit_key": r["subreddit_key"],
                "corpus_path": r["corpus_path"],
                "row_id": int(r["row_id"]),
            }
        )

    edges = pd.DataFrame(rows)
    print(f"[INFO] Built {len(edges)} TAGGED tag–prop edges.")
    return edges


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build tag-centric KG from tagging artifacts.")
    p.add_argument(
        "--sim_threshold",
        type=float,
        default=0.60,
        help="Cosine similarity threshold for USER_TO_CONTROL edges (default: 0.60).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-K controlled tags to consider per user tag (default: 5).",
    )
    p.add_argument(
        "--no_props",
        action="store_true",
        help="If set, do NOT create PROP nodes or TAGGED edges.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] TAG_DATA_DIR = {TAG_DATA_DIR}")

    # 1) Load controlled vocab + embeddings
    control_df, control_emb = load_control_vocab()

    # 2) Load user tags + assignments
    user_df = load_user_tags()
    assign_df = load_user_assignments()

    # 3) Build node table
    nodes_df, control_map, user_map = build_node_table(
        control_df,
        user_df,
        assign_df,
        include_props=not args.no_props,
    )

    # 4) Controlled edges (from tags_edges_cat.csv if possible)
    control_edges_raw = load_control_edges(control_df)
    control_edges = build_control_edges(control_edges_raw, control_map)

    # 5) User hierarchy edges
    user_parent_edges = build_user_parent_edges(user_df, control_map, user_map)

    # 6) USER_TO_CONTROL edges (embedding-based)
    if not user_df.empty:
        print("[INFO] Embedding user tags for mapping...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        user_emb = embed_user_tags(user_df, model)
        user2control_edges = build_user_to_control_edges(
            user_df,
            user_emb,
            control_df,
            control_emb,
            user_map,
            control_map,
            top_k=args.top_k,
            sim_threshold=args.sim_threshold,
        )
    else:
        print("[INFO] No user tags yet; skipping USER_TO_CONTROL edges.")
        user2control_edges = pd.DataFrame(
            columns=["src_id", "dst_id", "relation", "weight", "source"]
        )

    # 7) TAGGED edges (tag->prop)
    if not args.no_props:
        tag_prop_edges = build_tag_prop_edges(assign_df, control_map, user_map, nodes_df)
    else:
        tag_prop_edges = pd.DataFrame(
            columns=[
                "tag_id",
                "prop_id",
                "relation",
                "subreddit_key",
                "corpus_path",
                "row_id",
            ]
        )

    # 8) Concatenate tag-tag edges
    tag_tag_edges = pd.concat(
        [control_edges, user_parent_edges, user2control_edges],
        ignore_index=True,
    )

    # 9) Save outputs
    nodes_df.to_csv(KG_NODES_OUT, index=False)
    tag_tag_edges.to_csv(KG_EDGES_OUT, index=False)
    tag_prop_edges.to_csv(KG_PROP_EDGES_OUT, index=False)

    print(f"[DONE] Wrote KG nodes to:      {KG_NODES_OUT}")
    print(f"[DONE] Wrote KG tag-tag edges to:   {KG_EDGES_OUT}")
    print(f"[DONE] Wrote KG tag-prop edges to:  {KG_PROP_EDGES_OUT}")


if __name__ == "__main__":
    main()
