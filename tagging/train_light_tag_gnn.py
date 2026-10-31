#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import random
import math

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise SystemExit(
        "PyTorch is required for this script.\n"
        "Install it in your venv first, e.g.:\n"
        "pip install torch"
    ) from e

from sentence_transformers import SentenceTransformer

BASE_DIR = Path("/home/ubuntu/RedditAI_Vector_DB")
TAG_DIR = BASE_DIR / "faiss" / "tagging"

USED_NODES_PATH = TAG_DIR / "used_tag_nodes.csv"
USED_EDGES_INDEX_PATH = TAG_DIR / "used_tag_edges_index.csv"
TAG_VOCAB_PATH = TAG_DIR / "tags_vocab_cat.csv"
TAG_VOCAB_INDEX_PATH = TAG_DIR / "tags_vocab_index.csv"
TAG_VOCAB_EMB_PATH = TAG_DIR / "tags_vocab_embeddings.npy"
TAG_CANON_LOG_PATH = TAG_DIR / "tag_canonicalization_events.csv"

OUT_PRED_PATH = TAG_DIR / "gnn_tag_predictions.csv"
OUT_MODEL_PATH = TAG_DIR / "light_tag_gnn.pt"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_vocab():
    vocab = pd.read_csv(TAG_VOCAB_PATH)
    label_candidates = [
        c for c in vocab.columns
        if c.lower() in ("tag_label", "label", "tag", "canonical_label", "name")
    ]
    if not label_candidates:
        raise ValueError("Could not find tag label column in tags_vocab_cat.csv")
    if label_candidates[0] != "tag_label":
        vocab = vocab.rename(columns={label_candidates[0]: "tag_label"})
    vocab["tag_label"] = vocab["tag_label"].astype(str).str.strip()
    return vocab


def load_vocab_embeddings(vocab_df: pd.DataFrame):
    emb = np.load(TAG_VOCAB_EMB_PATH).astype("float32")

    if TAG_VOCAB_INDEX_PATH.exists():
        idx_df = pd.read_csv(TAG_VOCAB_INDEX_PATH)
        label_candidates = [
            c for c in idx_df.columns
            if c.lower() in ("tag_label", "label", "tag", "canonical_label", "name")
        ]
        idx_candidates = [
            c for c in idx_df.columns
            if c.lower() in ("tag_idx", "emb_idx", "index", "idx", "row", "row_id")
        ]
        if label_candidates and idx_candidates:
            lab_col = label_candidates[0]
            idx_col = idx_candidates[0]
            idx_map = {
                str(row[lab_col]).strip(): int(row[idx_col])
                for _, row in idx_df.iterrows()
            }
            keep = vocab_df["tag_label"].isin(idx_map.keys())
            vocab_df = vocab_df[keep].reset_index(drop=True)
            emb = np.vstack([emb[idx_map[label]] for label in vocab_df["tag_label"]]).astype("float32")
            return vocab_df, emb

    n = min(len(vocab_df), emb.shape[0])
    return vocab_df.iloc[:n].reset_index(drop=True), emb[:n]


def build_features(nodes_df: pd.DataFrame, vocab_df: pd.DataFrame, vocab_emb: np.ndarray):
    vocab_map = {
        label: vocab_emb[i]
        for i, label in enumerate(vocab_df["tag_label"].tolist())
    }

    labels_to_encode = []
    encode_idx = []

    X = np.zeros((len(nodes_df), vocab_emb.shape[1]), dtype="float32")

    for i, row in nodes_df.iterrows():
        label = str(row["label"]).strip()
        if label in vocab_map:
            X[i] = vocab_map[label]
        else:
            labels_to_encode.append(label)
            encode_idx.append(i)

    if labels_to_encode:
        model = SentenceTransformer(MODEL_NAME)
        enc = model.encode(labels_to_encode, convert_to_numpy=True).astype("float32")
        for pos, i in enumerate(encode_idx):
            X[i] = enc[pos]

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / norms
    return X


def build_adjacency(num_nodes: int, edges_df: pd.DataFrame):
    A = np.zeros((num_nodes, num_nodes), dtype="float32")
    for _, row in edges_df.iterrows():
        s = int(row["src_id"])
        d = int(row["dst_id"])
        w = float(row.get("weight", 1.0))
        A[s, d] += w
        A[d, s] += w

    # self loops
    A += np.eye(num_nodes, dtype="float32")

    deg = A.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    D = np.diag(deg_inv_sqrt)
    A_norm = D @ A @ D
    return A_norm.astype("float32")


def build_training_pairs(nodes_df: pd.DataFrame):
    canon_df = pd.read_csv(TAG_CANON_LOG_PATH) if TAG_CANON_LOG_PATH.exists() else pd.DataFrame()
    node_label_to_id = {str(r["label"]).strip(): int(r["node_id"]) for _, r in nodes_df.iterrows()}

    control_nodes = nodes_df[nodes_df["node_type"] == "controlled"]["node_id"].astype(int).tolist()
    user_nodes = nodes_df[nodes_df["node_type"] != "controlled"]["node_id"].astype(int).tolist()

    positives = []

    if not canon_df.empty:
        canon_df["original_label"] = canon_df["original_label"].astype(str).str.strip()
        canon_df["canonical_label"] = canon_df["canonical_label"].astype(str).str.strip()

        grouped = canon_df.groupby(["original_label", "canonical_label"]).size().reset_index(name="n")
        for _, row in grouped.iterrows():
            orig = row["original_label"]
            can = row["canonical_label"]
            if orig in node_label_to_id and can in node_label_to_id:
                u = node_label_to_id[orig]
                c = node_label_to_id[can]
                if u in user_nodes and c in control_nodes:
                    positives.append((u, c))

    positives = sorted(set(positives))
    if not positives:
        raise SystemExit(
            "No positive canonicalization pairs found.\n"
            "You need at least some rows in tag_canonicalization_events.csv before training this GNN."
        )

    pos_by_user = {}
    for u, c in positives:
        pos_by_user.setdefault(u, set()).add(c)

    negatives = []
    for u, c in positives:
        candidates = [x for x in control_nodes if x not in pos_by_user[u]]
        if not candidates:
            continue
        neg_c = random.choice(candidates)
        negatives.append((u, neg_c))

    if not negatives:
        raise SystemExit("Could not sample negatives; not enough distinct controlled tags.")

    return positives, negatives, control_nodes, user_nodes


class LightTagGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

        self.scorer = nn.Sequential(
            nn.Linear(out_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode(self, X: torch.Tensor, A: torch.Tensor):
        H = A @ X
        H = F.relu(self.lin1(H))
        Z = A @ H
        Z = self.lin2(Z)
        Z = F.normalize(Z, dim=1)
        return Z

    def score_pairs(self, Z: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor):
        z1 = Z[src_idx]
        z2 = Z[dst_idx]
        feats = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=1)
        return self.scorer(feats).squeeze(1)


def main():
    nodes_df = pd.read_csv(USED_NODES_PATH)
    edges_df = pd.read_csv(USED_EDGES_INDEX_PATH)

    if nodes_df.empty or edges_df.empty:
        raise SystemExit("Used graph files are empty. Run build_usage_enriched_tag_kg.py first.")

    vocab_df = load_vocab()
    vocab_df, vocab_emb = load_vocab_embeddings(vocab_df)

    X_np = build_features(nodes_df, vocab_df, vocab_emb)
    A_np = build_adjacency(len(nodes_df), edges_df)

    positives, negatives, control_nodes, user_nodes = build_training_pairs(nodes_df)

    X = torch.tensor(X_np, dtype=torch.float32)
    A = torch.tensor(A_np, dtype=torch.float32)

    pos_src = torch.tensor([u for u, _ in positives], dtype=torch.long)
    pos_dst = torch.tensor([c for _, c in positives], dtype=torch.long)
    neg_src = torch.tensor([u for u, _ in negatives], dtype=torch.long)
    neg_dst = torch.tensor([c for _, c in negatives], dtype=torch.long)

    model = LightTagGNN(in_dim=X.shape[1], hidden_dim=128, out_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(300):
        model.train()
        Z = model.encode(X, A)

        pos_scores = model.score_pairs(Z, pos_src, pos_dst)
        neg_scores = model.score_pairs(Z, neg_src, neg_dst)

        y_pos = torch.ones_like(pos_scores)
        y_neg = torch.zeros_like(neg_scores)

        loss = F.binary_cross_entropy_with_logits(pos_scores, y_pos) + \
               F.binary_cross_entropy_with_logits(neg_scores, y_neg)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 50 == 0:
            print(f"epoch={epoch+1} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        Z = model.encode(X, A)

    control_tensor = torch.tensor(control_nodes, dtype=torch.long)
    rows = []

    for u in user_nodes:
        src = torch.tensor([u] * len(control_nodes), dtype=torch.long)
        scores = torch.sigmoid(model.score_pairs(Z, src, control_tensor)).detach().cpu().numpy()
        order = np.argsort(scores)[::-1][:5]

        raw_label = str(nodes_df.loc[nodes_df["node_id"] == u, "label"].iloc[0])
        for rank, j in enumerate(order, start=1):
            c_id = control_nodes[j]
            pred_label = str(nodes_df.loc[nodes_df["node_id"] == c_id, "label"].iloc[0])
            rows.append(
                {
                    "raw_tag_label": raw_label,
                    "predicted_control_tag": pred_label,
                    "score": float(scores[j]),
                    "rank": rank,
                }
            )

    pred_df = pd.DataFrame(rows).sort_values(["raw_tag_label", "rank"])
    pred_df.to_csv(OUT_PRED_PATH, index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": X.shape[1],
            "hidden_dim": 128,
            "out_dim": 64,
        },
        OUT_MODEL_PATH,
    )

    print(f"[OK] wrote {OUT_PRED_PATH}")
    print(f"[OK] wrote {OUT_MODEL_PATH}")


if __name__ == "__main__":
    main()