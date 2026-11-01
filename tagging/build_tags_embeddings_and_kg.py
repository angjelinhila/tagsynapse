#!/usr/bin/env python
"""
Phase 2: Embeddings + KG index for FAST topical tags.

Inputs (from Phase 1) in the Tagging Pipeline directory:
  - tags_vocab_cat.csv
      tag_id,tag_label,alt_labels,facet
  - tags_edges_cat.csv
      src_tag_id,dst_tag_id,relation   # relation in {broader,narrower,related}

Outputs:
  - tags_vocab_embeddings.npy          # (n_tags, dim) float32
  - tags_vocab_index.csv               # tag_idx,tag_id,tag_label,alt_labels,facet
  - kg_edges_index.csv                 # src_idx,dst_idx,relation  (only edges where both endpoints have embeddings)
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> `tagsynapse` importable
from tagsynapse import config

# Defaults come from tagsynapse/config.py; both are still overridable on the CLI.
DEFAULT_TAGGING_DIR = config.TAG_DATA_DIR
DEFAULT_MODEL_NAME = config.EMBEDDING_MODEL_NAME


def load_vocab(vocab_path: Path) -> pd.DataFrame:
    """Load tags_vocab.csv and ensure stable index."""
    df = pd.read_csv(vocab_path)
    # Keep order as-is (Phase 1 already sorted by label), just make index explicit.
    df = df.reset_index(drop=True)
    return df


def build_embedding_text(row: pd.Series) -> str:
    """
    Build the text used for embedding.
    Here: prefLabel + alt_labels (if present) as one string.
    """
    label = str(row.get("tag_label", "") or "").strip()
    alt = str(row.get("alt_labels", "") or "").strip()

    if alt:
        return f"{label}. {alt}"
    return label


def embed_tags(df: pd.DataFrame, model_name: str, batch_size: int = 256) -> np.ndarray:
    """Compute SBERT embeddings for all tags."""
    print(f"[INFO] Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {dim}")

    texts = [build_embedding_text(row) for _, row in df.iterrows()]
    n = len(texts)
    print(f"[INFO] Embedding {n} tags in batches of {batch_size}...")

    all_embs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        embs = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embs.append(embs.astype("float32"))

    embeddings = np.vstack(all_embs)
    assert embeddings.shape[0] == n, "Row/embedding count mismatch"
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    return embeddings


def write_embeddings(out_dir: Path, embeddings: np.ndarray):
    out_path = out_dir / "tags_vocab_embeddings.npy"
    np.save(out_path, embeddings)
    print(f"[INFO] Wrote embeddings to {out_path}")


def write_vocab_index(out_dir: Path, df: pd.DataFrame):
    """
    Write tags_vocab_index.csv with explicit tag_idx.
    Index = row position = embedding row index.
    """
    df_idx = df.copy()
    df_idx = df_idx.reset_index(drop=True)
    df_idx.insert(0, "tag_idx", df_idx.index.astype("int32"))

    out_path = out_dir / "tags_vocab_index.csv"
    df_idx.to_csv(out_path, index=False)
    print(f"[INFO] Wrote vocab index to {out_path} (rows={len(df_idx)})")


def build_kg_edges_index(
    out_dir: Path,
    vocab_df: pd.DataFrame,
    edges_path: Path,
):
    """
    Map tags_edges.csv (src_tag_id,dst_tag_id,relation) into index space.

    Output: kg_edges_index.csv with src_idx,dst_idx,relation
    for edges where both endpoints exist in vocab_df.
    """
    if not edges_path.exists():
        print(f"[WARN] Edges file not found: {edges_path} – skipping KG index.")
        return

    edges_df = pd.read_csv(edges_path)

    id_to_idx = {
        tag_id: idx
        for idx, tag_id in enumerate(vocab_df["tag_id"].astype(str).tolist())
    }

    rows = []
    dropped = 0

    for _, row in edges_df.iterrows():
        src_id = str(row.get("src_tag_id", "")).strip()
        dst_id = str(row.get("dst_tag_id", "")).strip()
        rel = str(row.get("relation", "")).strip()

        src_idx = id_to_idx.get(src_id)
        dst_idx = id_to_idx.get(dst_id)

        if src_idx is None or dst_idx is None:
            dropped += 1
            continue

        rows.append({"src_idx": src_idx, "dst_idx": dst_idx, "relation": rel})

    if not rows:
        print("[WARN] No edges survived mapping to vocab indices; kg_edges_index.csv will be empty.")
        kg_df = pd.DataFrame(columns=["src_idx", "dst_idx", "relation"])
    else:
        kg_df = pd.DataFrame(rows)

    out_path = out_dir / "kg_edges_index.csv"
    kg_df.to_csv(out_path, index=False)
    print(
        f"[INFO] Wrote KG edge index to {out_path} "
        f"(edges kept={len(kg_df)}, edges dropped (no label)={dropped})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build SBERT embeddings + KG index for FAST topical tags."
    )
    parser.add_argument(
        "--tagging-dir",
        default=str(DEFAULT_TAGGING_DIR),
        help="Directory containing tags_vocab.csv and tags_edges.csv "
             "(default: Tagging Pipeline dir).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name "
             "(default: sentence-transformers/all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding (default: 256).",
    )
    args = parser.parse_args()

    tagging_dir = Path(args.tagging_dir)
    vocab_path = tagging_dir / "tags_vocab_cat.csv"
    edges_path = tagging_dir / "tags_edges_cat.csv"

    if not vocab_path.exists():
        raise FileNotFoundError(f"tags_vocab_cat.csv not found at {vocab_path}")

    print(f"[INFO] Loading vocab from {vocab_path}")
    vocab_df = load_vocab(vocab_path)
    print(f"[INFO] Vocab size: {len(vocab_df)} tags")

    # Embeddings
    embeddings = embed_tags(vocab_df, model_name=args.model_name, batch_size=args.batch_size)
    write_embeddings(tagging_dir, embeddings)

    # Vocab index file
    write_vocab_index(tagging_dir, vocab_df)

    # KG edges index (optional but useful for GNN later)
    build_kg_edges_index(tagging_dir, vocab_df, edges_path)

    print("[INFO] Phase 2 complete.")


if __name__ == "__main__":
    main()
