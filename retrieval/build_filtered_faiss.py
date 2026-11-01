#!/usr/bin/env python3
"""
build_filtered_faiss.py

Goal:
- Take your *_filtered_output.jsonl files (one doc per line).
- Build a lean corpus table (CSV + Parquet).
- Build a 384-dim FAISS index using MiniLM (cosine via inner product).

Inputs per subreddit (under config.RAW_JSONL_DIR):
  artificial_filtered_output.jsonl
  artificialint_filtered_output.jsonl
  chatgpt_filtered_output.jsonl
  openai_filtered_output.jsonl
  singularity_filtered_output.jsonl
  transhumanism_filtered_output.jsonl

Outputs per subreddit (under config.INDEX_DIR):
  <name>_filtered_output_corpus.csv
  <name>_filtered_output_corpus.parquet
  <name>_filtered_output_faiss.index
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> `tagsynapse` importable
from tagsynapse import config

# ---------- CONFIG ----------
# Where the *_filtered_output.jsonl files live. Set TAGSYNAPSE_RAW_JSONL_DIR
# (e.g. to a TACC allocation) - default is <DATA_DIR>/raw.
BASE_DIR = config.RAW_JSONL_DIR

INPUT_FILES = {
    "artificial": "artificial_filtered_output.jsonl",
    "artificialint": "artificialint_filtered_output.jsonl",
    "chatgpt": "chatgpt_filtered_output.jsonl",
    "openai": "openai_filtered_output.jsonl",
    "singularity": "singularity_filtered_output.jsonl",
    "transhumanism": "transhumanism_filtered_output.jsonl",
}

# Where to write the corpus + FAISS artifacts - the same dir the app and the
# faiss-service read from.
OUT_DIR = config.INDEX_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lighter embedding model (384-d) to stay RAM-friendly on a small box
MODEL_NAME = config.EMBEDDING_MODEL_NAME


# ---------- HELPERS ----------

def load_jsonl_to_corpus(path: Path, subreddit_name: str) -> pd.DataFrame:
    """
    Load a filtered JSONL file into a lean corpus DataFrame.

    For your filtered proposition files, we expect fields like:
        - 'user'
        - 'proposition'
        - 'timestamp'
        - 'sentiment_label', 'sentiment_score'
        - 'sim_ai', 'lex', 'dep'
    but we keep the function robust to other variants too.
    """
    records = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # --- MAIN TEXT FIELD ---
            # Prefer your 'proposition', then fall back to other common keys.
            text = (
                obj.get("proposition")
                or obj.get("text")
                or obj.get("body")
                or obj.get("selftext")
                or obj.get("content")
                or ""
            )
            text = (text or "").strip()
            if not text:
                # Skip empty text rows (these are "junk" we don't want)
                continue

            # --- METADATA FALLBACKS ---
            author = obj.get("author") or obj.get("user") or ""
            created_utc = obj.get("created_utc") or obj.get("timestamp") or ""

            # "score" might not exist in these filtered files
            score = obj.get("score", 0)

            # IDs / threading info – be conservative and keep original IDs if present
            submission_id = (
                obj.get("submission_id")
                or obj.get("post_id")
                or obj.get("id")
                or ""
            )
            comment_id = obj.get("comment_id", "")
            parent_id = obj.get("parent_id", "")

            # Thread id: prefer explicit thread_id, else submission_id/id, else doc index
            thread_id = (
                obj.get("thread_id")
                or submission_id
                or obj.get("id")
                or f"{subreddit_name}_{i}"
            )

            # Extra analytic fields from your example
            sentiment_label = obj.get("sentiment_label", None)
            sentiment_score = obj.get("sentiment_score", None)
            sim_ai = obj.get("sim_ai", None)
            lex = obj.get("lex", None)
            dep = obj.get("dep", None)

            rec = {
                "doc_id": i,  # stable doc id = row index used for FAISS
                "subreddit": obj.get("subreddit") or subreddit_name,
                "author": author,
                "created_utc": created_utc,  # string is fine; parse later if needed
                "score": score,
                "type": obj.get("type", ""),  # if you ever encode submission/comment etc.
                "submission_id": submission_id,
                "comment_id": comment_id,
                "parent_id": parent_id,
                "thread_id": thread_id,
                "text": text,
                # analytic fields from your filtered pipeline:
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "sim_ai": sim_ai,
                "lex": lex,
                "dep": dep,
            }

            records.append(rec)

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Keep memory tighter with dtypes where possible
    df["doc_id"] = df["doc_id"].astype("int32")

    if "score" in df.columns:
        df["score"] = (
            pd.to_numeric(df["score"], errors="coerce")
            .fillna(0)
            .astype("int32")
        )

    # sentiment_label often looks like int
    if "sentiment_label" in df.columns:
        df["sentiment_label"] = pd.to_numeric(
            df["sentiment_label"], errors="coerce"
        ).astype("Int8")

    # sentiment_score & sim_ai as float
    if "sentiment_score" in df.columns:
        df["sentiment_score"] = pd.to_numeric(
            df["sentiment_score"], errors="coerce"
        ).astype("float32")

    if "sim_ai" in df.columns:
        df["sim_ai"] = pd.to_numeric(
            df["sim_ai"], errors="coerce"
        ).astype("float32")

    # lex/dep are booleans in your example; coerce gently
    for col in ("lex", "dep"):
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    return df


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a cosine-similarity FAISS index by L2-normalizing and using Inner Product.
    """
    # Normalize in-place so inner product == cosine
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


# ---------- MAIN ----------

def main():
    print(f"Using embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print("Embedding dimension:", dim)

    for name, fname in INPUT_FILES.items():
        in_path = BASE_DIR / fname
        if not in_path.exists():
            print(f"[{name}] Skipping: file not found at {in_path}")
            continue

        print(f"\n=== Processing subreddit: {name} ===")
        print("Reading JSONL:", in_path)
        df = load_jsonl_to_corpus(in_path, subreddit_name=name)
        n_rows = len(df)
        print(f"[{name}] Rows after filtering empty text: {n_rows}")

        if n_rows == 0:
            print(f"[{name}] No usable rows, skipping index build.")
            continue

        # Compute embeddings in batches to avoid RAM spikes
        texts = df["text"].tolist()
        batch_size = 128
        all_embs = []

        print(f"[{name}] Encoding texts in batches of {batch_size}...")
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            batch_texts = texts[start:end]
            embs = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_embs.append(embs.astype("float32"))

        embeddings = np.vstack(all_embs)
        assert embeddings.shape[0] == n_rows, "Mismatch between rows and embeddings!"

        # Save corpus (CSV + Parquet)
        corpus_csv = OUT_DIR / f"{name}_filtered_output_corpus.csv"
        corpus_parquet = OUT_DIR / f"{name}_filtered_output_corpus.parquet"
        print(f"[{name}] Saving corpus CSV: {corpus_csv}")
        df.to_csv(corpus_csv, index=False)
        print(f"[{name}] Saving corpus Parquet: {corpus_parquet}")
        df.to_parquet(corpus_parquet, index=False)

        # Build and save FAISS index
        print(f"[{name}] Building FAISS index...")
        index = build_faiss_index(embeddings)
        index_path = OUT_DIR / f"{name}_filtered_output_faiss.index"
        print(f"[{name}] Writing FAISS index to: {index_path}")
        faiss.write_index(index, str(index_path))

        print(
            f"[{name}] Done. Corpus rows: {n_rows}, index size: {index.ntotal}, dim: {index.d}"
        )


if __name__ == "__main__":
    main()
