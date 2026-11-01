"""Read/write the runtime tagging state the app accumulates as people tag.

Three CSVs, all under config.TAG_DATA_DIR:

  user_tags.csv                  tag_id, tag_label, parent_tag_label
  user_tag_assignments.csv       subreddit_key, corpus_path, doc_id, row_id, tag_label
  tag_canonicalization_events.csv original_label, canonical_label, similarity,
                                 subreddit_key, corpus_path, row_id, timestamp

The loaders always return a DataFrame with the expected columns, empty if the
file isn't there yet, so callers never special-case first run.
"""
from pathlib import Path

import pandas as pd

from . import config

USER_TAG_COLS = ["tag_id", "tag_label", "parent_tag_label"]
ASSIGN_COLS = ["subreddit_key", "corpus_path", "doc_id", "row_id", "tag_label"]
CANON_COLS = ["original_label", "canonical_label", "similarity", "subreddit_key", "corpus_path", "row_id", "timestamp"]


def load_user_tags(path: Path = None) -> pd.DataFrame:
    path = path or config.USER_TAGS_PATH
    if Path(path).exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=USER_TAG_COLS)
    if "tag_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "tag_id"})
    if "parent_tag_label" not in df.columns:
        df["parent_tag_label"] = ""
    df["tag_label"] = df["tag_label"].astype(str)
    df["parent_tag_label"] = df["parent_tag_label"].fillna("").astype(str)
    return df


def save_user_tags(df: pd.DataFrame, path: Path = None) -> None:
    path = path or config.USER_TAGS_PATH
    df = df.copy()
    if "tag_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "tag_id"})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_tag_assignments(path: Path = None) -> pd.DataFrame:
    path = path or config.TAG_ASSIGN_PATH
    if Path(path).exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=ASSIGN_COLS)
    if not df.empty:
        df["subreddit_key"] = df["subreddit_key"].astype(str)
        df["corpus_path"] = df["corpus_path"].astype(str)
        df["tag_label"] = df["tag_label"].astype(str)
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype("int64")
    return df


def save_tag_assignments(df: pd.DataFrame, path: Path = None) -> None:
    path = path or config.TAG_ASSIGN_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.copy().to_csv(path, index=False)


def load_canonicalization_events(path: Path = None) -> pd.DataFrame:
    path = path or config.TAG_CANON_LOG_PATH
    if Path(path).exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=CANON_COLS)


def append_canonicalization_events(rows: list[dict], path: Path = None) -> None:
    if not rows:
        return
    path = path or config.TAG_CANON_LOG_PATH
    old = load_canonicalization_events(path)
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def next_tag_id(tags_df: pd.DataFrame) -> int:
    if tags_df.empty or "tag_id" not in tags_df.columns:
        return 0
    try:
        return int(tags_df["tag_id"].max()) + 1
    except Exception:
        return len(tags_df)
