"""Vocabulary snapshots: `python -m eval.snapshot` computes every metric
that is computable today from the tagging CSVs and appends one JSON line to
history/snapshots.jsonl, so we can see whether the vocabulary is actually
evolving across code and prompt changes - not just look at one moment.

Deliberately version-controlled (NOT in .gitignore). The question "did the
KG-expansion change in commit X change how people tag?" is a question about
this file's git history, not about today's numbers.

    python -m eval.snapshot                 # snapshot the configured TAG_DATA_DIR
    python -m eval.snapshot --label pilot-1 # tag the row so it can be filtered later
"""
import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tagsynapse import config, store
from tagsynapse import usage as U
from . import metrics as M

HISTORY_PATH = Path(__file__).parent / "history" / "snapshots.jsonl"


def _vocab_labels() -> set[str]:
    if not config.TAG_VOCAB_PATH.exists():
        return set()
    df = pd.read_csv(config.TAG_VOCAB_PATH)
    col = "tag_label" if "tag_label" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip())


def compute_snapshot(label: str = "default") -> dict:
    user_tags = store.load_user_tags()
    assigns = store.load_tag_assignments()
    canon = store.load_canonicalization_events()
    vocab = _vocab_labels()

    return {
        "snapshot_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "config": config.snapshot(),
        "counts": {
            "n_controlled_concepts": len(vocab),
            "n_user_tags": M.vocabulary_size(user_tags),
            "n_assignments": int(len(assigns)),
            "n_tagged_docs": len(U.doc_tag_sets(assigns)),
        },
        "vocabulary_metrics": {
            "controlled_vocab_coverage": M.controlled_vocab_coverage(assigns, vocab),
            "custom_tag_rate": M.custom_tag_rate(assigns, vocab),
            "tags_per_doc": M.tags_per_doc(assigns),
            "canonicalization_rate": M.canonicalization_rate(canon, assigns),
            **M.hierarchy_depth_stats(user_tags),
            **M.couse_graph_stats(assigns),
        },
        # Study-dependent metrics (eval/metrics.py) plug in here once the
        # conditions exist; until then the key is present and empty on purpose.
        "study_metrics": {},
    }


def record_snapshot(label: str = "default", path: Path = HISTORY_PATH) -> dict:
    snap = compute_snapshot(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(snap) + "\n")
    return snap


def load_history(path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="default", help="free-text tag for this snapshot (e.g. a study condition)")
    args = ap.parse_args()
    snap = record_snapshot(args.label)
    print(json.dumps(snap, indent=2))
    print(f"\nRecorded to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
