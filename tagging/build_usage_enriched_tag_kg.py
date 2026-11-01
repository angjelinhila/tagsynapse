#!/usr/bin/env python3
"""Add the empirical (R_emp) layer to the Tag Knowledge Graph from usage.

Reads the runtime tagging state the app accumulates and writes the usage
edges the GNN and the UI consume:

  user_tag_assignments.csv  --->  tag_cooccurrence_edges.csv
                                    src_label, dst_label, relation=coUsedWith,
                                    n_ij, n_i, n_j, jaccard, pmi, subsumption
                            --->  used_tag_nodes.csv
                                    node_id, label, node_type, n_assignments
                            --->  used_tag_edges.csv
                                    src_label, dst_label, relation, weight
                                    (coUsedWith + user_parent + canonicalized_to)
                            --->  used_tag_edges_index.csv
                                    src_id, dst_id, relation, weight

All of the statistics come from tagsynapse.usage (TAG_MODEL §7, §11) so they
are the same numbers the eval snapshots report. Nothing here touches the
ontological edges - empirical evidence stays in its own files, by design.

    python tagging/build_usage_enriched_tag_kg.py
    python tagging/build_usage_enriched_tag_kg.py --min-count 2
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> `tagsynapse` importable
from tagsynapse import config, store
from tagsynapse import usage as U


def _vocab_labels() -> set[str]:
    if not config.TAG_VOCAB_PATH.exists():
        return set()
    df = pd.read_csv(config.TAG_VOCAB_PATH)
    col = "tag_label" if "tag_label" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip())


def build(min_count: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assigns = store.load_tag_assignments()
    user_tags = store.load_user_tags()
    canon = store.load_canonicalization_events()
    vocab = _vocab_labels()

    couse = U.couse_edges(assigns, min_count=min_count)

    # --- nodes: every label that has actually been used -------------------
    doc_tags = U.doc_tag_sets(assigns)
    counts = U.tag_counts(doc_tags)
    user_labels = set(user_tags["tag_label"].astype(str).str.strip()) if not user_tags.empty else set()
    labels = set(counts)
    if not canon.empty:
        labels |= set(canon["original_label"].dropna().astype(str).str.strip())
        labels |= set(canon["canonical_label"].dropna().astype(str).str.strip())
    labels.discard("")

    def node_type(lbl: str) -> str:
        if lbl in vocab and lbl in user_labels:
            return "hybrid"
        if lbl in vocab:
            return "controlled"
        if lbl in user_labels:
            return "user"
        return "other"

    nodes = pd.DataFrame(
        [{"label": l, "node_type": node_type(l), "n_assignments": int(counts.get(l, 0))} for l in sorted(labels)],
        columns=["label", "node_type", "n_assignments"],
    )
    nodes.insert(0, "node_id", range(len(nodes)))

    # --- edges: empirical + user-asserted hierarchy + canonicalisation -----
    edges = []
    for _, r in couse.iterrows():
        edges.append({"src_label": r["src_label"], "dst_label": r["dst_label"], "relation": "coUsedWith", "weight": float(r["n_ij"])})
    if not user_tags.empty:
        for _, r in user_tags.iterrows():
            child, parent = str(r["tag_label"]).strip(), str(r["parent_tag_label"]).strip()
            if parent and child in labels and parent in labels:
                edges.append({"src_label": parent, "dst_label": child, "relation": "user_parent", "weight": 1.0})
    if not canon.empty:
        grp = canon.groupby(["original_label", "canonical_label"]).size().reset_index(name="w")
        for _, r in grp.iterrows():
            a, b = str(r["original_label"]).strip(), str(r["canonical_label"]).strip()
            if a != b and a in labels and b in labels:
                edges.append({"src_label": a, "dst_label": b, "relation": "canonicalized_to", "weight": float(r["w"])})
    edges_df = pd.DataFrame(edges, columns=["src_label", "dst_label", "relation", "weight"])
    return couse, nodes, edges_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-count", type=int, default=1, help="drop coUsedWith pairs seen fewer times than this")
    args = ap.parse_args()

    couse, nodes, edges = build(min_count=args.min_count)
    config.TAG_DATA_DIR.mkdir(parents=True, exist_ok=True)

    couse.to_csv(config.TAG_COOCCUR_PATH, index=False)
    nodes.to_csv(config.USED_NODES_PATH, index=False)
    edges.to_csv(config.USED_EDGES_PATH, index=False)

    id_of = dict(zip(nodes["label"], nodes["node_id"]))
    idx = edges.assign(src_id=edges["src_label"].map(id_of), dst_id=edges["dst_label"].map(id_of))
    idx[["src_id", "dst_id", "relation", "weight"]].to_csv(config.USED_EDGES_INDEX_PATH, index=False)

    print(f"[INFO] TAG_DATA_DIR = {config.TAG_DATA_DIR}")
    print(f"[INFO] {len(nodes)} used nodes, {len(couse)} coUsedWith pairs, {len(edges)} edges total")
    for p in (config.TAG_COOCCUR_PATH, config.USED_NODES_PATH, config.USED_EDGES_PATH, config.USED_EDGES_INDEX_PATH):
        print(f"[INFO] wrote {p}")


if __name__ == "__main__":
    main()
