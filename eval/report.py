"""Print how vocabulary metrics have moved across recorded snapshots. No
plotting library, no dashboard - a readable table is enough while the
history is short.

    python -m eval.report
    python -m eval.report pilot-1      # filter to one label
"""
import sys

from .snapshot import load_history


def main(label_filter: str = None) -> None:
    snaps = load_history()
    if label_filter:
        snaps = [s for s in snaps if s["label"] == label_filter]

    if not snaps:
        print("No recorded snapshots yet - run `python -m eval.snapshot` first.")
        return

    header = (
        f"{'snapshot':<10} {'when':<20} {'label':<12} {'concepts':<9} {'usertags':<9} "
        f"{'assigns':<8} {'docs':<6} {'coverage':<9} {'tags/doc':<9} {'couse':<6} {'subsume':<7}"
    )
    print(header)
    print("-" * len(header))
    for s in snaps:
        c, v = s["counts"], s["vocabulary_metrics"]
        print(
            f"{s['snapshot_id']:<10} {s['timestamp'][:19]:<20} {s['label']:<12} "
            f"{c['n_controlled_concepts']:<9} {c['n_user_tags']:<9} {c['n_assignments']:<8} {c['n_tagged_docs']:<6} "
            f"{v['controlled_vocab_coverage']:<9.2f} {v['tags_per_doc']:<9.2f} "
            f"{v['n_couse_edges']:<6} {v['n_subsumption_candidates']:<7}"
        )

    if len(snaps) >= 2:
        first, last = snaps[0], snaps[-1]
        print("\nFirst -> latest:")
        for label, (sec, key) in [
            ("n_user_tags", ("counts", "n_user_tags")),
            ("controlled_vocab_coverage", ("vocabulary_metrics", "controlled_vocab_coverage")),
            ("n_couse_edges", ("vocabulary_metrics", "n_couse_edges")),
            ("n_subsumption_candidates", ("vocabulary_metrics", "n_subsumption_candidates")),
        ]:
            print(f"  {label}: {first[sec][key]} -> {last[sec][key]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
