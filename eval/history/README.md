# Snapshot history

`snapshots.jsonl` lives here once the first `python -m eval.snapshot` creates
it. One JSON object per line, append-only.

This file is **intentionally version-controlled, not gitignored** (the
repo-wide `*.jsonl` ignore has an exception for it). The point of recording
snapshots isn't today's numbers - it's being able to look back across commits
and answer "did that change to KG expansion actually shift how people tag?"
by reading this file's git history, not by trusting memory.

Read it with:

```bash
python -m eval.report            # all snapshots, readable table
python -m eval.report pilot-1    # filtered to one label
```

See `docs/EVALUATION.md` for what each recorded field means and which
metrics are computable today vs. pending the controlled study.
