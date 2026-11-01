import json

from eval import snapshot as S


def test_snapshot_records_and_reloads(tmp_path, monkeypatch):
    # Point the whole config at an empty temp dir: snapshot must still succeed.
    from tagsynapse import config
    monkeypatch.setattr(config, "TAG_VOCAB_PATH", tmp_path / "tags_vocab_cat.csv")
    monkeypatch.setattr(config, "USER_TAGS_PATH", tmp_path / "user_tags.csv")
    monkeypatch.setattr(config, "TAG_ASSIGN_PATH", tmp_path / "assign.csv")
    monkeypatch.setattr(config, "TAG_CANON_LOG_PATH", tmp_path / "canon.csv")

    hist = tmp_path / "snapshots.jsonl"
    snap = S.record_snapshot("unit", path=hist)
    assert snap["label"] == "unit"
    assert snap["counts"]["n_assignments"] == 0
    assert "controlled_vocab_coverage" in snap["vocabulary_metrics"]
    assert snap["study_metrics"] == {}

    S.record_snapshot("unit", path=hist)
    rows = S.load_history(hist)
    assert len(rows) == 2
    assert json.dumps(rows[0])  # serialisable
