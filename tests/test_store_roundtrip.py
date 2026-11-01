import pandas as pd

from tagsynapse import store


def test_loaders_return_schema_when_files_missing(tmp_path):
    tags = store.load_user_tags(tmp_path / "nope.csv")
    assert list(tags.columns) == store.USER_TAG_COLS and tags.empty
    assigns = store.load_tag_assignments(tmp_path / "nope.csv")
    assert list(assigns.columns) == store.ASSIGN_COLS and assigns.empty
    canon = store.load_canonicalization_events(tmp_path / "nope.csv")
    assert list(canon.columns) == store.CANON_COLS and canon.empty


def test_user_tags_roundtrip_and_next_id(tmp_path):
    p = tmp_path / "user_tags.csv"
    df = pd.DataFrame([{"tag_id": 0, "tag_label": "ml", "parent_tag_label": ""},
                       {"tag_id": 7, "tag_label": "transformers", "parent_tag_label": "ml"}])
    store.save_user_tags(df, p)
    back = store.load_user_tags(p)
    assert back["tag_label"].tolist() == ["ml", "transformers"]
    assert back["parent_tag_label"].tolist() == ["", "ml"]
    assert store.next_tag_id(back) == 8
    assert store.next_tag_id(pd.DataFrame()) == 0


def test_assignments_roundtrip_coerces_row_id(tmp_path):
    p = tmp_path / "assign.csv"
    df = pd.DataFrame([{"subreddit_key": "s", "corpus_path": "c", "doc_id": "", "row_id": "12", "tag_label": "ml"},
                       {"subreddit_key": "s", "corpus_path": "c", "doc_id": "", "row_id": "bad", "tag_label": "x"}])
    store.save_tag_assignments(df, p)
    back = store.load_tag_assignments(p)
    assert back["row_id"].tolist() == [12, -1]
    assert str(back["row_id"].dtype) == "int64"


def test_canonicalization_append_creates_then_extends(tmp_path):
    p = tmp_path / "canon.csv"
    store.append_canonicalization_events([], p)
    assert not p.exists()
    store.append_canonicalization_events([{"original_label": "a", "canonical_label": "b", "similarity": 0.9,
                                           "subreddit_key": "s", "corpus_path": "c", "row_id": 0, "timestamp": "t"}], p)
    store.append_canonicalization_events([{"original_label": "c", "canonical_label": "c", "similarity": 1.0,
                                           "subreddit_key": "s", "corpus_path": "c", "row_id": 1, "timestamp": "t"}], p)
    assert len(store.load_canonicalization_events(p)) == 2
