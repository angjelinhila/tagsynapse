import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    p = Path(__file__).resolve().parent.parent / "tagging" / "build_usage_enriched_tag_kg.py"
    spec = importlib.util.spec_from_file_location("build_usage_enriched_tag_kg", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_from_toy_state(tmp_path, monkeypatch):
    from tagsynapse import config
    monkeypatch.setattr(config, "TAG_DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "TAG_VOCAB_PATH", tmp_path / "tags_vocab_cat.csv")
    monkeypatch.setattr(config, "USER_TAGS_PATH", tmp_path / "user_tags.csv")
    monkeypatch.setattr(config, "TAG_ASSIGN_PATH", tmp_path / "assign.csv")
    monkeypatch.setattr(config, "TAG_CANON_LOG_PATH", tmp_path / "canon.csv")

    pd.DataFrame({"tag_label": ["ml"]}).to_csv(config.TAG_VOCAB_PATH, index=False)
    pd.DataFrame([{"tag_id": 0, "tag_label": "transformers", "parent_tag_label": "ml"},
                  {"tag_id": 1, "tag_label": "ml", "parent_tag_label": ""}]).to_csv(config.USER_TAGS_PATH, index=False)
    pd.DataFrame([("s", "c", "", 0, "ml"), ("s", "c", "", 0, "transformers"), ("s", "c", "", 1, "ml")],
                 columns=["subreddit_key", "corpus_path", "doc_id", "row_id", "tag_label"]).to_csv(config.TAG_ASSIGN_PATH, index=False)

    mod = _load_script()
    couse, nodes, edges = mod.build()

    assert len(couse) == 1 and couse.iloc[0]["relation"] == "coUsedWith"
    assert set(nodes["label"]) == {"ml", "transformers"}
    assert nodes.set_index("label").loc["ml", "node_type"] == "hybrid"
    assert nodes.set_index("label").loc["transformers", "node_type"] == "user"
    assert set(edges["relation"]) == {"coUsedWith", "user_parent"}
