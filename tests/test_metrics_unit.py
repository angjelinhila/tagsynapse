import pandas as pd
import pytest

from eval import metrics as M


def _assign(rows):
    return pd.DataFrame(rows, columns=["subreddit_key", "corpus_path", "row_id", "tag_label"])


@pytest.fixture
def toy():
    return _assign([
        ("s", "c", 0, "ml"), ("s", "c", 0, "transformers"),
        ("s", "c", 1, "ml"), ("s", "c", 1, "transformers"),
        ("s", "c", 2, "ml"), ("s", "c", 2, "transformers"),
        ("s", "c", 3, "ml"), ("s", "c", 3, "ethics"),
    ])


def test_study_dependent_metrics_fail_loudly():
    with pytest.raises(NotImplementedError):
        M.tag_acceptance_rate(pd.DataFrame())
    with pytest.raises(NotImplementedError):
        M.retrieval_precision_recall([], set(), 10)


def test_coverage_and_custom_rate_are_complements(toy):
    vocab = {"ml"}
    cov = M.controlled_vocab_coverage(toy, vocab)
    assert cov == pytest.approx(4 / 8)
    assert M.custom_tag_rate(toy, vocab) == pytest.approx(1 - cov)
    assert M.controlled_vocab_coverage(_assign([]), vocab) == 0.0


def test_tags_per_doc(toy):
    assert M.tags_per_doc(toy) == pytest.approx(2.0)
    assert M.tags_per_doc(_assign([])) == 0.0


def test_vocabulary_size_ignores_blanks():
    df = pd.DataFrame({"tag_label": ["a", "b", "", "a"], "parent_tag_label": ["", "", "", ""]})
    assert M.vocabulary_size(df) == 2
    assert M.vocabulary_size(pd.DataFrame()) == 0


def test_hierarchy_depth_and_cycle_detection():
    df = pd.DataFrame([
        {"tag_label": "a", "parent_tag_label": "b"},
        {"tag_label": "b", "parent_tag_label": "c"},
        {"tag_label": "c", "parent_tag_label": ""},
        {"tag_label": "x", "parent_tag_label": "y"},
        {"tag_label": "y", "parent_tag_label": "x"},
    ])
    out = M.hierarchy_depth_stats(df)
    assert out["n_with_parent"] == 4
    assert out["max_depth"] == 2
    assert out["n_cycles"] == 2  # detected from both x and y


def test_couse_graph_stats(toy):
    out = M.couse_graph_stats(toy)
    assert out["n_couse_edges"] == 2
    assert out["n_subsumption_candidates"] >= 0
    assert M.couse_graph_stats(_assign([]))["n_couse_edges"] == 0


def test_canonicalization_rate_counts_only_changed_labels(toy):
    canon = pd.DataFrame([{"original_label": "AI", "canonical_label": "ml"},
                          {"original_label": "ml", "canonical_label": "ml"}])
    assert M.canonicalization_rate(canon, toy) == pytest.approx(1 / 8)
    assert M.canonicalization_rate(pd.DataFrame(), toy) == 0.0
