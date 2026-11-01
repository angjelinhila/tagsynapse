import math

import pandas as pd
import pytest

from tagsynapse import usage as U


def _assign(rows):
    return pd.DataFrame(rows, columns=["subreddit_key", "corpus_path", "row_id", "tag_label"])


@pytest.fixture
def toy():
    # 4 docs. "ml" appears in all 4; "transformers" in 3, always with "ml";
    # "ethics" alone on one doc.
    return _assign([
        ("s", "c", 0, "ml"), ("s", "c", 0, "transformers"),
        ("s", "c", 1, "ml"), ("s", "c", 1, "transformers"),
        ("s", "c", 2, "ml"), ("s", "c", 2, "transformers"),
        ("s", "c", 3, "ml"), ("s", "c", 3, "ethics"),
    ])


def test_doc_tag_sets_groups_by_document(toy):
    sets = U.doc_tag_sets(toy)
    assert len(sets) == 4
    assert sets["s||c||3"] == {"ml", "ethics"}


def test_doc_tag_sets_strips_and_drops_blank_labels():
    df = _assign([("s", "c", 0, " ml "), ("s", "c", 0, ""), ("s", "c", 0, None)])
    assert U.doc_tag_sets(df) == {"s||c||0": {"ml"}}


def test_couse_counts_are_unordered_and_lexical(toy):
    pairs = U.couse_counts(U.doc_tag_sets(toy))
    assert pairs[("ml", "transformers")] == 3
    assert pairs[("ethics", "ml")] == 1
    assert ("transformers", "ml") not in pairs


def test_jaccard_and_pmi_closed_forms():
    assert U.jaccard(3, 4, 3) == pytest.approx(3 / 4)
    assert U.jaccard(0, 0, 0) == 0.0
    # P(i,j)=3/4, P(i)=1, P(j)=3/4 -> PMI = log(1) = 0
    assert U.pmi(3, 4, 3, 4) == pytest.approx(0.0)
    assert U.pmi(0, 4, 3, 4) == 0.0  # guarded, not -inf
    assert U.pmi(1, 1, 1, 4) == pytest.approx(math.log(4))


def test_behavioral_subsumption_direction():
    # transformers -> ml always (P(ml|tf)=1), ml -> transformers 3/4 -> B>0 means
    # transformers is the narrower one. Argument order: (n_ij, n_i, n_j) for i=tf.
    b = U.behavioral_subsumption(3, 3, 4)
    assert b == pytest.approx(1.0 - 0.75)
    assert U.behavioral_subsumption(3, 4, 3) == pytest.approx(-b)
    assert U.behavioral_subsumption(0, 0, 0) == 0.0


def test_couse_edges_schema_and_relation_name(toy):
    edges = U.couse_edges(toy)
    assert list(edges.columns) == ["src_label", "dst_label", "relation", "n_ij", "n_i", "n_j", "jaccard", "pmi", "subsumption"]
    assert set(edges["relation"]) == {"coUsedWith"}
    top = edges.iloc[0]
    assert (top["src_label"], top["dst_label"], top["n_ij"]) == ("ml", "transformers", 3)


def test_couse_edges_min_count_filter(toy):
    assert len(U.couse_edges(toy, min_count=2)) == 1
    assert U.couse_edges(_assign([])).empty


def test_subsumption_candidates_oriented_narrower_to_broader(toy):
    edges = U.couse_edges(toy)
    cands = U.subsumption_candidates(edges, threshold=0.2, min_count=2)
    # edge row has src="ml", dst="transformers" so B = P(tf|ml) - P(ml|tf) = 0.75 - 1 = -0.25
    # -> orientation flips: transformers is narrower than ml.
    assert len(cands) == 1
    assert cands.iloc[0]["narrower"] == "transformers"
    assert cands.iloc[0]["broader"] == "ml"
    assert cands.iloc[0]["subsumption"] == pytest.approx(0.25)
