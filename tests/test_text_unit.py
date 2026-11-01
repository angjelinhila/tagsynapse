import numpy as np

from tagsynapse.text import STOPWORDS, normalize_rows, tokenize


def test_tokenize_lowercases_drops_stopwords_and_short_tokens():
    assert tokenize("The Transformers ARE a model of AI, in 2024!") == ["transformers", "model", "2024"]


def test_tokenize_handles_non_string():
    assert tokenize(None) == []
    assert tokenize(12345) == ["12345"]


def test_stopwords_is_immutable():
    assert isinstance(STOPWORDS, frozenset)
    assert "the" in STOPWORDS


def test_normalize_rows_gives_unit_vectors_and_survives_zero_rows():
    mat = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = normalize_rows(mat)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.all(np.isfinite(out[1]))
