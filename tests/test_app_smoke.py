"""Smoke test: the Streamlit app module imports and its pure helpers resolve
to the package - needs streamlit + sentence-transformers installed, skipped
automatically otherwise. Loading a corpus or model is NOT exercised here.
"""
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None
    or importlib.util.find_spec("sentence_transformers") is None,
    reason="needs streamlit + sentence-transformers installed to import the app",
)


def test_app_imports_and_uses_package_helpers():
    import importlib
    from tagsynapse import config, relations, text

    app = importlib.import_module("app.streamlit_app")
    assert app.FAISS_API_URL == config.FAISS_API_URL
    assert app._relation_bucket is relations.relation_bucket
    assert app.tokenize_query is text.tokenize
    assert app.SUBREDDIT_CONFIG is config.SUBREDDIT_CONFIG
