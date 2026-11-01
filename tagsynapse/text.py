"""Tokenisation shared by the BM25 / boolean / TF-IDF paths in the app.

Kept tiny and dependency-free so the lexical search behaviour is testable
without loading a corpus.
"""
import re

import numpy as np

STOPWORDS = frozenset({
    "the", "and", "for", "that", "with", "this", "you", "are", "was", "were",
    "but", "not", "have", "has", "had", "can", "could", "will", "would",
    "about", "from", "they", "them", "their", "there", "here", "into",
    "your", "our", "out", "one", "two", "three", "four", "five",
    "what", "when", "where", "why", "how", "who", "which", "also",
    "just", "like", "more", "some", "any", "all", "because", "than",
    "get", "got", "make", "made", "being", "been", "even", "much",
    "very", "such", "over", "under", "up", "down", "in", "on", "of", "to",
})

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokens, stopwords and anything <= 2 chars dropped."""
    if text is None:
        return []
    toks = _TOKEN.findall(str(text).lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalise each row so a dot product is a cosine. Epsilon-guarded."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-8)
