"""Central config. Nothing clever here on purpose - one file, one glance to see
every knob that changes how a run behaves, and every path the pipeline reads
or writes.

Before this file existed there were four different path conventions across
the repo, three of them hardcoded to a specific machine (a laptop, a TACC
allocation, a Lightsail box). Everything now resolves from TAGSYNAPSE_DATA_DIR,
which defaults to `<repo>/faiss/` so an existing checkout keeps working with
no .env at all.
"""
import os
from pathlib import Path

try:  # optional - the app and scripts work without python-dotenv installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Data layout -----------------------------------------------------------
# <DATA_DIR>/
#   index/     corpus CSVs + FAISS .index files  (retrieval/build_filtered_faiss.py)
#   tagging/   controlled vocab, KG, user tags, GNN outputs (tagging/*.py, the app)
DATA_DIR = Path(os.getenv("TAGSYNAPSE_DATA_DIR", REPO_ROOT / "faiss")).expanduser()
INDEX_DIR = DATA_DIR / "index"
TAG_DATA_DIR = DATA_DIR / "tagging"

# Raw inputs for the retrieval build (the *_filtered_output.jsonl files).
# Only build_filtered_faiss.py reads this.
RAW_JSONL_DIR = Path(os.getenv("TAGSYNAPSE_RAW_JSONL_DIR", DATA_DIR / "raw")).expanduser()

# --- Models ----------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv(
    "TAGSYNAPSE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
SEED = int(os.getenv("TAGSYNAPSE_SEED", "42"))

# --- Services --------------------------------------------------------------
# The Streamlit app talks to the FAISS service over HTTP (retrieval/faiss_service.py).
FAISS_API_URL = os.getenv("TAGSYNAPSE_FAISS_URL", "http://127.0.0.1:5000/search")
# Inside the faiss-service container the index dir is mounted read-only here.
FAISS_INDEX_PATH = os.getenv(
    "TAGSYNAPSE_FAISS_INDEX_PATH",
    "/mnt/faiss_index/artificial_filtered_output_faiss.index",
)

# --- Corpora ---------------------------------------------------------------
# Each key maps to candidate corpus files, most-enriched first. The app uses
# the first one that exists. Paths are relative to INDEX_DIR.
SUBREDDIT_CONFIG = {
    "artificial_filtered": {
        "label": "r/artificial – filtered propositions",
        "corpus_candidates": [
            "artificial_filtered_output_corpus_with_topics_labeled.csv",
            "artificial_filtered_output_corpus_with_topics.csv",
            "artificial_filtered_output_corpus.csv",
            "filtered_index/artificial_filtered_output_corpus_with_topics_labeled.csv",
            "filtered_index/artificial_filtered_output_corpus_with_topics.csv",
            "filtered_index/artificial_filtered_output_corpus.csv",
        ],
    },
    "artificialint_filtered": {
        "label": "r/artificialintelligence – filtered propositions",
        "corpus_candidates": [
            "artificialint_filtered_output_corpus_with_topics_labeled.csv",
            "artificialint_filtered_output_corpus_with_topics.csv",
            "artificialint_filtered_output_corpus.csv",
            "filtered_index/artificialint_filtered_output_corpus_with_topics_labeled.csv",
            "filtered_index/artificialint_filtered_output_corpus_with_topics.csv",
            "filtered_index/artificialint_filtered_output_corpus.csv",
        ],
    },
    "singularity_filtered": {
        "label": "r/singularity – filtered propositions",
        "corpus_candidates": [
            "singularity_filtered_output_corpus_with_topics_labeled.csv",
            "singularity_filtered_output_corpus_with_topics.csv",
            "singularity_filtered_output_corpus.csv",
            "filtered_index/singularity_filtered_output_corpus_with_topics_labeled.csv",
            "filtered_index/singularity_filtered_output_corpus_with_topics.csv",
            "filtered_index/singularity_filtered_output_corpus.csv",
        ],
    },
}

# --- Search knobs (Streamlit app) -----------------------------------------
TFIDF_MAX_FEATURES = 5000
TFIDF_MAX_DOCS = 100_000
BM25_MAX_DOCS = 100_000

# --- Tagging-dir files -----------------------------------------------------
# Controlled vocabulary + base KG (inputs, produced by tagging/build_tags_embeddings_and_kg.py)
TAG_VOCAB_PATH = TAG_DATA_DIR / "tags_vocab_cat.csv"
TAG_EDGES_CAT_PATH = TAG_DATA_DIR / "tags_edges_cat.csv"
TAG_VOCAB_INDEX_PATH = TAG_DATA_DIR / "tags_vocab_index.csv"
TAG_VOCAB_EMB_PATH = TAG_DATA_DIR / "tags_vocab_embeddings.npy"
KG_EDGES_INDEX_PATH = TAG_DATA_DIR / "kg_edges_index.csv"

# Full KG (tagging/build_tag_kg.py)
KG_NODES_PATH = TAG_DATA_DIR / "kg_nodes.csv"
KG_EDGES_PATH = TAG_DATA_DIR / "kg_edges.csv"
KG_PROP_EDGES_PATH = TAG_DATA_DIR / "kg_prop_edges.csv"

# Runtime tagging state (written by the app as users tag)
USER_TAGS_PATH = TAG_DATA_DIR / "user_tags.csv"
TAG_ASSIGN_PATH = TAG_DATA_DIR / "user_tag_assignments.csv"
TAG_CANON_LOG_PATH = TAG_DATA_DIR / "tag_canonicalization_events.csv"

# Usage-enriched graph (tagging/build_usage_enriched_tag_kg.py + the app)
TAG_COOCCUR_PATH = TAG_DATA_DIR / "tag_cooccurrence_edges.csv"
USED_NODES_PATH = TAG_DATA_DIR / "used_tag_nodes.csv"
USED_EDGES_PATH = TAG_DATA_DIR / "used_tag_edges.csv"
USED_EDGES_INDEX_PATH = TAG_DATA_DIR / "used_tag_edges_index.csv"
USER_TAG_EMB_PATH = TAG_DATA_DIR / "user_tag_embeddings.npy"
TAG_CENTRALITY_PATH = TAG_DATA_DIR / "tag_centrality_scores.csv"
HYBRID_GRAPH_PATH = TAG_DATA_DIR / "hybrid_tag_graph_edges.csv"

# GNN (tagging/train_light_tag_gnn.py)
TAG_GNN_PRED_PATH = TAG_DATA_DIR / "gnn_tag_predictions.csv"
TAG_GNN_MODEL_PATH = TAG_DATA_DIR / "light_tag_gnn.pt"


def snapshot() -> dict:
    """Every knob that changes a run, as a plain dict - recorded alongside each
    eval snapshot so a metric change is never ambiguous about its cause."""
    return {
        "data_dir": str(DATA_DIR),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "faiss_api_url": FAISS_API_URL,
        "seed": SEED,
    }
