"""TagSynapse: structure without vocabulary homogenization.

Importable core that the Streamlit app (app/), the pipeline scripts
(retrieval/, tagging/) and the evaluation code (eval/) all share:

    config     - every path and knob, env-overridable, one file
    relations  - R_ont / R_emp vocabulary + raw-relation bucketing
    usage      - co-use counts, Jaccard, PMI, behavioral subsumption
    text       - tokenisation + stopwords shared by the lexical search paths
    store      - read/write the runtime tagging CSVs

Nothing in here imports streamlit, torch or sentence-transformers, so the
unit tests run anywhere pandas does.
"""
__version__ = "0.2.0"
