"""Relation vocabulary for the Tag Knowledge Graph (docs/TAG_MODEL.md §2).

Two relation classes that must stay semantically distinct:

  R_ont  - ontological: broader / narrower / equivalent / associated.
           Asserted by the controlled vocabulary or validated by a human.
  R_emp  - empirical: coUsedWith / usedFor / createdBy.
           Observed from collaborative usage. Evidence, not assertion.

Real edge files carry a zoo of relation strings (FAST relation codes, our own
"user_parent", "canonicalized_to", "cooccurs_with", ...). `relation_bucket`
collapses them into the handful of buckets the UI and the eval code reason
about, so that logic never has to string-match raw relation names.
"""

ONTOLOGICAL_RELATIONS = frozenset({"broader", "narrower", "equivalent", "associated"})
EMPIRICAL_RELATIONS = frozenset({"coUsedWith", "usedFor", "createdBy"})

# Buckets the UI exposes as filters. Order matters for display only.
BUCKETS = ("equivalence", "broader", "narrower", "cooccurs_with", "related")

# Inverse pairs for hierarchy; used by constraint checking (TAG_MODEL §13).
INVERSE = {"broader": "narrower", "narrower": "broader", "equivalence": "equivalence"}


def relation_bucket(rel) -> str:
    """Map any raw relation string onto one of BUCKETS.

    Substring rules, deliberately loose: FAST exports and our own pipeline
    scripts name the same relation several ways ("broader", "parent",
    "SUPER"...). Anything unrecognised is "related" (associative), never a
    hierarchy bucket - a mis-bucketed hierarchy edge is worse than a missed
    one, because it would flow into expansion and inference as a parent link.
    """
    r = str(rel).lower()
    if "equiv" in r or "canon" in r or "alias" in r or "same" in r:
        return "equivalence"
    if "broad" in r or "parent" in r or "super" in r:
        return "broader"
    if "narrow" in r or "child" in r or "sub" in r:
        return "narrower"
    if "cooccur" in r or "coused" in r or "co_used" in r:
        return "cooccurs_with"
    return "related"


def is_hierarchical(bucket: str) -> bool:
    return bucket in ("broader", "narrower")


def is_empirical(bucket: str) -> bool:
    return bucket == "cooccurs_with"
