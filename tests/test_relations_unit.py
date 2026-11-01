from tagsynapse.relations import BUCKETS, INVERSE, is_empirical, is_hierarchical, relation_bucket


def test_known_relation_strings_bucket_correctly():
    assert relation_bucket("broader") == "broader"
    assert relation_bucket("USER_PARENT") == "broader"
    assert relation_bucket("narrower") == "narrower"
    assert relation_bucket("subClassOf") == "narrower"
    assert relation_bucket("equivalent") == "equivalence"
    assert relation_bucket("canonicalized_to") == "equivalence"
    assert relation_bucket("cooccurs_with") == "cooccurs_with"
    assert relation_bucket("coUsedWith") == "cooccurs_with"


def test_unknown_relation_falls_back_to_related_not_hierarchy():
    # A mis-bucketed hierarchy edge would flow into expansion as a parent link;
    # anything unrecognised must land in the associative bucket instead.
    for raw in ("", None, "FAST_REL_42", "usedFor", "createdBy"):
        assert relation_bucket(raw) == "related"


def test_every_bucket_is_declared():
    for raw in ("broader", "narrower", "equivalent", "cooccurs_with", "whatever"):
        assert relation_bucket(raw) in BUCKETS


def test_hierarchy_predicates_and_inverse():
    assert is_hierarchical("broader") and is_hierarchical("narrower")
    assert not is_hierarchical("related") and not is_hierarchical("cooccurs_with")
    assert is_empirical("cooccurs_with") and not is_empirical("broader")
    assert INVERSE["broader"] == "narrower" and INVERSE["narrower"] == "broader"
