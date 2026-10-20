"""Heuristic scorers of evaluation/evaluate.py (no judge, no LLM)."""

import evaluate as ev

ISHARES, UBS = "IE00B4L5Y983", "IE00BD4TXV59"


def _chunk(isin, doc_type, text="", year=2026):
    return {"etf_isin": isin, "doc_type": doc_type, "year": year, "text": text}


SOURCES = [
    {"etf_isin": ISHARES, "issuer": "ishares", "doc_type": "factsheet", "year": 2026},
    {"etf_isin": UBS, "issuer": "ubs", "doc_type": "factsheet", "year": 2026},
]


def test_retrieval_precision_and_recall():
    question = {"qid": "T2_X", "query_type": 2, "sources": SOURCES}
    entry = {"retrieved_chunks": [_chunk(ISHARES, "factsheet"), _chunk(ISHARES, "kid"),
                                  _chunk(ISHARES, "factsheet", year=2025)]}
    r = ev.evaluate_retrieval(question, entry)
    assert r.precision_at_k == 0.333
    assert r.recall_at_k == 0.5
    assert r.any_relevant_retrieved


def test_retrieval_nothing_retrieved():
    r = ev.evaluate_retrieval({"qid": "Q", "query_type": 1, "sources": SOURCES}, {})
    assert (r.precision_at_k, r.recall_at_k, r.any_relevant_retrieved) == (0.0, 0.0, False)


def test_chunk_value_recall_scopes_isin_labels():
    question = {"expected_values": {ISHARES: ["0.20%"], UBS: ["0.06%"]}}
    retrieved = [_chunk(ISHARES, "factsheet", "TER 0.20%"), _chunk(UBS, "factsheet", "TER 0.20%")]
    assert ev.chunk_value_recall(question, retrieved) == 0.5
    assert ev.chunk_value_recall({}, retrieved) is None


def test_value_present_numeric_and_alternatives():
    assert ev.value_present("0.2%", "the TER is 0.20 %")
    assert ev.value_present("0,20%", "TER 0.20%")
    assert not ev.value_present("0.2%", "TER 0.2")                  # percent must match
    assert ev.value_present(["full replication", "fully replicated"], "It is Fully  Replicated.")
    assert not ev.value_present("sampling", "physical replication")


def test_extract_numbers_skips_isins_years_and_small_integers():
    nums = ev.extract_numbers("IE00B4L5Y983 in 2026: SRI 4, TER 0.20%, AUM 1,309", skip_trivial=True)
    assert [(v, p) for v, p, _ in nums] == [(0.20, True), (1309.0, False)]


def test_expected_values_type1_and_abstention_sentences():
    question = {"query_type": 1, "expected_values": ["0.20%"]}
    found, _ = ev._answer_contains_expected(question, "The TER is 0.20% [IE00B4L5Y983 | ishares | factsheet | 2026].")
    assert found
    # a value listed in a "not stated" sentence does not count
    found, _ = ev._answer_contains_expected(question, "The documents do not state whether it is 0.20%.")
    assert not found


def test_expected_values_type2_checks_each_etf_segment():
    question = {"query_type": 2, "sources": SOURCES,
                "expected_values": {ISHARES: ["0.20%"], UBS: ["0.06%"]}}
    good = "iShares charges 0.20%. UBS charges 0.06%."
    swapped = "iShares charges 0.06%. UBS charges 0.20%."
    assert ev._answer_contains_expected(question, good)[0]
    assert not ev._answer_contains_expected(question, swapped)[0]


def test_unsupported_numbers():
    retrieved = [_chunk(ISHARES, "factsheet", "TER 0.20% and 1309 holdings")]
    assert ev._unsupported_numbers("TER 0.2%, 1,309 holdings, return 12.5%", retrieved) == ["12.5%"]


def test_rubric_keywords_and_min_match():
    rubric = {"must_mention": [
        {"item": "costs", "keywords": ["TER", ["0.20%", "0.2%"]], "min_match": 2},
        {"item": "risk", "keywords": ["SRI", "volatility"], "min_match": 1},
    ], "must_not_contain": ["advice"]}
    scores = ev._score_rubric("TER 0.2% and an SRI of 4.", rubric)
    assert scores["must_mention"] == {"costs": True, "risk": True}
    assert scores["summary"]["overall_pass"] is True
    assert scores["must_not_contain"] == {"advice": None}


def test_attribution():
    question = {"qid": "Q", "query_type": 2, "sources": SOURCES}
    cited = [{"etf_isin": ISHARES, "issuer": "ishares", "doc_type": "factsheet", "year": 2026},
             {"etf_isin": UBS, "issuer": "ubs", "doc_type": "kid", "year": 2026}]
    a = ev.evaluate_attribution(question, {"cited_sources": cited})
    assert a.attribution_score == 0.5
    assert a.wrong_sources_cited and not a.correct_sources_cited
