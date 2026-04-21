"""Unit tests for :mod:`lapua_rag.retrieve.query_rewrite`.

The rewriter is pure and deterministic — no network, no LLM — so the
tests just pin each heuristic's behaviour on known shapes of Finnish
municipal queries.
"""

from __future__ import annotations

from lapua_rag.retrieve.query_rewrite import (
    _MAX_REWRITES,
    heuristic_rewrites,
    rewrite_query,
)


def test_original_query_is_always_first() -> None:
    original = "Kuka on kaupunginjohtaja?"
    rewrites = heuristic_rewrites(original)
    assert rewrites[0] == original


def test_strips_leading_question_word() -> None:
    rewrites = heuristic_rewrites("Kuka on kaupunginjohtaja?")
    # Either plain noun (from _extract_who_is) or the question-word
    # stripped variant ("on kaupunginjohtaja") — both encode the topic
    # without the interrogative head.
    assert any("kaupunginjohtaja" in r and not r.startswith("Kuka") for r in rewrites)


def test_adds_domain_tag_when_missing() -> None:
    rewrites = heuristic_rewrites("Kuka on kaupunginjohtaja?")
    assert any("Lapuan kaupunki" in r for r in rewrites)


def test_does_not_add_domain_tag_when_present() -> None:
    rewrites = heuristic_rewrites("Lapuan kaupunginjohtaja")
    tagged = [r for r in rewrites if r.lower().startswith("lapuan kaupunki ")]
    # Original already contains 'lapuan' → no extra tagged variant.
    assert tagged == []


def test_extracts_noun_phrase_from_who_is() -> None:
    rewrites = heuristic_rewrites("Kuka on Satu Kankare?")
    assert "Satu Kankare" in rewrites


def test_caps_at_three_rewrites() -> None:
    rewrites = heuristic_rewrites("Kuka on Lapuan kaupunginjohtaja tällä hetkellä?")
    assert len(rewrites) <= _MAX_REWRITES


def test_deduplicates_identical_variants() -> None:
    rewrites = heuristic_rewrites("Lapuan kaupunki")
    assert len(rewrites) == len({r.lower() for r in rewrites})


def test_rewrite_query_returns_non_empty_list() -> None:
    result = rewrite_query("Mitä päätettiin talousarviosta?")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0] == "Mitä päätettiin talousarviosta?"


def test_rewrite_query_is_deterministic() -> None:
    a = rewrite_query("Kuka on kaupunginjohtaja?")
    b = rewrite_query("Kuka on kaupunginjohtaja?")
    assert a == b


def test_non_question_input_still_has_domain_tag_rewrite() -> None:
    """Topic-phrase-only input gets a domain-tagged companion."""
    rewrites = heuristic_rewrites("talousarviokäsittely 2025")
    assert any("Lapuan kaupunki" in r for r in rewrites)
