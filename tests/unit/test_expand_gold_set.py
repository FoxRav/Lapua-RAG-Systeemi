"""Unit tests for scripts/expand_gold_set.py.

Covers the pure helpers (classification, serialisation, assembly) so
we don't need a live Qdrant instance during the test run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from scripts import expand_gold_set


class TestExtractGoldFromChunk:
    def test_selects_decision_sentence(self) -> None:
        chunk = {
            "text": (
                "## Päätös\n"
                "Kaupunginhallitus valittiin puheenjohtajaksi Kai Pöntinen "
                "kaudelle 2025-2028."
            ),
            "doc_type": "poytakirja",
            "section": "## Päätös",
            "section_id": "§12",
            "doc_id": "abc12345",
        }
        item = expand_gold_set.extract_gold_from_chunk(chunk)
        assert item is not None
        assert "valittiin" in item.question.lower()
        assert item.should_abstain is False
        assert item.source_doc_id == "abc12345"
        assert "§12" in item.question  # disambiguator present
        assert item.expected_contains  # at least one
        assert "Kai Pöntinen" in item.expected_contains[0]

    def test_returns_none_without_decision_keyword(self) -> None:
        chunk = {
            "text": "Pitkä teksti jossa kerrotaan osallistujalista ja aikataulu " * 3,
            "doc_type": "poytakirja",
            "section": "Saapuvillaolleet",
            "doc_id": "xyz789",
        }
        assert expand_gold_set.extract_gold_from_chunk(chunk) is None

    def test_returns_none_for_empty_chunk(self) -> None:
        assert expand_gold_set.extract_gold_from_chunk({"text": ""}) is None

    def test_uses_doc_type_when_section_missing(self) -> None:
        chunk = {
            "text": "Kunnanhallitus valittiin järjestäytymisvaiheessa "
                    "uusi puheenjohtaja kauden ajaksi.",
            "doc_type": "poytakirja",
            "section": "",
            "doc_id": "d1",
        }
        item = expand_gold_set.extract_gold_from_chunk(chunk)
        assert item is not None
        assert "poytakirja" in item.question.lower()


class TestBuildOfftopicItems:
    def test_all_marked_abstain(self) -> None:
        items = expand_gold_set.build_offtopic_items()
        assert len(items) >= 3
        assert all(item.should_abstain is True for item in items)
        assert all(item.expected_contains == [] for item in items)


class TestAssemble:
    def _sample_chunk(self, doc_id: str) -> dict[str, object]:
        return {
            "text": (
                "## Päätös\n"
                f"Kaupunginhallitus {doc_id} hyväksyttiin talousarvio "
                f"vuodelle 2026 kokonaisuudessaan."
            ),
            "doc_type": "poytakirja",
            "section": "## Päätös",
            "section_id": f"§{doc_id[-2:]}",
            "doc_id": doc_id,
        }

    def test_appends_offtopic_then_extract_until_target(self) -> None:
        chunks = [self._sample_chunk(f"doc{i}") for i in range(10)]
        rows = expand_gold_set.assemble(
            existing=[],
            chunks=chunks,
            target=8,
            rng=random.Random(0),
        )
        assert len(rows) == 8
        abstain_n = sum(1 for r in rows if r["should_abstain"])
        assert abstain_n >= 3  # at least some off-topic seeds included

    def test_preserves_existing_rows(self) -> None:
        existing = [
            {
                "question": "Kuka on kaupunginjohtaja?",
                "expected_contains": ["Kaupunginjohtaja"],
                "doc_type": "poytakirja",
                "should_abstain": False,
            }
        ]
        rows = expand_gold_set.assemble(
            existing=existing,
            chunks=[],
            target=5,
            rng=random.Random(1),
        )
        assert rows[0] == existing[0]

    def test_deduplicates_by_question(self) -> None:
        existing = [
            {
                "question": "Mikä on Tampereen kaupunginjohtajan nimi?",
                "expected_contains": [],
                "doc_type": "any",
                "should_abstain": True,
            }
        ]
        rows = expand_gold_set.assemble(
            existing=existing,
            chunks=[],
            target=10,
            rng=random.Random(2),
        )
        questions = [r["question"] for r in rows]
        assert questions.count("Mikä on Tampereen kaupunginjohtajan nimi?") == 1


class TestLoadExisting:
    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        assert expand_gold_set.load_existing(tmp_path / "missing.jsonl") == []

    def test_reads_jsonl_ignores_blanks(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.jsonl"
        path.write_text(
            json.dumps({"question": "q1", "should_abstain": False}) + "\n\n"
            + json.dumps({"question": "q2", "should_abstain": True}) + "\n",
            encoding="utf-8",
        )
        rows = expand_gold_set.load_existing(path)
        assert len(rows) == 2
        assert rows[0]["question"] == "q1"
