"""Tests for scripts.build_v3_dataset (pure functions only).

The Qdrant chunk source is not exercised here — the dataset builder
accepts a plain sequence of ``Chunk`` so unit tests can feed fixtures
directly.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_v3_dataset.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_v3_dataset", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_v3_dataset"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


def _chunk(
    *,
    text: str,
    section: str = "## Päätös",
    doc_type: str = "poytakirja",
    chunk_id: str = "c1",
) -> object:
    return builder.Chunk(chunk_id=chunk_id, text=text, doc_type=doc_type, section=section)


class TestChunkFromPayload:
    def test_picks_section_title_over_section(self) -> None:
        payload: dict[str, object] = {
            "chunk_id": "abc",
            "text": "body",
            "doc_type": "poytakirja",
            "section_title": "## Päätös § 12",
        }
        chunk = builder.Chunk.from_payload(payload)
        assert chunk.section == "## Päätös § 12"
        assert chunk.chunk_id == "abc"


class TestExtractQuestion:
    def test_returns_none_when_not_paatos_section(self) -> None:
        chunk = _chunk(
            text="Pitkä teksti jossa kerrotaan osallistujalista ja aikataulu.",
            section="## Saapuvillaolleet jäsenet",
        )
        assert builder.extract_question_from_chunk(chunk) is None

    def test_returns_question_for_paatos_with_verb_valittiin(self) -> None:
        chunk = _chunk(
            text=(
                "Kokouksen aluksi esitellään tilanne.\n"
                "Kaupunginhallituksen puheenjohtajaksi valittiin Kai Pöntinen "
                "esityksen mukaisesti.\n"
                "Muita asioita ei käsitelty."
            ),
            section="## Päätös § 12",
        )
        result = builder.extract_question_from_chunk(chunk)
        assert result is not None
        question, quote = result
        assert "Kuka valittiin" in question
        assert "Kai Pöntinen" in quote

    def test_returns_question_for_hyvaksyttiin(self) -> None:
        chunk = _chunk(
            text=(
                "Valmistelun perusteella hyväksyttiin uusi talousarvio ilman "
                "muutoksia esittelijän pohjaan."
            ),
            section="## Päätös § 14",
        )
        result = builder.extract_question_from_chunk(chunk)
        assert result is not None
        assert "hyväksyttiin" in result[0].lower()

    def test_returns_none_when_quote_too_short(self) -> None:
        chunk = _chunk(text="Päätös § 1\nLyhyt.\n", section="## Päätös § 1")
        assert builder.extract_question_from_chunk(chunk) is None


class TestMakeExamples:
    def test_extract_example_shape(self) -> None:
        chunk = _chunk(text="Lainaus kohteesta Lapuan päätöksistä syvemmin")
        example = builder.make_extract_example(chunk, "Kysymys?", "Lainaus kohteesta.")
        assert list(example) == ["messages"]
        roles = [m["role"] for m in example["messages"]]
        assert roles == ["system", "user", "assistant"]
        assistant = json.loads(example["messages"][2]["content"])
        assert assistant == {
            "quote": "Lainaus kohteesta.",
            "chunk_index": 0,
            "no_match": False,
        }

    def test_abstain_example_shape(self) -> None:
        chunk = _chunk(text="mitä tahansa")
        example = builder.make_abstain_example(chunk, "Off-topic?")
        assistant = json.loads(example["messages"][2]["content"])
        assert assistant == {"quote": "", "chunk_index": 0, "no_match": True}


class TestBuildDataset:
    def _corpus(self) -> list[object]:
        return [
            _chunk(
                chunk_id=f"c{i}",
                section="## Päätös § " + str(i),
                text=(
                    f"Asian käsittelyssä {i} valittiin Henkilö-{i} tehtävään "
                    "esityksen mukaisesti ja yksimielisesti."
                ),
            )
            for i in range(10)
        ]

    def test_respects_max_counts(self) -> None:
        examples, counts = builder.build_dataset(
            self._corpus(),
            max_extract=3,
            max_abstain=2,
            rng=random.Random(0),
        )
        assert counts.extract == 3
        assert counts.abstain == 2
        assert len(examples) == 5

    def test_deterministic_with_seed(self) -> None:
        corpus = self._corpus()
        out1, _ = builder.build_dataset(corpus, max_extract=5, max_abstain=3, rng=random.Random(0))
        out2, _ = builder.build_dataset(corpus, max_extract=5, max_abstain=3, rng=random.Random(0))
        assert out1 == out2

    def test_skips_chunks_without_paatos_for_extract(self) -> None:
        non_paatos = [
            _chunk(
                chunk_id=f"x{i}",
                section="## Saapuvillaolleet",
                text="Osallistujat listattu tähän sekä muut seikat pitkästi.",
            )
            for i in range(5)
        ]
        _, counts = builder.build_dataset(
            non_paatos, max_extract=5, max_abstain=0, rng=random.Random(0)
        )
        assert counts.extract == 0

    def test_counts_reports_abstain_pct(self) -> None:
        _, counts = builder.build_dataset(
            self._corpus(), max_extract=4, max_abstain=4, rng=random.Random(0)
        )
        assert counts.total == 8
        assert counts.abstain_pct == 50.0


class TestWriteJsonl:
    def test_roundtrips_examples(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        examples = [
            {"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "b"}]},
        ]
        written = builder.write_jsonl(examples, path)
        assert written == 2
        loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert loaded == examples
