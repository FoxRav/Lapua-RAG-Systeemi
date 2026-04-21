"""Tests for scripts.eval_rag (pure-function scorer paths only).

The HTTP path is not exercised here — we inject a fake query callable
so unit tests don't need the live FastAPI backend.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_rag.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_rag", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_rag"] = module
    spec.loader.exec_module(module)
    return module


eval_module = _load_module()


def _gold(
    *,
    question: str = "Kuka?",
    expected: tuple[str, ...] = (),
    should_abstain: bool = False,
) -> object:
    return eval_module.GoldItem(
        question=question,
        expected_contains=expected,
        doc_type="poytakirja",
        should_abstain=should_abstain,
    )


class TestGoldItem:
    def test_from_json_minimal(self) -> None:
        item = eval_module.GoldItem.from_json(
            {"question": "Kuka?", "should_abstain": False}
        )
        assert item.question == "Kuka?"
        assert item.should_abstain is False
        assert item.doc_type == "any"
        assert item.expected_contains == ()

    def test_from_json_with_expected(self) -> None:
        item = eval_module.GoldItem.from_json(
            {
                "question": "Q",
                "should_abstain": False,
                "expected_contains": ["Kai", "Pöntinen"],
                "doc_type": "poytakirja",
            }
        )
        assert item.expected_contains == ("Kai", "Pöntinen")
        assert item.doc_type == "poytakirja"

    def test_from_json_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="question"):
            eval_module.GoldItem.from_json({"should_abstain": False})

    def test_from_json_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            eval_module.GoldItem.from_json(["not", "a", "dict"])


class TestLoadGold:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        gold_path = tmp_path / "gold.jsonl"
        gold_path.write_text(
            '{"question":"a","should_abstain":false}\n'
            '{"question":"b","should_abstain":true}\n',
            encoding="utf-8",
        )
        items = eval_module.load_gold(gold_path)
        assert len(items) == 2
        assert items[0].question == "a"
        assert items[1].should_abstain is True

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            eval_module.load_gold(tmp_path / "nope.jsonl")

    def test_malformed_line_raises_with_lineno(self, tmp_path: Path) -> None:
        gold_path = tmp_path / "bad.jsonl"
        gold_path.write_text(
            '{"question":"a","should_abstain":false}\n{ malformed\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            eval_module.load_gold(gold_path)


class TestExtractResponseText:
    def test_combines_all_answer_parts(self) -> None:
        answer = {
            "johtopaatos": "Kai Pöntinen",
            "perustelut": "§12 mukaan",
            "lahteet": [
                {"snippet": "Kokouksen päätös"},
                {"snippet": "Allekirjoitettu 2026"},
            ],
        }
        body = eval_module.extract_response_text(answer)
        assert "kai pöntinen" in body
        assert "§12" in body
        assert "kokouksen päätös" in body
        assert "allekirjoitettu 2026" in body


class TestScoreAnswer:
    def test_correct_extract_when_expected_substrings_present(self) -> None:
        gold = _gold(expected=("Kai Pöntinen",))
        answer = {
            "johtopaatos": "Kai Pöntinen valittiin",
            "abstained": False,
            "max_source_score": 0.95,
        }
        result = eval_module.score_answer(answer, gold, latency_s=1.0)
        assert result.abstain_correct is True
        assert result.extract_correct is True
        assert result.max_source_score == pytest.approx(0.95)

    def test_incorrect_extract_when_expected_missing(self) -> None:
        gold = _gold(expected=("Satu Kankare",))
        answer = {"johtopaatos": "Joku muu", "abstained": False}
        result = eval_module.score_answer(answer, gold, latency_s=0.5)
        assert result.extract_correct is False

    def test_abstained_matches_gold_should_abstain(self) -> None:
        gold = _gold(should_abstain=True)
        answer = {"abstained": True}
        result = eval_module.score_answer(answer, gold, latency_s=0.1)
        assert result.abstain_correct is True
        assert result.extract_correct is None

    def test_abstain_mismatch_counted_wrong(self) -> None:
        gold = _gold(should_abstain=True)
        answer = {"abstained": False, "johtopaatos": "hölynpölyä"}
        result = eval_module.score_answer(answer, gold, latency_s=0.1)
        assert result.abstain_correct is False
        assert result.extract_correct is None


class TestRunEval:
    def test_happy_path_summary(self) -> None:
        gold = [
            _gold(question="Q1", expected=("hit",)),
            _gold(question="Q2", should_abstain=True),
        ]

        def fake_query(*, question: str, mode: str) -> dict[str, object]:
            if question == "Q1":
                return {"johtopaatos": "perfect hit here", "abstained": False}
            return {"abstained": True}

        ticks = iter([0.0, 1.0, 2.0, 3.0])
        clock = lambda: next(ticks)  # noqa: E731
        summary = eval_module.run_eval(gold, fake_query, "extract", clock=clock)
        assert summary.errors == 0
        assert summary.abstain_accuracy == pytest.approx(1.0)
        assert summary.extract_accuracy == pytest.approx(1.0)

    def test_http_error_recorded_but_batch_continues(self) -> None:
        gold = [_gold(question="Q1"), _gold(question="Q2")]

        def flaky_query(*, question: str, mode: str) -> dict[str, object]:
            if question == "Q1":
                raise httpx.ConnectError("connection refused")
            return {"abstained": False, "johtopaatos": "done"}

        summary = eval_module.run_eval(gold, flaky_query, "extract")
        assert summary.errors == 1
        assert len(summary.valid) == 1

    def test_format_summary_reports_ok_or_not(self) -> None:
        gold = [_gold(question="Q", expected=("x",))]
        answer = {"johtopaatos": "x", "abstained": False}

        def fake_query(*, question: str, mode: str) -> dict[str, object]:
            return answer

        summary = eval_module.run_eval(gold, fake_query, "extract")
        rendered = eval_module.format_summary(summary)
        assert "TULOKSET (moodi: extract)" in rendered
        assert "Abstain-tarkkuus" in rendered
        assert "Extract-tarkkuus" in rendered


class TestRunCliMissingGold:
    def test_exit_code_2_when_gold_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = eval_module.run_cli(["--gold", str(tmp_path / "missing.jsonl")])
        assert rc == 2
        assert "Kultajoukko" in capsys.readouterr().out


class TestGoldSeedFile:
    def test_repo_seed_is_valid(self) -> None:
        """Make sure the committed gold seed parses cleanly."""
        seed = Path(__file__).resolve().parents[1] / "fixtures" / "gold" / "lapua_gold_v1.jsonl"
        items = eval_module.load_gold(seed)
        assert len(items) >= 4
        assert any(item.should_abstain for item in items)
        assert any(not item.should_abstain for item in items)
        for line in seed.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parsed = json.loads(line)
                assert set(parsed).issuperset({"question", "should_abstain", "expected_contains"})
