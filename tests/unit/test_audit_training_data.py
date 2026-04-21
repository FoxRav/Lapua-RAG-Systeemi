"""Tests for scripts.audit_training_data.

We import the module via a path-based loader so the test suite doesn't
depend on ``scripts/`` being on sys.path at collection time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_training_data.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_training_data", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_training_data"] = module
    spec.loader.exec_module(module)
    return module


audit_module = _load_module()


def _chatml(assistant: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Kuka on kaupunginjohtaja?"},
            {"role": "assistant", "content": assistant},
        ]
    }


class TestClassifyExample:
    def test_json_abstained_true_is_abstain(self) -> None:
        ex = _chatml(json.dumps({"abstained": True, "johtopaatos": "x"}))
        assert audit_module.classify_example(ex) == audit_module.LABEL_ABSTAIN

    def test_json_no_match_true_is_abstain(self) -> None:
        ex = _chatml(json.dumps({"quote": "", "chunk_index": 0, "no_match": True}))
        assert audit_module.classify_example(ex) == audit_module.LABEL_ABSTAIN

    def test_json_with_quote_is_extract(self) -> None:
        ex = _chatml(json.dumps({"quote": "Kai Pöntinen valittiin.", "no_match": False}))
        assert audit_module.classify_example(ex) == audit_module.LABEL_EXTRACT

    def test_json_with_johtopaatos_is_extract(self) -> None:
        ex = _chatml(json.dumps({"johtopaatos": "Kai Pöntinen.", "perustelut": "…"}))
        assert audit_module.classify_example(ex) == audit_module.LABEL_EXTRACT

    def test_plain_text_refusal_is_abstain(self) -> None:
        ex = _chatml("En löydä Systeemistä vastausta tähän kysymykseen.")
        assert audit_module.classify_example(ex) == audit_module.LABEL_ABSTAIN

    def test_plain_text_long_answer_is_extract(self) -> None:
        ex = _chatml("Satu Kankare on Lapuan kaupunginjohtaja valtuuston päätöksen mukaisesti.")
        assert audit_module.classify_example(ex) == audit_module.LABEL_EXTRACT

    def test_short_opaque_text_is_unknown(self) -> None:
        ex = _chatml("ok")
        assert audit_module.classify_example(ex) == audit_module.LABEL_UNKNOWN

    def test_missing_assistant_is_unknown(self) -> None:
        assert audit_module.classify_example({"messages": []}) == audit_module.LABEL_UNKNOWN

    def test_non_dict_input_is_unknown(self) -> None:
        assert audit_module.classify_example("not a dict") == audit_module.LABEL_UNKNOWN


class TestAuditFile:
    def test_counts_mixed_labels(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "train.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps(_chatml(json.dumps({"abstained": True}))),
                    json.dumps(_chatml(json.dumps({"quote": "long enough quote here"}))),
                    "   ",  # blank — skipped
                    "{not valid json",  # parse_error
                ]
            ),
            encoding="utf-8",
        )
        counts = audit_module.audit_file(jsonl)
        assert counts[audit_module.LABEL_ABSTAIN] == 1
        assert counts[audit_module.LABEL_EXTRACT] == 1
        assert counts[audit_module.LABEL_PARSE_ERROR] == 1


class TestAuditReport:
    def _report(self, abstain: int, extract: int) -> object:
        counts: Counter[str] = Counter(
            {audit_module.LABEL_ABSTAIN: abstain, audit_module.LABEL_EXTRACT: extract}
        )
        return audit_module.AuditReport(counts=counts, files_scanned=1)

    def test_verdict_warns_when_abstain_dominant(self) -> None:
        report = self._report(abstain=70, extract=30)
        assert "retrain" in report.verdict.lower()

    def test_verdict_warns_when_extract_dominant(self) -> None:
        report = self._report(abstain=20, extract=80)
        assert "liian dominoiva" in report.verdict

    def test_verdict_ok_when_balanced(self) -> None:
        report = self._report(abstain=50, extract=50)
        assert "OK" in report.verdict

    def test_verdict_empty_when_no_data(self) -> None:
        report = audit_module.AuditReport(counts=Counter(), files_scanned=0)
        assert "Ei dataa" in report.verdict


class TestRunCli:
    def test_exit_code_2_when_nothing_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = audit_module.run_cli(["--path", str(tmp_path / "missing")])
        assert rc == 2
        assert "Koulutustiedostoja ei löydy" in capsys.readouterr().out

    def test_exit_code_0_with_data(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        jsonl = tmp_path / "a.jsonl"
        jsonl.write_text(
            json.dumps(_chatml(json.dumps({"quote": "Lainaus pitkä teksti Lapualta."}))) + "\n",
            encoding="utf-8",
        )
        rc = audit_module.run_cli(["--path", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "YHTEENSÄ: 1" in out
        assert "extract" in out
