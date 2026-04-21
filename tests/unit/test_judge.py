"""Unit tests for :mod:`lapua_rag.eval.judge`.

No real network calls are made — :func:`httpx.post` is patched and we
assert both the happy path (verdict parsing, score clamping) and the
failure-mode fallbacks (missing key, API error, malformed JSON).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lapua_rag.eval.judge import JudgeVerdict, _parse_response, judge_answer


def _anthropic_response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"content": [{"text": json.dumps(payload)}]}
    return resp


def test_judge_correct_returns_acceptable_verdict() -> None:
    with patch(
        "lapua_rag.eval.judge.httpx.post",
        return_value=_anthropic_response(
            {"verdict": "correct", "score": 0.95, "reason": "Oikea"},
        ),
    ):
        verdict = judge_answer(
            question="Kuka on kaupunginjohtaja?",
            rag_answer={"johtopaatos": "Satu Kankare", "abstained": False, "lahteet": []},
            api_key="sk-test",
        )
    assert verdict.verdict == "correct"
    assert verdict.score == 0.95
    assert verdict.is_acceptable is True


def test_judge_abstain_correct_is_acceptable() -> None:
    with patch(
        "lapua_rag.eval.judge.httpx.post",
        return_value=_anthropic_response(
            {"verdict": "abstain_correct", "score": 1.0, "reason": "ok"},
        ),
    ):
        verdict = judge_answer(
            question="Mikä on Helsingin pormestarin nimi?",
            rag_answer={"abstained": True, "lahteet": []},
            should_abstain=True,
            api_key="sk-test",
        )
    assert verdict.verdict == "abstain_correct"
    assert verdict.is_acceptable is True


def test_judge_incorrect_is_not_acceptable() -> None:
    with patch(
        "lapua_rag.eval.judge.httpx.post",
        return_value=_anthropic_response(
            {"verdict": "incorrect", "score": 0.1, "reason": "Väärin"},
        ),
    ):
        verdict = judge_answer(
            question="Kuka on kaupunginjohtaja?",
            rag_answer={"johtopaatos": "Väärä nimi", "abstained": False},
            api_key="sk-test",
        )
    assert verdict.verdict == "incorrect"
    assert verdict.is_acceptable is False


def test_judge_missing_api_key_returns_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    verdict = judge_answer(
        question="Testi?",
        rag_answer={"abstained": False},
        api_key=None,
    )
    assert verdict.verdict == "incorrect"
    assert verdict.score == 0.0
    assert "api-avainta" in verdict.reason.lower()


def test_judge_network_error_returns_fallback() -> None:
    with patch(
        "lapua_rag.eval.judge.httpx.post",
        side_effect=httpx.ConnectError("timeout"),
    ):
        verdict = judge_answer(
            question="Testi?",
            rag_answer={"abstained": False},
            api_key="sk-test",
        )
    assert verdict.verdict == "incorrect"
    assert verdict.score == 0.0
    assert "API-virhe" in verdict.reason


def test_judge_malformed_json_returns_fallback() -> None:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"content": [{"text": "not even json"}]}
    with patch("lapua_rag.eval.judge.httpx.post", return_value=resp):
        verdict = judge_answer(
            question="Testi?",
            rag_answer={"abstained": False},
            api_key="sk-test",
        )
    assert verdict.verdict == "incorrect"
    assert verdict.score == 0.0


def test_parse_response_clamps_score_to_unit_interval() -> None:
    """``score > 1`` or ``< 0`` must clamp to [0, 1]."""
    high = _parse_response('{"verdict": "correct", "score": 2.5, "reason": "x"}')
    low = _parse_response('{"verdict": "correct", "score": -1.0, "reason": "x"}')
    assert high.score == 1.0
    assert low.score == 0.0


def test_parse_response_handles_fenced_json() -> None:
    fenced = '```json\n{"verdict": "partial", "score": 0.5, "reason": "ok"}\n```'
    verdict = _parse_response(fenced)
    assert verdict.verdict == "partial"
    assert verdict.score == 0.5


def test_parse_response_unknown_verdict_falls_back_to_incorrect() -> None:
    verdict = _parse_response('{"verdict": "???", "score": 0.5, "reason": ""}')
    assert verdict.verdict == "incorrect"


def test_judge_verdict_dataclass_is_frozen() -> None:
    v = JudgeVerdict(verdict="correct", score=1.0, reason="ok")
    with pytest.raises(AttributeError):
        v.score = 0.5  # type: ignore[misc]
