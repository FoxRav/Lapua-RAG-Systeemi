"""LLM-as-judge: grade a RAG answer via the Anthropic API.

The judge receives (question, RAG answer, optional gold, whether the
system should have abstained) and returns a structured
:class:`JudgeVerdict`. We intentionally keep the contract narrow — the
caller decides how to wire judge results into CI gates (e.g. treat
``correct``/``abstain_correct`` as passes).

Network failures and missing credentials never raise: they collapse to
a fallback verdict with ``score=0`` so a mis-configured eval run fails
closed instead of silently skipping cases.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Final, Literal

import httpx

from lapua_rag.observability import get_logger

_log = get_logger(__name__)

JUDGE_MODEL: Final[str] = "claude-sonnet-4-20250514"
_ANTHROPIC_API_URL: Final[str] = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION: Final[str] = "2023-06-01"
_MAX_SNIPPET_CHARS: Final[int] = 200
_MAX_SOURCES_IN_PROMPT: Final[int] = 3

Verdict = Literal[
    "correct",
    "partial",
    "incorrect",
    "abstain_correct",
    "abstain_wrong",
]
_VALID_VERDICTS: Final[frozenset[str]] = frozenset(
    ("correct", "partial", "incorrect", "abstain_correct", "abstain_wrong")
)

JUDGE_SYSTEM: Final[str] = (
    "Olet RAG-järjestelmän laadunarvioija. Tehtäväsi on arvioida, vastaako "
    "annettu RAG-vastaus kysymykseen oikein Lapuan kaupungin asiakirjojen "
    "perusteella.\n\n"
    "Palauta AINOASTAAN JSON-objekti ilman selityksiä tai koodilohkoa:\n"
    '{"verdict": "correct"|"partial"|"incorrect"|"abstain_correct"|'
    '"abstain_wrong", "score": 0.0-1.0, "reason": "lyhyt perustelu, max 80 merkkiä"}\n\n'
    "Verdict-ohjeet:\n"
    "- correct: vastaus on oikea ja lähdeviite relevantti\n"
    "- partial: vastaus osin oikea tai epätarkka, ei selvästi väärä\n"
    "- incorrect: vastaus on väärä tai harhaanjohtava\n"
    "- abstain_correct: järjestelmä pidättäytyi oikein (kysymys ei kuulu korpukseen)\n"
    "- abstain_wrong: järjestelmä pidättäytyi vaikka korpuksessa oli vastaus"
)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Structured judge response. ``score`` is clamped to [0.0, 1.0]."""

    verdict: Verdict
    score: float
    reason: str

    @property
    def is_acceptable(self) -> bool:
        """True iff the verdict counts as a pass for v1.0 acceptance."""
        return self.verdict in ("correct", "abstain_correct")


def _format_sources(sources: list[Any]) -> str:
    lines: list[str] = []
    for source in sources[:_MAX_SOURCES_IN_PROMPT]:
        if not isinstance(source, dict):
            continue
        doc = source.get("doc_id", "?")
        page = source.get("page", "?")
        snippet = str(source.get("snippet", ""))[:_MAX_SNIPPET_CHARS]
        lines.append(f"- [{doc} s.{page}]: {snippet}")
    return "\n".join(lines) if lines else "  (ei lähteitä)"


def _build_user_prompt(
    *,
    question: str,
    rag_answer: dict[str, Any],
    gold_answer: str | None,
    should_abstain: bool,
) -> str:
    abstained = bool(rag_answer.get("abstained", False))
    johtopaatos = str(rag_answer.get("johtopaatos", "") or "")
    sources = rag_answer.get("lahteet") or []
    lahteet_block = _format_sources(sources if isinstance(sources, list) else [])

    parts: list[str] = [
        f"Kysymys: {question}",
        "",
        "RAG-järjestelmän vastaus:",
        f"- Johtopaatos: {johtopaatos or '(järjestelmä pidättäytyi)'}",
        f"- Abstained: {abstained}",
        "- Lähteet:",
        lahteet_block,
        "",
    ]
    if gold_answer:
        parts.append(f"Odotettu vastaus: {gold_answer}")
    parts.append(f"Pitikö pidättäytyä: {should_abstain}")
    parts.append("")
    parts.append("Arvioi vastauksen oikeellisuus ja palauta JSON.")
    return "\n".join(parts)


def _coerce_verdict(raw: object) -> Verdict:
    if isinstance(raw, str) and raw in _VALID_VERDICTS:
        return raw  # type: ignore[return-value]
    return "incorrect"


def _parse_response(body: str) -> JudgeVerdict:
    cleaned = body.strip()
    # Claude occasionally wraps JSON in ```json ... ``` despite instructions.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response is not a JSON object")
    verdict = _coerce_verdict(parsed.get("verdict"))
    raw_score = parsed.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    reason_raw = parsed.get("reason", "")
    reason = str(reason_raw)[:200] if reason_raw is not None else ""
    return JudgeVerdict(verdict=verdict, score=score, reason=reason)


def judge_answer(
    *,
    question: str,
    rag_answer: dict[str, Any],
    gold_answer: str | None = None,
    should_abstain: bool = False,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> JudgeVerdict:
    """Grade a RAG response via :data:`JUDGE_MODEL`.

    Returns a fallback :class:`JudgeVerdict` (``incorrect`` + score 0)
    on any failure — callers treat that as a CI failure signal without
    the eval run crashing mid-batch.
    """
    key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        _log.warning("judge.no_api_key")
        return JudgeVerdict(
            verdict="incorrect",
            score=0.0,
            reason="Ei API-avainta judge-LLM:lle",
        )

    user_prompt = _build_user_prompt(
        question=question,
        rag_answer=rag_answer,
        gold_answer=gold_answer,
        should_abstain=should_abstain,
    )
    payload = {
        "model": JUDGE_MODEL,
        "max_tokens": 200,
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": _ANTHROPIC_API_VERSION,
        "content-type": "application/json",
    }

    try:
        resp = httpx.post(_ANTHROPIC_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        content = body.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError("Anthropic response missing content blocks")
        text = content[0].get("text", "")
        if not isinstance(text, str) or not text:
            raise ValueError("Empty text in judge response")
        return _parse_response(text)
    except Exception as exc:
        _log.warning("judge.api_error", error=str(exc)[:200])
        return JudgeVerdict(
            verdict="incorrect",
            score=0.0,
            reason=f"API-virhe: {type(exc).__name__}",
        )
