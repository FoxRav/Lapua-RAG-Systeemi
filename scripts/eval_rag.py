"""Evaluate the RAG backend against a gold set.

Measures three things on a JSONL gold file (see
``tests/fixtures/gold/lapua_gold_v1.jsonl``):

* **Abstain-correctness** — did the service abstain when it should have,
  and answer when it should have.
* **Extract-correctness** — for non-abstain cases, do all
  ``expected_contains`` substrings appear in the answer body.
* **Latency** — per-question wall-clock from POST /v1/query.

The response-scoring primitives (``extract_response_text``,
``score_answer``) are pure functions so we can unit-test them without
standing up an HTTP server.

Usage::

    python scripts/eval_rag.py                          # default: extract mode
    python scripts/eval_rag.py --mode synth
    python scripts/eval_rag.py --gold path/to/gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, Literal, Protocol

import httpx

DEFAULT_API: Final[str] = "http://127.0.0.1:8080"
DEFAULT_GOLD: Final[Path] = Path("tests/fixtures/gold/lapua_gold_v1.jsonl")
AnswerMode = Literal["extract", "retrieve", "synth"]
DEFAULT_MODE: Final[AnswerMode] = "extract"

# Hyväksymisrajat v1.0:n acceptance-kriteereistä (README §5).
_ABSTAIN_TARGET: Final[float] = 0.80
_EXTRACT_TARGET: Final[float] = 0.70
# LLM-as-judge kynnys v1.0:ssä (CURSOR_v0.9_OHJE.md §C): ≥ 80 % oikeellinen.
_JUDGE_TARGET: Final[float] = 0.80
# Model name inline'd to avoid a hard import path from this script —
# keeps `python scripts/eval_rag.py --help` cheap on systems that don't
# have httpx available (although we currently always import httpx).
_JUDGE_MODEL_NAME: Final[str] = "claude-sonnet-4-20250514"


@dataclass(frozen=True, slots=True)
class GoldItem:
    """One row of the gold JSONL."""

    question: str
    expected_contains: tuple[str, ...]
    doc_type: str
    should_abstain: bool

    @classmethod
    def from_json(cls, raw: object) -> GoldItem:
        if not isinstance(raw, dict):
            raise ValueError(f"Gold row must be a JSON object, got {type(raw).__name__}")
        try:
            question = str(raw["question"])
            should_abstain = bool(raw["should_abstain"])
        except KeyError as err:
            raise ValueError(f"Gold row missing required field: {err.args[0]}") from err
        expected_raw = raw.get("expected_contains", [])
        if not isinstance(expected_raw, list):
            raise ValueError("expected_contains must be a list")
        expected = tuple(str(s) for s in expected_raw)
        doc_type = str(raw.get("doc_type", "any"))
        return cls(
            question=question,
            expected_contains=expected,
            doc_type=doc_type,
            should_abstain=should_abstain,
        )


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Per-question outcome."""

    question: str
    abstained: bool
    should_abstain: bool
    abstain_correct: bool
    extract_correct: bool | None
    max_source_score: float
    latency_s: float
    error: str | None = None
    # LLM-as-judge fields (None when --judge is off or the call failed
    # before reaching judge_answer).
    judge_verdict: str | None = None
    judge_score: float | None = None
    judge_reason: str | None = None


@dataclass(slots=True)
class EvalSummary:
    """Aggregate result of an eval run."""

    mode: AnswerMode
    results: list[EvalResult] = field(default_factory=list)

    @property
    def valid(self) -> list[EvalResult]:
        return [r for r in self.results if r.error is None]

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    @property
    def abstain_accuracy(self) -> float:
        if not self.valid:
            return 0.0
        return sum(1 for r in self.valid if r.abstain_correct) / len(self.valid)

    @property
    def extract_accuracy(self) -> float | None:
        rated = [r for r in self.valid if r.extract_correct is not None]
        if not rated:
            return None
        return sum(1 for r in rated if r.extract_correct) / len(rated)

    @property
    def average_latency_s(self) -> float:
        if not self.valid:
            return 0.0
        return sum(r.latency_s for r in self.valid) / len(self.valid)

    @property
    def judged(self) -> list[EvalResult]:
        return [r for r in self.valid if r.judge_verdict is not None]

    @property
    def judge_accuracy(self) -> float | None:
        judged = self.judged
        if not judged:
            return None
        acceptable = sum(
            1
            for r in judged
            if r.judge_verdict in ("correct", "abstain_correct")
        )
        return acceptable / len(judged)


def load_gold(path: Path) -> list[GoldItem]:
    """Load the JSONL gold set; raises ``FileNotFoundError`` if missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Kultajoukko ei löydy: {path}")
    items: list[GoldItem] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as err:
            raise ValueError(f"{path}:{lineno}: rikkinäinen JSON: {err}") from err
        items.append(GoldItem.from_json(parsed))
    return items


def extract_response_text(answer: dict[str, object]) -> str:
    """Flatten a RagAnswer payload into a single lowercased search blob.

    We compare against ``johtopaatos``, ``perustelut`` and every source
    ``snippet`` so the gold ``expected_contains`` matches regardless of
    which part of the answer surfaced the fact.
    """
    chunks: list[str] = []
    for key in ("johtopaatos", "perustelut"):
        value = answer.get(key)
        if isinstance(value, str):
            chunks.append(value)
    sources = answer.get("lahteet", [])
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                snippet = source.get("snippet")
                if isinstance(snippet, str):
                    chunks.append(snippet)
    return " ".join(chunks).lower()


def score_answer(answer: dict[str, object], gold: GoldItem, latency_s: float) -> EvalResult:
    """Build an :class:`EvalResult` from an API response.

    Pure: doesn't talk to the network — the caller supplies both the
    answer and the measured latency.
    """
    abstained = bool(answer.get("abstained", False))
    max_score_raw = answer.get("max_source_score")
    max_score = float(max_score_raw) if isinstance(max_score_raw, (int, float)) else 0.0
    abstain_correct = abstained == gold.should_abstain

    extract_correct: bool | None = None
    if not gold.should_abstain and not abstained:
        body = extract_response_text(answer)
        extract_correct = all(expected.lower() in body for expected in gold.expected_contains)

    return EvalResult(
        question=gold.question,
        abstained=abstained,
        should_abstain=gold.should_abstain,
        abstain_correct=abstain_correct,
        extract_correct=extract_correct,
        max_source_score=max_score,
        latency_s=round(latency_s, 3),
    )


class _Clock(Protocol):
    def __call__(self) -> float: ...


class _QueryFn(Protocol):
    def __call__(self, *, question: str, mode: AnswerMode) -> dict[str, object]: ...


class _JudgeFn(Protocol):
    def __call__(
        self,
        *,
        question: str,
        rag_answer: dict[str, object],
        gold_answer: str | None,
        should_abstain: bool,
    ) -> tuple[str, float, str]:
        """Return ``(verdict, score, reason)`` for one item."""


def run_eval(
    gold: Iterable[GoldItem],
    query: _QueryFn,
    mode: AnswerMode,
    *,
    clock: _Clock = time.perf_counter,
    judge: _JudgeFn | None = None,
) -> EvalSummary:
    """Run the scorer loop. ``query`` is injected so tests can stub it.

    Network / transport errors are captured per-item and reported in
    :attr:`EvalSummary.errors`; they do not terminate the batch. When
    ``judge`` is supplied it is called once per successful answer and
    its verdict attached to the :class:`EvalResult`.
    """
    summary = EvalSummary(mode=mode)
    for item in gold:
        started = clock()
        try:
            answer = query(question=item.question, mode=mode)
        except httpx.HTTPError as err:
            summary.results.append(
                EvalResult(
                    question=item.question,
                    abstained=False,
                    should_abstain=item.should_abstain,
                    abstain_correct=False,
                    extract_correct=None,
                    max_source_score=0.0,
                    latency_s=round(clock() - started, 3),
                    error=str(err),
                )
            )
            continue
        scored = score_answer(answer, item, clock() - started)
        if judge is not None:
            gold_text = " ".join(item.expected_contains) or None
            verdict, score, reason = judge(
                question=item.question,
                rag_answer=answer,
                gold_answer=gold_text,
                should_abstain=item.should_abstain,
            )
            scored = replace(
                scored,
                judge_verdict=verdict,
                judge_score=score,
                judge_reason=reason,
            )
        summary.results.append(scored)
    return summary


def _http_query(api_base: str, timeout: float) -> _QueryFn:
    """Build a concrete query function hitting the live FastAPI backend."""

    def _do_query(*, question: str, mode: AnswerMode) -> dict[str, object]:
        resp = httpx.post(
            f"{api_base}/v1/query",
            json={"query": question, "mode": mode},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError(f"API palautti ei-dict payloadin: {type(payload).__name__}")
        return payload

    return _do_query


def _build_judge(anthropic_key: str | None) -> _JudgeFn:
    """Wrap :func:`lapua_rag.eval.judge.judge_answer` as a :class:`_JudgeFn`.

    Import is deferred so ``--help`` and non-judge runs don't require
    the ``lapua_rag`` package to be installed on sys.path.
    """
    from lapua_rag.eval.judge import judge_answer  # noqa: PLC0415

    def _judge(
        *,
        question: str,
        rag_answer: dict[str, object],
        gold_answer: str | None,
        should_abstain: bool,
    ) -> tuple[str, float, str]:
        verdict = judge_answer(
            question=question,
            rag_answer=rag_answer,
            gold_answer=gold_answer,
            should_abstain=should_abstain,
            api_key=anthropic_key,
        )
        return verdict.verdict, verdict.score, verdict.reason

    return _judge


def format_summary(summary: EvalSummary) -> str:
    """Render the summary as a human-readable multi-line string."""
    lines: list[str] = []
    lines.append("=" * 50)
    lines.append(f"TULOKSET (moodi: {summary.mode})")
    lines.append(f"  Yhteensä:          {len(summary.results)} kysymystä")
    lines.append(f"  Virheitä:          {summary.errors}")
    total_valid = len(summary.valid)
    correct = sum(1 for r in summary.valid if r.abstain_correct)
    lines.append(
        f"  Abstain-tarkkuus:  {correct}/{total_valid} "
        f"({100 * summary.abstain_accuracy:.1f} %)"
    )
    extract_acc = summary.extract_accuracy
    if extract_acc is not None:
        rated = [r for r in summary.valid if r.extract_correct is not None]
        rated_correct = sum(1 for r in rated if r.extract_correct)
        lines.append(
            f"  Extract-tarkkuus:  {rated_correct}/{len(rated)} ({100 * extract_acc:.1f} %)"
        )
    lines.append(f"  Keskim. latenssi:  {summary.average_latency_s:.2f} s")

    judge_acc = summary.judge_accuracy
    if judge_acc is not None:
        judged = summary.judged
        judge_correct = sum(
            1 for r in judged if r.judge_verdict in ("correct", "abstain_correct")
        )
        judge_partial = sum(1 for r in judged if r.judge_verdict == "partial")
        avg_score = (
            sum(r.judge_score or 0.0 for r in judged) / len(judged) if judged else 0.0
        )
        lines.append("")
        lines.append(f"  Judge-LLM ({_JUDGE_MODEL_NAME}):")
        lines.append(
            f"    Oikein:           {judge_correct}/{len(judged)} "
            f"({100 * judge_acc:.1f} %)"
        )
        lines.append(f"    Osittain:         {judge_partial}/{len(judged)}")
        lines.append(f"    Keskim. pisteet:  {avg_score:.2f}/1.00")

    lines.append("")
    lines.append("Hyväksymisrajat (v1.0):")
    lines.append(
        f"  Abstain-tarkkuus ≥ {int(100 * _ABSTAIN_TARGET)} %: "
        f"{'OK' if summary.abstain_accuracy >= _ABSTAIN_TARGET else 'EI OK'}"
    )
    if extract_acc is not None:
        lines.append(
            f"  Extract-tarkkuus ≥ {int(100 * _EXTRACT_TARGET)} %: "
            f"{'OK' if extract_acc >= _EXTRACT_TARGET else 'EI OK'}"
        )
    if judge_acc is not None:
        lines.append(
            f"  Judge-tarkkuus ≥ {int(100 * _JUDGE_TARGET)} %: "
            f"{'OK' if judge_acc >= _JUDGE_TARGET else 'EI OK'}"
        )
    return "\n".join(lines)


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API, help="FastAPI base URL")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Kultajoukko-tiedosto")
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=("extract", "retrieve", "synth"),
        help="Vastaustila",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP-timeout sekuntia")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Käytä LLM-as-judge arviointia (Anthropic API, ANTHROPIC_API_KEY vaaditaan).",
    )
    parser.add_argument(
        "--anthropic-key",
        default=None,
        help="Anthropic-API-avain (tai env ANTHROPIC_API_KEY).",
    )
    args = parser.parse_args(argv)
    mode: AnswerMode = args.mode

    try:
        gold = load_gold(args.gold)
    except FileNotFoundError as err:
        print(f"ERROR: {err}")
        print("Luo se ensin: tests/fixtures/gold/lapua_gold_v1.jsonl")
        return 2

    print(f"Kultajoukko: {len(gold)} kysymystä, moodi: {mode}")
    print(f"API: {args.api}\n")

    judge_fn = _build_judge(args.anthropic_key) if args.judge else None
    summary = run_eval(
        gold,
        _http_query(args.api, args.timeout),
        mode,
        judge=judge_fn,
    )
    for result in summary.results:
        marker = "✓" if result.abstain_correct else "✗"
        if result.error:
            print(f"  ERROR {result.question[:60]}: {result.error}")
            continue
        print(
            f"  {marker} abstain={result.abstained} "
            f"score={result.max_source_score:.3f} ({result.latency_s:.1f}s) "
            f"— {result.question[:60]}"
        )

    print()
    print(format_summary(summary))
    return 0 if summary.errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
