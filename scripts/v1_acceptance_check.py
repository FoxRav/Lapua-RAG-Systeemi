"""v1.0 acceptance-checklist — aja ennen julkaisua.

Skripti tarkistaa automaattisesti kaikki v1.0-hyväksymiskriteerit ja
tulostaa yhteenvedon + poistuu statuskoodilla 0 (GO) tai 1 (NO-GO).

Kriteerit kattavat kolme tasoa:
1. **Korpus / systeemi**: dokumenttimäärä, lähteiden läsnäolo.
2. **Rajapinta**: Prometheus ``/metrics``, auditloki, hash-stabiliteetti.
3. **Laatu**: closed-book guard, pytest, ruff, valinnainen LLM-as-judge.

Judge-tarkistus ohitetaan jos ``ANTHROPIC_API_KEY`` ei ole asetettuna
tai ``--skip-judge`` on annettu — tämän avulla skripti on CI-valmis
myös avaimettomissa ympäristöissä.

Käyttö::

    python scripts/v1_acceptance_check.py
    python scripts/v1_acceptance_check.py --api http://localhost:8080 --skip-judge
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

DEFAULT_API: Final[str] = "http://127.0.0.1:8080"
DEFAULT_GOLD: Final[Path] = Path("tests/fixtures/gold/lapua_gold_v1.jsonl")

# v1.0-kriteerit (README §5 + CURSOR_v0.9_OHJE.md §E).
_TARGET_CORPUS_DOCS: Final[int] = 200
_TARGET_JUDGE_ACCEPT: Final[float] = 0.80
_JUDGE_MAX_API_CALLS: Final[int] = 10
_OFF_TOPIC_QUESTIONS: Final[tuple[str, ...]] = (
    "Mikä on Helsingin pormestarin nimi?",
    "Paljonko Suomen valtio käytti puolustukseen 2020?",
)
_SMOKE_QUESTIONS: Final[tuple[str, ...]] = (
    "Kuka on Lapuan kaupunginjohtaja?",
    "Mitä päätettiin talousarviosta?",
)

_CRITERIA_LABELS: Final[dict[str, str]] = {
    "corpus_docs": f"Korpuksessa ≥{_TARGET_CORPUS_DOCS} dokumenttia",
    "corpus_sources": "Jokainen vastaus sisältää ≥1 lähdeviitteen",
    "closed_book_guard": "Off-topic-kysymykset johtavat pidättäytymiseen",
    "judge_accept": f"Judge-LLM hyväksyy ≥{int(100 * _TARGET_JUDGE_ACCEPT)} % vastauksista",
    "system_version_stable": "Systeemi-hash on stabiili",
    "prometheus_live": "Prometheus /metrics vastaa 200 OK",
    "audit_log_live": "Auditloki vastaa 200 OK",
    "tests_pass": "pytest tests/unit vihreinä",
    "ruff_clean": "ruff check puhdas",
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    @property
    def label(self) -> str:
        return _CRITERIA_LABELS.get(self.name, self.name)


def _mark(result: CheckResult) -> str:
    marker = "✓" if result.ok else "✗"
    tail = f" — {result.detail}" if result.detail else ""
    return f"  {marker} {result.label}{tail}"


def _check_corpus_size(client: httpx.Client) -> CheckResult:
    try:
        data = client.get("/v1/system/stats").json()
        count = int(data.get("document_count", 0))
        return CheckResult("corpus_docs", count >= _TARGET_CORPUS_DOCS, f"{count} dok")
    except Exception as exc:
        return CheckResult("corpus_docs", False, str(exc)[:60])


def _check_prometheus(client: httpx.Client) -> CheckResult:
    try:
        resp = client.get("/metrics")
        return CheckResult("prometheus_live", resp.status_code == 200)
    except Exception as exc:
        return CheckResult("prometheus_live", False, str(exc)[:60])


def _check_audit_log(client: httpx.Client) -> CheckResult:
    try:
        resp = client.get("/v1/audit", params={"limit": 1})
        return CheckResult("audit_log_live", resp.status_code == 200)
    except Exception as exc:
        return CheckResult("audit_log_live", False, str(exc)[:60])


def _check_system_version_stable(client: httpx.Client) -> CheckResult:
    try:
        first = client.get("/v1/system/version").json()
        second = client.get("/v1/system/version").json()
        hash_a = first.get("content_hash") or first.get("version_hash")
        hash_b = second.get("content_hash") or second.get("version_hash")
        ok = hash_a is not None and hash_a == hash_b
        return CheckResult("system_version_stable", ok, f"hash={str(hash_a)[:12]}")
    except Exception as exc:
        return CheckResult("system_version_stable", False, str(exc)[:60])


def _check_closed_book_guard(client: httpx.Client) -> CheckResult:
    abstained = 0
    for question in _OFF_TOPIC_QUESTIONS:
        try:
            resp = client.post(
                "/v1/query",
                json={"query": question, "mode": "extract"},
            )
            if resp.status_code == 200 and bool(resp.json().get("abstained")):
                abstained += 1
        except Exception:
            continue
    return CheckResult(
        "closed_book_guard",
        abstained == len(_OFF_TOPIC_QUESTIONS),
        f"{abstained}/{len(_OFF_TOPIC_QUESTIONS)} pidättäytyi",
    )


def _check_sources(client: httpx.Client) -> CheckResult:
    with_sources = 0
    for question in _SMOKE_QUESTIONS:
        try:
            resp = client.post(
                "/v1/query",
                json={"query": question, "mode": "extract"},
            )
            data = resp.json() if resp.status_code == 200 else {}
            if not data.get("abstained") and len(data.get("lahteet", []) or []) >= 1:
                with_sources += 1
        except Exception:
            continue
    return CheckResult(
        "corpus_sources",
        with_sources == len(_SMOKE_QUESTIONS),
        f"{with_sources}/{len(_SMOKE_QUESTIONS)} lähteellistä",
    )


def _run_subprocess(cmd: list[str]) -> tuple[int, str]:
    """Run ``cmd`` and return (returncode, last-line-of-stdout)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "").strip()
    tail = out.splitlines()[-1] if out else ""
    return proc.returncode, tail


def _check_pytest() -> CheckResult:
    code, summary = _run_subprocess(
        ["pytest", "tests/unit", "-q", "--no-cov", "--tb=no"],
    )
    return CheckResult("tests_pass", code == 0, summary[:80])


def _check_ruff() -> CheckResult:
    code, _ = _run_subprocess(["ruff", "check", "src", "tests", "scripts"])
    return CheckResult("ruff_clean", code == 0)


def _check_judge(
    client: httpx.Client,
    gold_path: Path,
    api_key: str,
) -> CheckResult:
    if not gold_path.is_file():
        return CheckResult(
            "judge_accept",
            False,
            f"kultajoukko puuttuu: {gold_path}",
        )
    from lapua_rag.eval.judge import judge_answer  # noqa: PLC0415

    items: list[dict[str, object]] = []
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        items.append(json.loads(stripped))
    batch = items[:_JUDGE_MAX_API_CALLS]
    if not batch:
        return CheckResult("judge_accept", False, "kultajoukko tyhjä")

    acceptable = 0
    for item in batch:
        question = str(item.get("question", ""))
        should_abstain = bool(item.get("should_abstain", False))
        gold_text = " ".join(str(s) for s in item.get("expected_contains", [])) or None
        try:
            resp = client.post(
                "/v1/query",
                json={"query": question, "mode": "extract"},
            )
            answer = resp.json() if resp.status_code == 200 else {}
        except Exception:
            answer = {}
        verdict = judge_answer(
            question=question,
            rag_answer=answer,
            gold_answer=gold_text,
            should_abstain=should_abstain,
            api_key=api_key,
        )
        if verdict.is_acceptable:
            acceptable += 1
    ratio = acceptable / len(batch)
    return CheckResult(
        "judge_accept",
        ratio >= _TARGET_JUDGE_ACCEPT,
        f"{acceptable}/{len(batch)} ({int(100 * ratio)} %)",
    )


def _render(results: Iterable[CheckResult]) -> str:
    return "\n".join(_mark(r) for r in results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API, help="FastAPI base URL")
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD,
        help="Kultajoukon polku judge-arviointiin.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Ohita LLM-as-judge -tarkistus (esim. jos ANTHROPIC_API_KEY puuttuu).",
    )
    parser.add_argument(
        "--anthropic-key",
        default=None,
        help="Anthropic-API-avain (tai env ANTHROPIC_API_KEY).",
    )
    args = parser.parse_args(argv)

    import os  # noqa: PLC0415 - env read deferred so --help is cheap

    api_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")

    print("Lapua-RAG v1.0 acceptance-check")
    print(f"API: {args.api}")
    print("=" * 60)

    client = httpx.Client(base_url=args.api, timeout=30.0)
    results: list[CheckResult] = [
        _check_corpus_size(client),
        _check_prometheus(client),
        _check_audit_log(client),
        _check_system_version_stable(client),
        _check_closed_book_guard(client),
        _check_sources(client),
        _check_pytest(),
        _check_ruff(),
    ]

    if args.skip_judge or not api_key:
        reason = "ohitettu (--skip-judge)" if args.skip_judge else "ohitettu (ei API-avainta)"
        print(_render(results))
        print(f"  - {_CRITERIA_LABELS['judge_accept']}: {reason}")
    else:
        results.append(_check_judge(client, args.gold, api_key))
        print(_render(results))

    print("=" * 60)
    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"Tulos: {passed}/{total} kriteeriä täyttyy")

    if passed == total:
        print("GO — valmis v1.0-julkaisuun")
        return 0
    failed = ", ".join(r.name for r in results if not r.ok)
    print(f"NO-GO — korjaa ensin: {failed}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
