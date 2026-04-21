"""Query rewriting for multi-query retrieval.

BGE/E5 encoders are sensitive to surface form: "Kuka on X?" and "X"
produce meaningfully different vectors. Issuing 2–3 phrased variants
of the same question and max-pooling their retrieval rankings recovers
a few extra percent of top-1 accuracy without retraining.

This module is intentionally heuristic and dependency-free:

* Finnish question-word stripping (``Kuka/Mitä/Missä/...``).
* Optional domain tag (``Lapuan kaupunki``) for off-topic guardrails.
* Extracted noun phrase from ``Kuka on X?`` patterns.

Pure functions, deterministic, unit-testable. A future revision can
plug in an LLM-based rewriter behind the same :func:`rewrite_query`
signature.
"""

from __future__ import annotations

import re
from typing import Final

from lapua_rag.observability import get_logger

_log = get_logger(__name__)

_MAX_REWRITES: Final[int] = 3
_QUESTION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"^(kuka|ket[äa]|mit[äa]|miss[äa]|milloin|miten|miksi|kuinka(?:\s+\w+)?)\s+",
    flags=re.IGNORECASE,
)
_WHO_IS_RE: Final[re.Pattern[str]] = re.compile(
    r"^kuka\s+on\s+(.+?)\s*\??$",
    flags=re.IGNORECASE,
)
_DOMAIN_TAG: Final[str] = "Lapuan kaupunki"


def _strip_question_word(query: str) -> str | None:
    """Return ``query`` with the leading Finnish question word dropped.

    Returns ``None`` if no stripping applied, so the caller can skip
    duplicates without a separate equality check.
    """
    stripped = _QUESTION_WORD_RE.sub("", query, count=1).strip().rstrip("?").strip()
    if stripped and stripped.lower() != query.strip().lower():
        return stripped
    return None


def _extract_who_is(query: str) -> str | None:
    match = _WHO_IS_RE.match(query.strip())
    if match is None:
        return None
    noun = match.group(1).strip().rstrip(".!?").strip()
    return noun or None


def _add_domain_tag(query: str) -> str | None:
    if _DOMAIN_TAG.lower() in query.lower() or "lapua" in query.lower():
        return None
    return f"{_DOMAIN_TAG} {query.strip()}"


def _dedup(items: list[str], *, max_items: int) -> list[str]:
    """Order-preserving dedup (case-insensitive) with a hard cap."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def heuristic_rewrites(query: str) -> list[str]:
    """Return the heuristic-only rewrites for ``query``.

    Always includes the original as the first element, capped at three
    variants total. Pure function — no network, no LLM.
    """
    candidates: list[str] = [query.strip()]
    # Prefer the "who is X?" noun phrase first — it's the highest-signal
    # rewrite for the common case and duplicates the question-word-strip
    # path anyway. Falling back to the generic stripper keeps the
    # rewriter useful for "Mitä päätettiin ...?" style queries.
    who = _extract_who_is(query)
    if who is not None:
        candidates.append(who)
    else:
        stripped = _strip_question_word(query)
        if stripped is not None:
            candidates.append(stripped)
    # Domain-tag last so we always reserve a slot for it within the
    # _MAX_REWRITES cap — the off-topic guardrail relies on at least one
    # variant carrying the "Lapuan kaupunki" tag.
    tagged = _add_domain_tag(query)
    if tagged is not None:
        candidates.append(tagged)
    return _dedup(candidates, max_items=_MAX_REWRITES)


def rewrite_query(query: str, *, use_llm: bool = False) -> list[str]:
    """Return 1–3 query reformulations (original always first).

    ``use_llm`` is reserved for a future LLM-backed rewriter and is
    currently a no-op — the heuristic layer is deliberately the only
    production path so retrieval quality never depends on vLLM being
    reachable.
    """
    del use_llm
    rewrites = heuristic_rewrites(query)
    _log.debug("query_rewrite", original=query, rewrites=rewrites)
    return rewrites
