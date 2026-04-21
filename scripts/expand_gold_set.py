"""Expand the gold evaluation set semi-automatically from the live corpus.

Generates extract-type questions from decision-bearing chunks in Qdrant
and pads the abstain set with canned off-topic questions so the
final file hits the target size (default 30).

Usage::

    python scripts/expand_gold_set.py --target 30

The Qdrant client is imported lazily so unit tests can exercise the
pure helpers without a running Qdrant service.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_DECISION_KEYWORDS: tuple[str, ...] = (
    "valittiin",
    "hyväksyttiin",
    "päätettiin",
    "nimitettiin",
)

# Canned abstain seeds: off-topic questions a Lapua-only corpus cannot
# answer. Gold-set invariant: should_abstain=True.
_OFFTOPIC_QUESTIONS: tuple[str, ...] = (
    "Mikä on Tampereen kaupunginjohtajan nimi?",
    "Milloin Suomi liittyi EU:hun?",
    "Kuka voitti viime kauden Liigan?",
    "Mikä on Suomen väkiluku?",
    "Paljonko valtion budjetti on?",
)


@dataclass(slots=True, frozen=True)
class GoldItem:
    """One gold-set row, serialisable to the project's JSONL format."""

    question: str
    expected_contains: list[str]
    doc_type: str
    should_abstain: bool
    source_doc_id: str | None = None
    generated: bool = True

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "question": self.question,
            "expected_contains": self.expected_contains,
            "doc_type": self.doc_type,
            "should_abstain": self.should_abstain,
            "generated": self.generated,
        }
        if self.source_doc_id is not None:
            payload["source_doc_id"] = self.source_doc_id
        return payload


class _ChunkSource(Protocol):
    """Minimal interface the generator needs — lets tests inject lists."""

    def fetch(self) -> list[dict[str, object]]: ...


def extract_gold_from_chunk(chunk: dict[str, object]) -> GoldItem | None:
    """Return a GoldItem built from a decision-bearing chunk, or None.

    Pure: no I/O, no randomness. We require a decision keyword in the
    body so we don't generate questions against attendance lists and
    boilerplate.
    """
    text = str(chunk.get("text", "") or "")
    if not text:
        return None
    doc_type = str(chunk.get("doc_type", "") or "")
    section = str(chunk.get("section", "") or "").replace("## ", "").strip()
    section_id = str(chunk.get("section_id", "") or "")
    doc_id = str(chunk.get("doc_id", "unknown") or "unknown")

    keyword = next((kw for kw in _DECISION_KEYWORDS if kw in text.lower()), None)
    if keyword is None:
        return None

    # Prefer substantive sentences (>= 40 chars) — lists of names or
    # single-line boilerplate make poor questions.
    candidates = [line.strip() for line in text.split("\n") if len(line.strip()) >= 40]
    if not candidates:
        return None
    answer_line = max(candidates[:5], key=len)

    # Disambiguator keeps identical-section chunks (very common in
    # meeting minutes where "## Päätös" repeats on every §) from
    # collapsing to the same question after dedup.
    disambiguator_parts: list[str] = []
    if section_id:
        disambiguator_parts.append(section_id)
    disambiguator_parts.append(doc_id[:8])
    disambiguator = " ".join(disambiguator_parts)

    topic = section if section else doc_type or "päätöksestä"
    if keyword in {"valittiin", "nimitettiin"}:
        question = f"Kuka valittiin {topic} ({disambiguator})?"
    elif keyword == "hyväksyttiin":
        question = f"Mitä hyväksyttiin {topic} ({disambiguator})?"
    else:
        question = f"Mitä päätettiin {topic} ({disambiguator})?"

    return GoldItem(
        question=question,
        expected_contains=[answer_line[:60]],
        doc_type=doc_type or "any",
        should_abstain=False,
        source_doc_id=doc_id,
        generated=True,
    )


def build_offtopic_items() -> list[GoldItem]:
    """Return canned abstain questions (deterministic order)."""
    return [
        GoldItem(
            question=q,
            expected_contains=[],
            doc_type="any",
            should_abstain=True,
            source_doc_id=None,
            generated=True,
        )
        for q in _OFFTOPIC_QUESTIONS
    ]


def load_existing(path: Path) -> list[dict[str, object]]:
    """Read an existing JSONL gold file; empty list if absent."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def assemble(
    *,
    existing: list[dict[str, object]],
    chunks: list[dict[str, object]],
    target: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    """Return `target`-or-more gold items, deduplicated by question text.

    Strategy: keep every existing row, append off-topic seeds up to
    ``target`` budget, then fill remaining slots with generated extract
    rows. When the chunk pool is too small we return whatever we could
    generate — the caller prints the final count.
    """
    seen_questions: set[str] = {str(row.get("question", "")) for row in existing}
    result: list[dict[str, object]] = list(existing)

    def _push(item: GoldItem) -> bool:
        if item.question in seen_questions:
            return False
        seen_questions.add(item.question)
        result.append(item.to_dict())
        return True

    # Off-topic seeds come first — they're deterministic and small.
    for item in build_offtopic_items():
        if len(result) >= target:
            break
        _push(item)

    shuffled = list(chunks)
    rng.shuffle(shuffled)
    for chunk in shuffled:
        if len(result) >= target:
            break
        extract_item = extract_gold_from_chunk(chunk)
        if extract_item is None:
            continue
        _push(extract_item)

    return result


def _default_source(tenant: str, url: str, collection: str) -> _ChunkSource:
    """Return a Qdrant-backed chunk source. Lazy-imports qdrant_client."""

    class _QdrantSource:
        def fetch(self) -> list[dict[str, object]]:  # pragma: no cover - network
            # Import inside the method so module-level imports stay lean
            # and unit tests can monkey-patch this factory.
            from qdrant_client import QdrantClient  # noqa: PLC0415

            client = QdrantClient(url=url)
            rows, _ = client.scroll(
                collection_name=collection,
                scroll_filter=None,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            # Only keep payloads for the requested tenant — the live
            # collection may be multi-tenant even when `tenant` is a
            # single-value config.
            return [
                dict(r.payload)
                for r in rows
                if isinstance(r.payload, dict)
                and (r.payload.get("tenant") in (tenant, None))
            ]

    return _QdrantSource()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/gold/lapua_gold_v1.jsonl"),
    )
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    ns = parser.parse_args(argv)

    rng = random.Random(ns.seed)

    # Lazy settings import — avoids dragging pydantic-settings into the
    # test path when the CLI isn't invoked.
    from lapua_rag.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    source = _default_source(
        tenant=settings.tenant,
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
    )
    chunks = source.fetch()

    existing = load_existing(ns.output)
    items = assemble(
        existing=existing,
        chunks=chunks,
        target=ns.target,
        rng=rng,
    )

    ns.output.parent.mkdir(parents=True, exist_ok=True)
    with ns.output.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    abstain_n = sum(1 for i in items if i.get("should_abstain"))
    print(f"Gold set: {len(items)} questions → {ns.output}")
    print(f"  extract: {len(items) - abstain_n}")
    print(f"  abstain: {abstain_n}")


if __name__ == "__main__":  # pragma: no cover
    main()
