"""Build a balanced ``lapua-llm-v3`` fine-tuning dataset from the live
Qdrant index.

Goal: produce a 50/50 mix of ``extract`` and ``abstain`` ChatML examples
using chunks already indexed from the Lapua corpus. This avoids the
chicken-and-egg problem of training a better LoRA when we only have
abstain-heavy v2 data.

Strategy:

* **Extract examples** — walk the corpus, pick chunks whose
  ``section_title`` / body carries a decision verb (valittiin,
  hyväksyttiin, päätettiin), synthesise a plausible question from the
  section, and emit ``(system, user, assistant)`` where the assistant
  answer is a JSON ``{quote, chunk_index, no_match=false}``.
* **Abstain examples** — pair random chunks with off-topic questions
  unrelated to Lapua / municipal business, and emit assistant answers
  ``{quote: "", chunk_index: 0, no_match: true}``.

The Qdrant scan is isolated in :func:`_iter_chunks`; the generator
primitives (:func:`extract_question_from_chunk`, :func:`make_extract_example`,
:func:`make_abstain_example`, :func:`build_dataset`) are pure and fully
unit-testable.

Usage::

    python scripts/build_v3_dataset.py --output data/training/v3_dataset.jsonl
    python scripts/build_v3_dataset.py --max-extract 300 --max-abstain 300
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

SYSTEM_PROMPT: Final[str] = (
    "Olet Lapuan kaupungin asiakirja-assistentti. "
    "Tehtäväsi on lainata annetusta kontekstista täsmälliset virkkeet "
    "jotka vastaavat kysymykseen. Palauta AINOASTAAN JSON-objekti ilman selityksiä."
)

_USER_INSTRUCTION_TAIL: Final[str] = (
    "Palauta JSON jossa kentät: quote (string), chunk_index (integer), no_match (boolean)."
)

# Off-topic questions that the Lapua corpus cannot answer. Used as the
# user turn for abstain-style training pairs.
ABSTAIN_QUESTIONS: Final[tuple[str, ...]] = (
    "Mikä on Helsingin pormestarin nimi?",
    "Kuinka paljon Suomen valtio käytti puolustukseen vuonna 2020?",
    "Milloin Suomi liittyi Euroopan unioniin?",
    "Kuka voitti viime vuoden SM-liigan?",
    "Mikä on Suomen väkiluku tällä hetkellä?",
    "Milloin seuraavat presidentinvaalit järjestetään?",
    "Kuinka monta kuntaa Suomessa on?",
    "Mikä on Tampereen kaupunginjohtajan nimi?",
    "Paljonko Suomen bruttokansantuote on?",
    "Milloin Suomi itsenäistyi?",
)

# Per-chunk context budget in characters. Keep under 1500 so the ChatML
# turns stay within the 2-3k-token training window comfortably.
_CONTEXT_BUDGET: Final[int] = 1500
_MIN_QUOTE_CHARS: Final[int] = 40


@dataclass(frozen=True, slots=True)
class Chunk:
    """Minimal chunk representation the dataset builder needs.

    Deliberately smaller than ``ChunkRow`` — the builder never writes
    back to the DB, it only needs the payload fields carried in Qdrant.
    """

    chunk_id: str
    text: str
    doc_type: str
    section: str
    payload: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> Chunk:
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            text=str(payload.get("text", "")),
            doc_type=str(payload.get("doc_type", "")),
            section=str(payload.get("section_title") or payload.get("section") or ""),
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class ExampleCounts:
    extract: int
    abstain: int

    @property
    def total(self) -> int:
        return self.extract + self.abstain

    @property
    def abstain_pct(self) -> float:
        return 100.0 * self.abstain / self.total if self.total else 0.0


def _clean_section(section: str) -> str:
    """Strip markdown header tokens from a section title."""
    return section.replace("## ", "").replace("# ", "").strip()


def _choose_quote(text: str) -> str | None:
    """Pick the longest non-trivial line from the top of the chunk.

    Returns None if no line meets the minimum quote length. The
    heuristic favours lines from the first five — most pöytäkirja
    templates put the decision verb near the top of the section.
    """
    candidates = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and len(line.strip()) >= 30
    ]
    if not candidates:
        return None
    top = candidates[:5]
    quote = max(top, key=len)
    if len(quote) < _MIN_QUOTE_CHARS:
        return None
    return quote


def _compose_question(section: str, quote: str) -> str:
    """Build a plausible question using verb cues from the quote."""
    cleaned = _clean_section(section) or "päätösasiasta"
    lowered = quote.lower()
    if "valittiin" in lowered or "valittu" in lowered:
        return f"Kuka valittiin asiassa {cleaned}?"
    if "hyväksyttiin" in lowered or "hyväksytty" in lowered:
        return f"Mitä hyväksyttiin kohdassa {cleaned}?"
    if "päätti" in lowered or "päätettiin" in lowered:
        return f"Mitä päätettiin asiassa {cleaned}?"
    return f"Mitä asiakirja kertoo kohdasta {cleaned}?"


def extract_question_from_chunk(chunk: Chunk) -> tuple[str, str] | None:
    """Derive a (question, quote) pair from a chunk, or None if unsuitable.

    Pure function — consulted purely against the text / section data;
    no I/O, no randomness. Only ``Päätös``-style sections yield a pair,
    because those are the sections with a decidable verb.
    """
    section_lower = chunk.section.lower()
    text_head = chunk.text[:200].lower()
    if "päätös" not in section_lower and "päätös" not in text_head:
        return None
    quote = _choose_quote(chunk.text)
    if quote is None:
        return None
    question = _compose_question(chunk.section, quote)
    return question, quote


def _chatml(user: str, assistant_payload: dict[str, object]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload, ensure_ascii=False),
            },
        ]
    }


def make_extract_example(chunk: Chunk, question: str, quote: str) -> dict[str, object]:
    """Build an extract (positive) ChatML example."""
    context = chunk.text[:_CONTEXT_BUDGET]
    user_turn = (
        f"Kysymys: {question}\n\n"
        f"Konteksti:\n{context}\n\n"
        f"{_USER_INSTRUCTION_TAIL}"
    )
    return _chatml(user_turn, {"quote": quote, "chunk_index": 0, "no_match": False})


def make_abstain_example(chunk: Chunk, question: str) -> dict[str, object]:
    """Build an abstain (negative) ChatML example."""
    context = chunk.text[:_CONTEXT_BUDGET]
    user_turn = (
        f"Kysymys: {question}\n\n"
        f"Konteksti:\n{context}\n\n"
        f"{_USER_INSTRUCTION_TAIL}"
    )
    return _chatml(user_turn, {"quote": "", "chunk_index": 0, "no_match": True})


def build_dataset(
    chunks: Sequence[Chunk],
    *,
    max_extract: int,
    max_abstain: int,
    abstain_questions: Sequence[str] = ABSTAIN_QUESTIONS,
    rng: random.Random | None = None,
) -> tuple[list[dict[str, object]], ExampleCounts]:
    """Build a shuffled dataset from a chunk sequence.

    Deterministic when ``rng`` is supplied — tests pin a seed so the
    output order is reproducible. When called from the CLI with default
    ``rng=None``, a fresh ``Random(42)`` instance is used.
    """
    rng = rng if rng is not None else random.Random(42)

    extract_pool: list[dict[str, object]] = []
    shuffled_for_extract = list(chunks)
    rng.shuffle(shuffled_for_extract)
    for chunk in shuffled_for_extract:
        if len(extract_pool) >= max_extract:
            break
        result = extract_question_from_chunk(chunk)
        if result is None:
            continue
        question, quote = result
        extract_pool.append(make_extract_example(chunk, question, quote))

    abstain_pool: list[dict[str, object]] = []
    if abstain_questions:
        shuffled_for_abstain = list(chunks)
        rng.shuffle(shuffled_for_abstain)
        for idx, chunk in enumerate(shuffled_for_abstain):
            if len(abstain_pool) >= max_abstain:
                break
            question = abstain_questions[idx % len(abstain_questions)]
            abstain_pool.append(make_abstain_example(chunk, question))

    combined = extract_pool + abstain_pool
    rng.shuffle(combined)
    counts = ExampleCounts(extract=len(extract_pool), abstain=len(abstain_pool))
    return combined, counts


def write_jsonl(examples: Iterable[dict[str, object]], output: Path) -> int:
    """Write examples as JSONL; returns the number of lines written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False))
            fh.write("\n")
            written += 1
    return written


class _ChunkSource(Protocol):
    def __call__(self) -> Iterator[Chunk]: ...


def _qdrant_chunk_source(url: str, collection: str, batch_size: int = 100) -> _ChunkSource:
    """Closure that yields Chunks from Qdrant.

    Lazy-imports ``qdrant_client`` so unit tests that monkeypatch the
    chunk source never touch the real SDK.
    """

    def _source() -> Iterator[Chunk]:
        from qdrant_client import QdrantClient  # noqa: PLC0415 — lazy heavy dep

        client = QdrantClient(url=url)
        offset: object | None = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                if isinstance(payload, dict):
                    yield Chunk.from_payload(payload)
            if next_offset is None:
                break
            offset = next_offset

    return _source


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/v3_dataset.jsonl"),
        help="Ulostulotiedosto",
    )
    parser.add_argument("--max-extract", type=int, default=300)
    parser.add_argument("--max-abstain", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Ohittaa Settings.qdrant_url:n (debug)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant-kokoelma (oletus Settings.qdrant_collection)",
    )
    args = parser.parse_args(argv)

    # Lazy-import config so unit tests can bypass the whole CLI path
    # without pulling pydantic-settings + its .env lookup.
    from lapua_rag.config import get_settings  # noqa: PLC0415 — keep CLI-only

    settings = get_settings()
    url = args.qdrant_url or settings.qdrant_url
    collection = args.collection or settings.qdrant_collection

    print(f"Haetaan chunkit kokoelmasta: {collection} ({url})")
    source = _qdrant_chunk_source(url=url, collection=collection)
    chunks = list(source())
    print(f"Chunkkeja yhteensä: {len(chunks)}")
    if not chunks:
        print("VAROITUS: Qdrant-kokoelma on tyhjä — ei datasetia rakennettavaksi.")
        return 1

    rng = random.Random(args.seed)
    examples, counts = build_dataset(
        chunks, max_extract=args.max_extract, max_abstain=args.max_abstain, rng=rng
    )
    written = write_jsonl(examples, args.output)

    print(f"\nValmis: {written} esimerkkiä → {args.output}")
    print(f"  Extract: {counts.extract}")
    print(f"  Abstain: {counts.abstain}")
    print(f"  Abstain-osuus: {counts.abstain_pct:.1f} %")
    print("\nSeuraava askel: tarkista 10 satunnaista esimerkkiä manuaalisesti.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
