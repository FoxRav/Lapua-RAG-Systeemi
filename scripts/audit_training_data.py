"""Audit the abstain/extract balance of a LoRA fine-tuning dataset.

Reads JSONL/JSON training files (ChatML-style ``messages`` format) and
reports per-file and aggregate counts of three labels:

* ``abstain`` — assistant explicitly refuses / JSON ``abstained=true``
* ``extract`` — assistant produces a substantive answer or quote
* ``unknown`` — could not classify (assistant message missing or opaque)
* ``parse_error`` — JSONL line failed to parse

Usage::

    python scripts/audit_training_data.py
    python scripts/audit_training_data.py --path data/training

The pure-function classifier ``classify_example`` and file-level
``audit_file`` are exported for reuse in unit tests; ``main`` is only
the CLI driver.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Canonical labels emitted by the classifier. A Counter keyed by these
# strings is returned from every audit step.
Label = str
LABEL_ABSTAIN: Final[Label] = "abstain"
LABEL_EXTRACT: Final[Label] = "extract"
LABEL_UNKNOWN: Final[Label] = "unknown"
LABEL_PARSE_ERROR: Final[Label] = "parse_error"

# Default search locations when --path is not given. Non-existing paths
# are simply skipped; no error if the repo hasn't received training data
# yet.
DEFAULT_SEARCH_PATHS: Final[tuple[Path, ...]] = (
    Path("data/training"),
    Path("data/finetune"),
    Path("tmp/training_data"),
    Path("../training_data"),
)

# Threshold below which extract example body is considered too short to
# be a substantive answer (but not structured abstain either). Keeps
# one-word assistant messages out of the extract bucket.
_MIN_EXTRACT_CHARS: Final[int] = 50

# Heuristic abstain phrases — case-insensitive substring match on
# assistant content when the JSON path doesn't apply.
_ABSTAIN_PHRASES: Final[tuple[str, ...]] = (
    "en löydä",
    "en löyda",
    "abstain",
    "no_match",
)


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Aggregated audit result across one or many files."""

    counts: Counter[Label]
    files_scanned: int

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def abstain_pct(self) -> float:
        """Share of classified examples labelled ``abstain`` (0..100)."""
        classified = self.counts[LABEL_ABSTAIN] + self.counts[LABEL_EXTRACT]
        if classified == 0:
            return 0.0
        return 100.0 * self.counts[LABEL_ABSTAIN] / classified

    @property
    def verdict(self) -> str:
        """Human-readable recommendation for the retrain decision."""
        total = self.total
        if total == 0:
            return "Ei dataa auditoitavaksi."
        pct = self.abstain_pct
        if pct > 60:
            return (
                f"VAROITUS: abstain-osuus {pct:.1f} % — retrain tarvitaan "
                "(tavoite 50/50)"
            )
        if pct < 40:
            return (
                f"VAROITUS: extract-osuus liian dominoiva "
                f"({100 - pct:.1f} %) — lisää abstain-esimerkkejä"
            )
        return f"OK: abstain/extract-suhde tasapainossa ({pct:.1f} % / {100 - pct:.1f} %)"


def _assistant_content(example: object) -> str | None:
    """Return the last assistant message content, or None if missing."""
    if not isinstance(example, dict):
        return None
    messages = example.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            content = msg.get("content")
            return content if isinstance(content, str) else None
    return None


def _classify_json_content(content: str) -> Label | None:
    """Classify a JSON-encoded assistant message.

    Returns None if the string isn't valid JSON or lacks the well-known
    schema fields, leaving the heuristic path to take over.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("abstained") is True or parsed.get("no_match") is True:
        return LABEL_ABSTAIN
    quote = parsed.get("quote")
    if isinstance(quote, str) and quote.strip():
        return LABEL_EXTRACT
    johtopaatos = parsed.get("johtopaatos")
    if isinstance(johtopaatos, str) and johtopaatos.strip():
        return LABEL_EXTRACT
    return None


def classify_example(example: object) -> Label:
    """Classify a single training example as abstain/extract/unknown.

    Pure function — no I/O, no global state. Accepts the decoded JSON
    object (dict) directly; see :func:`audit_file` for the file-level
    wrapper.
    """
    content = _assistant_content(example)
    if content is None:
        return LABEL_UNKNOWN
    json_label = _classify_json_content(content)
    if json_label is not None:
        return json_label
    lowered = content.lower()
    if any(phrase in lowered for phrase in _ABSTAIN_PHRASES):
        return LABEL_ABSTAIN
    if len(content.strip()) >= _MIN_EXTRACT_CHARS:
        return LABEL_EXTRACT
    return LABEL_UNKNOWN


def audit_file(path: Path) -> Counter[Label]:
    """Count labels in a JSONL file. Malformed lines are counted under
    ``parse_error`` rather than raising — auditing a partially corrupt
    export should still produce an overall picture.
    """
    counts: Counter[Label] = Counter()
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                example = json.loads(stripped)
            except json.JSONDecodeError:
                counts[LABEL_PARSE_ERROR] += 1
                continue
            counts[classify_example(example)] += 1
    return counts


def audit_paths(paths: Iterable[Path]) -> AuditReport:
    """Walk each given path for ``*.jsonl`` / ``*.json`` and aggregate."""
    total: Counter[Label] = Counter()
    scanned = 0
    for base in paths:
        if not base.exists():
            continue
        if base.is_file():
            files: list[Path] = [base]
        else:
            files = sorted({*base.rglob("*.jsonl"), *base.rglob("*.json")})
        for file_path in files:
            total.update(audit_file(file_path))
            scanned += 1
    return AuditReport(counts=total, files_scanned=scanned)


def _format_counts(counts: Counter[Label], total: int) -> str:
    lines: list[str] = []
    for label in sorted(counts):
        n = counts[label]
        pct = 100.0 * n / total if total else 0.0
        lines.append(f"  {label:12s}: {n:5d}  ({pct:.1f} %)")
    return "\n".join(lines)


def run_cli(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint; returns a process exit code.

    Exposed as a function (not inlined in ``__main__``) so tests can
    invoke it with a captured ``capsys`` and a temp-dir ``--path``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=None, help="Polku dataan tai tiedostoon")
    args = parser.parse_args(argv)

    search_paths: tuple[Path, ...] = (args.path,) if args.path else DEFAULT_SEARCH_PATHS

    report = audit_paths(search_paths)
    if report.files_scanned == 0:
        print("ERROR: Koulutustiedostoja ei löydy. Anna --path tai luo data/training/.")
        print("Etsityt sijainnit:", [str(p) for p in search_paths])
        return 2

    print(f"Auditoitu {report.files_scanned} tiedosto(a)\n")
    print("=" * 40)
    print(f"YHTEENSÄ: {report.total} esimerkkiä")
    print(_format_counts(report.counts, report.total))
    print()
    print(report.verdict)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
