"""Systeemi – the curated data corpus.

*Systeemi* is the **only** knowledge source the SLM (Qwen2.5-1.5B + LoRA
`lapua-llm-v2`) is allowed to draw from. Anything outside Systeemi must
result in explicit abstention, never in hallucinated answers from the
model's pretraining memory.

This subpackage provides the observable surface of Systeemi:

* :mod:`.stats`      – document / chunk / token counts, per-tenant breakdown
* :mod:`.versioning` – per-doc manifest + global Systeemi version hash
* :mod:`.coverage`   – what is in Systeemi today (per doc_type / pvm-range)
"""

from __future__ import annotations

from lapua_rag.systeemi.coverage import CoverageReport, compute_coverage
from lapua_rag.systeemi.stats import SystemStats, gather_stats
from lapua_rag.systeemi.versioning import SystemVersion, compute_system_version

__all__ = [
    "CoverageReport",
    "SystemStats",
    "SystemVersion",
    "compute_coverage",
    "compute_system_version",
    "gather_stats",
]
