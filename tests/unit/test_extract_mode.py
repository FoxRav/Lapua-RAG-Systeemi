"""Tests for AnswerService extract-mode (v0.6 bridge).

Extract-mode contract:

* The LoRA performs only a narrow quote-extraction task and returns
  a JSON string ``{"quote": "...", "chunk_index": N, "no_match": bool}``.
* If the LoRA returns ``no_match=true``, an empty quote, malformed JSON,
  or fails entirely, Python falls back to picking the highest
  token-overlap sentence from the top reranked chunk.
* Same ``no_context`` and ``below_threshold`` gates apply pre-LLM.
* The user always gets one coherent ``RagAnswer`` with ``abstained=False``
  in extract-mode (the gate handles the abstain cases).

v0.6.1: switched the LLM transport from ``generate_json`` (outlines
guided_json) to ``generate_text`` because the small RTX 4050 GPU could
not sustain outlines with 5 * 1500-char prompts. Tests now stub
``generate_text``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from lapua_rag.rag.answer import (
    AnswerService,
    _parse_extract_response,
    _python_fallback_across_chunks,
    _python_fallback_quote,
    _split_sentences,
)
from lapua_rag.retrieve.search import RetrievalResult


class _RecordingLlm:
    """LLM stub that returns a canned text reply (extract-mode contract).

    Pass a dict to have it serialised to JSON; pass a raw string to
    simulate non-JSON noise; pass ``None`` to make ``generate_text`` raise
    ``RuntimeError`` (used to test the LLM-failure fallback path).
    """

    def __init__(
        self,
        *,
        payload: dict[str, object] | None = None,
        raw_text: str | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._payload = payload
        self._raw_text = raw_text

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        self.calls.append((system, prompt))
        if self._raw_text is not None:
            return self._raw_text
        if self._payload is None:
            raise RuntimeError("LLM unavailable")
        return json.dumps(self._payload, ensure_ascii=False)

    def generate_json(  # pragma: no cover - extract no longer uses this path
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict[str, object],
    ) -> dict[str, object]:
        raise AssertionError(
            "extract-mode must call generate_text, not generate_json"
        )


@dataclass
class _FixedSearch:
    hits: list[RetrievalResult] = field(default_factory=list)

    def search(
        self,
        *,
        query: str,
        tenant: str,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        return self.hits


def _hits() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=5,
            section_id="§ 12",
            score=0.91,
            text=(
                "Tarkastuslautakunta on kutsunut kaupunginjohtaja Satu "
                "Kankareen kertomaan kaupungin ajankohtaisista asioista "
                "ja konsernivalvonnan tilanteesta. "
                "Kokouksessa käsiteltiin myös talousarvion toteumaa."
            ),
        ),
        RetrievalResult(
            chunk_id="c2",
            doc_id="d2",
            page_no=8,
            section_id=None,
            score=0.45,
            text="Toinen tukikatkelma joka ei sisällä vastausta.",
        ),
    ]


def test_extract_mode_uses_llm_quote_when_provided() -> None:
    """Happy path: LoRA returns a verbatim quote and the correct index."""
    llm = _RecordingLlm(
        payload={
            "quote": (
                "Tarkastuslautakunta on kutsunut kaupunginjohtaja Satu "
                "Kankareen kertomaan kaupungin ajankohtaisista asioista."
            ),
            "chunk_index": 1,
            "no_match": False,
        },
    )
    svc = AnswerService(
        search=_FixedSearch(hits=_hits()),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
        retrieve_snippet_chars=300,
    )

    ans = svc.answer(query="Kuka on kaupunginjohtaja?", tenant="lapua")

    assert len(llm.calls) == 1
    assert ans.abstained is False
    assert ans.abstain_reason is None
    assert "Satu Kankare" in ans.johtopaatos
    assert ans.perustelut.startswith("Lainattu suoraan lähteestä")
    # Quoted source comes first, supporting source second.
    assert len(ans.lahteet) == 2
    assert ans.lahteet[0].chunk_id == "c1"
    assert ans.lahteet[1].chunk_id == "c2"
    assert ans.max_source_score == pytest.approx(0.91)


def test_extract_mode_falls_back_to_python_when_llm_returns_no_match() -> None:
    """LoRA's abstain bias: model returns no_match=true even when the
    answer is in the chunk. Python fallback must pick a relevant sentence
    from the top chunk so the user still gets a coherent answer."""
    llm = _RecordingLlm(
        payload={"quote": "", "chunk_index": 1, "no_match": True},
    )
    svc = AnswerService(
        search=_FixedSearch(hits=_hits()),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
    )

    ans = svc.answer(query="Kuka on kaupunginjohtaja?", tenant="lapua")

    assert ans.abstained is False
    assert ans.perustelut.startswith("Python-fallback")
    # Python fallback must surface the sentence with the question keyword.
    assert "kaupunginjohtaja" in ans.johtopaatos.lower()
    assert "Satu Kankare" in ans.johtopaatos


def test_extract_mode_falls_back_to_python_when_llm_returns_empty_quote() -> None:
    """Empty string quote is treated as no-match (defensive: schema allows it)."""
    llm = _RecordingLlm(
        payload={"quote": "   ", "chunk_index": 1, "no_match": False},
    )
    svc = AnswerService(
        search=_FixedSearch(hits=_hits()),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
    )

    ans = svc.answer(query="Kuka on kaupunginjohtaja?", tenant="lapua")

    assert ans.abstained is False
    assert ans.perustelut.startswith("Python-fallback")


def test_extract_mode_falls_back_to_python_when_llm_raises() -> None:
    """Network/JSON errors must not break extract mode — log and fall back."""
    llm = _RecordingLlm(payload=None)  # raises RuntimeError on call
    svc = AnswerService(
        search=_FixedSearch(hits=_hits()),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
    )

    ans = svc.answer(query="Kuka on kaupunginjohtaja?", tenant="lapua")

    assert ans.abstained is False
    assert ans.perustelut.startswith("Python-fallback")
    assert "Satu Kankare" in ans.johtopaatos


def test_extract_mode_clamps_out_of_range_chunk_index() -> None:
    """Defensive: LLM occasionally returns chunk_index past the rendered N.
    The response must clamp to the available range, not crash."""
    llm = _RecordingLlm(
        payload={
            "quote": "Toinen tukikatkelma joka ei sisällä vastausta.",
            "chunk_index": 99,
            "no_match": False,
        },
    )
    svc = AnswerService(
        search=_FixedSearch(hits=_hits()),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
    )

    ans = svc.answer(query="?", tenant="lapua")

    assert ans.abstained is False
    # Clamps to last available chunk in the rendered context. The fixture
    # has 2 chunks, so the clamp limit is 2 → c2.
    assert ans.lahteet[0].chunk_id == "c2"


def test_extract_mode_respects_no_context_gate() -> None:
    llm = _RecordingLlm(payload={"quote": "x", "chunk_index": 1, "no_match": False})
    svc = AnswerService(
        search=_FixedSearch(hits=[]),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
    )

    ans = svc.answer(query="?", tenant="lapua")

    assert ans.abstained is True
    assert ans.abstain_reason == "no_context"
    assert llm.calls == []  # gate fires pre-LLM


def test_extract_mode_recovers_json_from_markdown_fence() -> None:
    """LoRA sometimes wraps JSON in ``` fences or prepends a stray word.
    The lenient parser must recover instead of forcing the Python fallback.
    """
    payload = {
        "quote": (
            "Tarkastuslautakunta on kutsunut kaupunginjohtaja Satu "
            "Kankareen kertomaan kaupungin ajankohtaisista asioista."
        ),
        "chunk_index": 1,
        "no_match": False,
    }
    raw = "Vastaus:\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    llm = _RecordingLlm(raw_text=raw)
    svc = AnswerService(
        search=_FixedSearch(hits=_hits()),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
    )

    ans = svc.answer(query="Kuka on kaupunginjohtaja?", tenant="lapua")

    assert ans.abstained is False
    assert ans.perustelut.startswith("Lainattu suoraan lähteestä")
    assert "Satu Kankare" in ans.johtopaatos


def test_extract_mode_falls_back_when_json_unparseable() -> None:
    """Garbage from the LoRA must trigger the deterministic Python path."""
    llm = _RecordingLlm(raw_text="ei tämä ole JSONia ollenkaan")
    svc = AnswerService(
        search=_FixedSearch(hits=_hits()),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
    )

    ans = svc.answer(query="Kuka on kaupunginjohtaja?", tenant="lapua")

    assert ans.abstained is False
    assert ans.perustelut.startswith("Python-fallback")
    assert "Satu Kankare" in ans.johtopaatos


def test_parse_extract_response_accepts_plain_json() -> None:
    parsed = _parse_extract_response('{"quote": "hi", "chunk_index": 2, "no_match": false}')
    assert parsed is not None
    assert parsed.quote == "hi"
    assert parsed.chunk_index == 2
    assert parsed.no_match is False


def test_parse_extract_response_returns_none_for_garbage() -> None:
    assert _parse_extract_response("") is None
    assert _parse_extract_response("not json") is None


def test_extract_mode_respects_below_threshold_gate() -> None:
    llm = _RecordingLlm(payload={"quote": "x", "chunk_index": 1, "no_match": False})
    hits = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=0,
            section_id=None,
            score=-1.5,
            text="irrelevant",
        )
    ]
    svc = AnswerService(
        search=_FixedSearch(hits=hits),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="extract",
    )

    ans = svc.answer(query="?", tenant="lapua")

    assert ans.abstained is True
    assert ans.abstain_reason == "below_threshold"
    assert ans.max_source_score == pytest.approx(-1.5)
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Python fallback / sentence-splitter unit tests.
# ---------------------------------------------------------------------------


def test_split_sentences_handles_markdown_and_html_noise() -> None:
    raw = (
        "## Päätös\n\n"
        "Kaupunginhallitus valitsi puheenjohtajaksi Maija Mallikkaan. "
        "<table><tr><td>kausi</td><td>2025</td></tr></table> "
        "Päätös tehtiin yksimielisesti."
    )
    sentences = _split_sentences(raw)
    assert any("Maija Mallikkaan" in s for s in sentences)
    assert any("yksimielisesti" in s for s in sentences)
    # Markdown header must be stripped, not retained as its own sentence.
    assert not any(s.startswith("##") for s in sentences)


def test_python_fallback_quote_picks_highest_overlap_sentence() -> None:
    chunk = (
        "Kokouksessa käsiteltiin talousarvion muutoksia. "
        "Kaupunginjohtaja Satu Kankare alusti asiaa. "
        "Päätös syntyi yksimielisesti."
    )
    quote = _python_fallback_quote("Kuka on kaupunginjohtaja?", chunk)
    assert "Satu Kankare" in quote


def test_python_fallback_quote_handles_empty_split() -> None:
    """Very short chunks where the splitter produces no sentences must
    still return something coherent (truncated raw text)."""
    quote = _python_fallback_quote("?", "Lyhyt.")
    assert quote == "Lyhyt."


def test_python_fallback_across_chunks_picks_best_overlap_chunk() -> None:
    """v0.6.2: when LoRA says no_match and the right answer is in chunk #2
    (not #1), the cross-chunk fallback must surface it instead of dumping
    chunk #1's noise. Replicates the Q2 pattern where reranker chose an
    attendance list and the actual answer was in a later chunk."""
    results = [
        RetrievalResult(
            chunk_id="c_attend",
            doc_id="d1",
            page_no=1,
            section_id=None,
            score=0.99,
            text=(
                "Saapuvillaolleet jäsenet Anneli Jäätteenmäki, puheenjohtaja "
                "Kai Pöntinen, 1. varapuheenjohtaja Marcus Toppari, 2. "
                "varapuheenjohtaja. Kokous oli päätösvaltainen."
            ),
        ),
        RetrievalResult(
            chunk_id="c_decision",
            doc_id="d2",
            page_no=5,
            section_id="§ 4",
            score=0.71,
            text=(
                "## 4 Kaupunginjohtajan kuuleminen LAPDno-2024-975. "
                "Tarkastuslautakunta on kutsunut kaupunginjohtaja Satu "
                "Kankareen kertomaan kaupungin ajankohtaisista asioista "
                "ja konsernivalvonnan tilanteesta."
            ),
        ),
    ]
    chunk_idx, quote = _python_fallback_across_chunks(
        "Kuka on Lapuan kaupunginjohtaja?", results
    )
    assert chunk_idx == 2
    assert "Satu Kankare" in quote


def test_python_fallback_across_chunks_keeps_top1_when_no_overlap_anywhere() -> None:
    """If no chunk has any term overlap, surface the top-1 chunk so the
    user still gets a cited reply (legacy behaviour for safety)."""
    results = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=0,
            section_id=None,
            score=0.5,
            text="Aivan eri aihepiirin teksti pitkänä virkkeenä jossa kahdeksankymmentä merkkiä.",
        ),
        RetrievalResult(
            chunk_id="c2",
            doc_id="d2",
            page_no=0,
            section_id=None,
            score=0.4,
            text="Toinen myös aivan eri aihepiirin pitkä virke vailla osumia kysymykseen.",
        ),
    ]
    chunk_idx, quote = _python_fallback_across_chunks(
        "Kuka on kaupunginjohtaja?", results
    )
    assert chunk_idx == 1
    assert quote
