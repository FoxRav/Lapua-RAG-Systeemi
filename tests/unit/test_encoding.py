from __future__ import annotations

from lapua_rag.postprocess.encoding import fix_finnish_mojibake, looks_like_mojibake


def test_looks_like_mojibake_detects_replacement() -> None:
    assert looks_like_mojibake("k\ufffdsittely") is True


def test_looks_like_mojibake_detects_double_encoded() -> None:
    assert looks_like_mojibake("pÃ¤Ã¤tÃ¶s") is True


def test_fix_finnish_mojibake_roundtrip() -> None:
    original = "päätös"
    corrupted = original.encode("utf-8").decode("cp1252")
    assert fix_finnish_mojibake(corrupted) == original


def test_fix_finnish_mojibake_is_noop_for_clean_text() -> None:
    assert fix_finnish_mojibake("pöytäkirja § 42") == "pöytäkirja § 42"
