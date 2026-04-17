from __future__ import annotations

from lapua_rag.index.hybrid import rrf_fuse


def test_rrf_fuse_prefers_chunks_in_both_rankings() -> None:
    dense = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
    sparse = [("b", 2.0), ("d", 1.5), ("a", 1.0)]
    fused = rrf_fuse(rankings=[dense, sparse])
    assert fused[0][0] in {"a", "b"}
    assert dict(fused)["b"] > dict(fused)["c"]


def test_rrf_fuse_empty() -> None:
    assert rrf_fuse(rankings=[[], []]) == []
