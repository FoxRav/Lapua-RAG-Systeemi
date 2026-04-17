"""BGE reranker v2 m3 wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field

from lapua_rag.observability import get_logger

_log = get_logger(__name__)


@dataclass(slots=True)
class Reranker:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    _model: object | None = field(default=None, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        _log.info("rerank.model_load", model=self.model_name, device=self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        *,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Rerank ``(chunk_id, text)`` pairs and return ``(chunk_id, score)``.

        Pure wrapper over the cross-encoder; no I/O other than the model call.
        """
        if not candidates:
            return []
        self._ensure_loaded()
        assert self._model is not None
        scores = self._model.predict(  # type: ignore[attr-defined]
            [(query, text) for _, text in candidates],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        paired = zip(candidates, scores, strict=True)
        ranked = sorted(
            ((chunk_id, float(score)) for (chunk_id, _), score in paired),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
