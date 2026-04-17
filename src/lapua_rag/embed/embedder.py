"""sentence-transformers wrapper with E5/BGE prefixing conventions."""

from __future__ import annotations

from dataclasses import dataclass, field

from lapua_rag.observability import get_logger

_log = get_logger(__name__)


@dataclass(slots=True)
class Embedder:
    """Dense embedder wrapping sentence-transformers.

    Handles E5-family prefixes (``passage: `` / ``query: ``) automatically.
    """

    model_name: str
    device: str = "cpu"
    batch_size: int = 16
    _model: object | None = field(default=None, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        _log.info("embed.model_load", model=self.model_name, device=self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def _is_e5(self) -> bool:
        return "e5" in self.model_name.lower()

    def _prep(self, texts: list[str], *, role: str) -> list[str]:
        if not self._is_e5:
            return texts
        prefix = "query: " if role == "query" else "passage: "
        return [prefix + t for t in texts]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(  # type: ignore[attr-defined]
            self._prep(texts, role="passage"),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        self._ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(  # type: ignore[attr-defined]
            self._prep([text], role="query"),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors[0].tolist()

    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
