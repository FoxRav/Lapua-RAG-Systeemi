"""Central configuration using pydantic-settings.

All environment variables are prefixed with ``LAPUA_`` and can be overridden via
``.env``, shell env, or constructor arguments during tests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Lapua-RAG service."""

    model_config = SettingsConfigDict(
        env_prefix="LAPUA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Tenant ------------------------------------------------------------
    tenant: str = "lapua"

    # Storage -----------------------------------------------------------
    storage_root: Path = Field(default=Path("data/storage"))
    inbox_dir: Path = Field(default=Path("data/inbox"))
    index_dir: Path = Field(default=Path("data/index"))

    # Metadata DB -------------------------------------------------------
    database_url: str = "sqlite:///data/index/metadata.sqlite"

    # Vector DB ---------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "lapua_chunks"

    # Embeddings --------------------------------------------------------
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = 16

    # Reranker ----------------------------------------------------------
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: Literal["cpu", "cuda", "mps"] = "cpu"

    # LLM ---------------------------------------------------------------
    llm_base: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_lora: str = "CCG-FAKTUM/lapua-llm-v2"
    llm_device: Literal["cpu", "cuda", "mps"] = "cpu"
    # bf16 default: halves model weights and attention workspace vs. fp32.
    # Qwen2.5-1.5B in fp32 OOMs on 16GB RAM laptops during CPU self-attention;
    # bf16 makes single-query inference fit without GPU.
    llm_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    llm_max_new_tokens: int = 256
    llm_vllm_url: str | None = None
    # vLLM model identifier to pass in the OpenAI payload. When a LoRA adapter is
    # served alongside the base (e.g. `--lora-modules lapua-v2=<path>`), set this
    # to the LoRA module name so the adapter is actually applied; otherwise leave
    # as None to hit the base model.
    llm_vllm_model: str | None = "lapua-v2"

    # AnswerService -----------------------------------------------------
    # "synth"   = retrieve → rerank → vLLM extracts a JSON answer.
    # "retrieve"= retrieve → rerank → return top-N cited chunks verbatim,
    #             skipping the LLM entirely. Use this when the synthesis
    #             model is unreliable (e.g. lapua-llm-v2 is overtrained
    #             to abstain — see README §11) and we prefer surfacing
    #             evidence to the user instead of hallucinated answers.
    answer_mode: Literal["synth", "retrieve"] = "synth"
    # BGE reranker-v2-m3 score below which we abstain without calling the LLM.
    # Typical relevant ~0..+5, irrelevant negative; 0.0 keeps the gate open.
    answer_min_score: float = 0.0
    # Number of reranked chunks fed into the LLM context. Must match or be
    # below SearchService.top_k_final (5) so we don't silently drop top results.
    answer_max_context_chunks: int = 5
    # Per-chunk character budget. 5 * 1500 = 7500 chars ≈ 1900 tokens, well
    # within Qwen2.5-1.5B's 4k window plus system + answer overhead.
    answer_max_chars_per_chunk: int = 1500
    # Retrieve-mode snippet length per source — longer than synth-mode's 200
    # because the user reads the snippet directly without LLM summarisation.
    answer_retrieve_snippet_chars: int = 600

    # OCR ---------------------------------------------------------------
    ocr_device: str = "gpu:0"
    ocr_use_vl_fallback: bool = True
    ocr_vl_confidence_threshold: float = 0.6

    # API ---------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Observability -----------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    @field_validator("storage_root", "inbox_dir", "index_dir", mode="after")
    @classmethod
    def _ensure_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings()
