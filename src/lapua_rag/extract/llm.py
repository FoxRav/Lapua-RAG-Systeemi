"""LLM client abstraction.

Two implementations:

* :class:`LocalLlmClient`   – transformers + PEFT, CPU or GPU, in-process.
* :class:`RemoteVllmClient` – OpenAI-compatible vLLM endpoint (Linux / WSL2).

Heavy imports are deferred to the subclasses' ``load`` methods so the module is
importable without transformers/torch loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from lapua_rag.config import get_settings
from lapua_rag.observability import get_logger

_log = get_logger(__name__)


class LlmClient(Protocol):
    """Minimal interface consumed by the extraction pipeline."""

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict[str, object],
    ) -> dict[str, object]:
        ...


@dataclass(slots=True)
class LocalLlmClient:
    """Local Qwen2.5 + LoRA client using transformers + PEFT + lm-format-enforcer."""

    base_model: str
    lora_adapter: str
    device: str
    dtype: str
    max_new_tokens: int = 512
    _model: object | None = field(default=None, repr=False, compare=False)
    _tokenizer: object | None = field(default=None, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch  # imported lazily
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[self.dtype]

        _log.info(
            "llm.local_load",
            base=self.base_model,
            lora=self.lora_adapter,
            device=self.device,
            dtype=self.dtype,
        )
        # low_cpu_mem_usage avoids a 2x weight-size RAM spike during load
        # (meta init + single copy instead of two full-precision copies).
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, self.lora_adapter)
        if self.device == "cpu":
            model = model.to("cpu")
        model.eval()
        self._model = model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self._tokenizer = tokenizer

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict[str, object],
    ) -> dict[str, object]:
        self._ensure_loaded()
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        assert self._model is not None
        assert self._tokenizer is not None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # return_dict=True so we get both input_ids and attention_mask;
        # passing attention_mask explicitly silences the transformers warning
        # and is required for reliable generation when pad == eos.
        encoded = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            self._tokenizer, JsonSchemaParser(json_schema),
        )
        output = self._model.generate(  # type: ignore[attr-defined]
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_fn,
            pad_token_id=self._tokenizer.pad_token_id,  # type: ignore[attr-defined]
        )
        generated = output[0][input_ids.shape[-1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)  # type: ignore[attr-defined]
        return json.loads(text)


@dataclass(slots=True)
class RemoteVllmClient:
    """OpenAI-compatible vLLM server (preferred for batch extraction)."""

    base_url: str
    model: str
    max_new_tokens: int = 512
    timeout: float = 120.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict[str, object],
    ) -> dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "guided_json": json_schema,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


def default_client() -> LlmClient:
    """Factory: remote vLLM when configured, otherwise local."""
    settings = get_settings()
    if settings.llm_vllm_url:
        return RemoteVllmClient(
            base_url=settings.llm_vllm_url.rstrip("/"),
            model=settings.llm_base,
            max_new_tokens=settings.llm_max_new_tokens,
        )
    return LocalLlmClient(
        base_model=settings.llm_base,
        lora_adapter=settings.llm_lora,
        device=settings.llm_device,
        dtype=settings.llm_dtype,
        max_new_tokens=settings.llm_max_new_tokens,
    )
