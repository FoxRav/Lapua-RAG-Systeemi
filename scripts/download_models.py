"""Pre-download Qwen2.5-1.5B-Instruct + lapua-llm-v2 LoRA adapter.

Runs outside the Lapua-RAG pipeline so the long download can be monitored
separately; caches into the standard HF hub cache so later runs hit it
directly.
"""

from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA = "CCG-FAKTUM/lapua-llm-v2"


def _download(repo_id: str, *, allow_patterns: list[str] | None = None) -> str:
    start = time.time()
    print(f"[download] {repo_id} …", flush=True)
    path = snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)
    elapsed = time.time() - start
    print(f"[done]     {repo_id} -> {path} ({elapsed:.1f} s)", flush=True)
    return path


def main() -> int:
    _download(
        BASE,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer*",
            "merges.txt",
            "vocab.json",
        ],
    )
    _download(LORA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
