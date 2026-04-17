"""Verify PaddleOCR environment: Paddle GPU, paddleocr CLI, imports.

Prints a compact report of the full stack so we can detect regressions.
"""
from __future__ import annotations

import importlib
from typing import Iterable


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def _check_many(names: Iterable[str]) -> None:
    for name in names:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "n/a")
            print(f"  OK    {name:24s} {ver}")
        except Exception as exc:
            print(f"  FAIL  {name:24s} {type(exc).__name__}: {exc}")


def main() -> int:
    import sys

    section("Python")
    print(sys.version)

    section("paddle")
    import paddle

    print(f"paddle.__version__        = {paddle.__version__}")
    print(f"compiled_with_cuda()      = {paddle.is_compiled_with_cuda()}")
    print(f"cuda.device_count()       = {paddle.device.cuda.device_count()}")
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        print(f"cuda.get_device_name(0)   = {paddle.device.cuda.get_device_name(0)}")

    section("paddle.utils.run_check()")
    try:
        paddle.utils.run_check()
    except Exception as exc:  # pragma: no cover
        print(f"run_check raised: {type(exc).__name__}: {exc}")

    section("PaddleOCR core")
    _check_many(
        (
            "paddleocr",
            "paddlex",
            "paddleformers",
        )
    )

    section("Repo subpackages (editable)")
    _check_many(
        (
            "langchain_paddleocr",
            "paddleocr_mcp",
        )
    )

    section("Inference runtimes")
    _check_many(
        (
            "onnx",
            "onnxruntime",
            "paddle2onnx",
        )
    )

    section("Document / PDF pipeline")
    _check_many(
        (
            "cv2",
            "skimage",
            "albumentations",
            "pypdfium2",
            "pypandoc",
            "docx",
            "fitz" if False else "bs4",
            "lxml",
            "PIL",
        )
    )

    section("VLM / Transformers stack")
    _check_many(
        (
            "transformers",
            "accelerate",
            "timm",
            "sentencepiece",
            "tokenizers",
            "einops",
            "torch",
            "torchvision",
            "datasets",
        )
    )

    section("Serving / MCP")
    _check_many(
        (
            "fastapi",
            "uvicorn",
            "mcp",
            "fastmcp",
        )
    )

    section("LLM / GenAI client")
    _check_many(
        (
            "openai",
            "tiktoken",
            "langchain",
            "langchain_core",
            "langchain_openai",
            "langchain_community",
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
