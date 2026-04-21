"""Tests for scripts/train_v3_lora.py.

We only test the pure parts (argument parsing + JSONL loader) so the
suite stays hermetic — the real training loop requires Unsloth + CUDA
and only runs in WSL2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import train_v3_lora


class TestParseArgs:
    def test_defaults(self, tmp_path: Path) -> None:
        data = tmp_path / "dataset.jsonl"
        data.write_text("", encoding="utf-8")
        cfg = train_v3_lora.parse_args(["--data", str(data)])
        assert cfg.data == data
        assert cfg.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.epochs == 4
        assert cfg.lora_rank == 16
        assert cfg.batch_size == 2
        assert cfg.seed == 42

    def test_custom_epochs(self, tmp_path: Path) -> None:
        data = tmp_path / "dataset.jsonl"
        data.write_text("", encoding="utf-8")
        cfg = train_v3_lora.parse_args(["--data", str(data), "--epochs", "7"])
        assert cfg.epochs == 7

    def test_missing_required_data_argument(self) -> None:
        with pytest.raises(SystemExit):
            train_v3_lora.parse_args([])


class TestLoadRecords:
    def test_reads_jsonl_and_ignores_blank_lines(self, tmp_path: Path) -> None:
        data = tmp_path / "dataset.jsonl"
        data.write_text(
            json.dumps({"messages": [{"role": "user", "content": "Hei"}]})
            + "\n\n"
            + json.dumps({"messages": [{"role": "user", "content": "Maailma"}]})
            + "\n",
            encoding="utf-8",
        )
        records = train_v3_lora.load_records(data)
        assert len(records) == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            train_v3_lora.load_records(tmp_path / "missing.jsonl")
