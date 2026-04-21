"""lapua-llm-v3 LoRA training script (Unsloth, run in WSL2 with CUDA).

Usage (inside the ``lapua-vllm`` WSL distro, vLLM stopped to free VRAM)::

    python /mnt/f/-DEV-/76.PaddleOCR/scripts/train_v3_lora.py \
        --data /root/training/v3_dataset.jsonl \
        --output /root/lapua-llm-v3 \
        --epochs 4

The script is intentionally split into two phases:

* ``parse_args()`` runs immediately and imports only the standard library,
  so ``--help`` works on Windows or in CI without the Unsloth / torch
  stack available.
* ``run(...)`` imports the heavy ML dependencies lazily — only when a real
  training run is invoked.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class TrainConfig:
    """Frozen view of the CLI arguments consumed by :func:`run`."""

    data: Path
    output: Path
    base_model: str
    epochs: int
    lora_rank: int
    batch_size: int
    grad_accum: int
    lr: float
    eval_split: float
    seed: int


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse CLI flags into a :class:`TrainConfig`.

    Kept free of heavy imports so the script's ``--help`` works outside
    WSL2 and in unit tests where torch / unsloth are not installed.
    """
    parser = argparse.ArgumentParser(
        description="Train lapua-llm-v3 LoRA on balanced ChatML data via Unsloth.",
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to the ChatML JSONL dataset.")
    parser.add_argument("--output", type=Path, default=Path("/root/lapua-llm-v3"),
                        help="Directory where the trained LoRA adapter will be saved.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model id for the base model.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = bs * grad_accum).")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction of the dataset reserved for validation.")
    parser.add_argument("--seed", type=int, default=42)
    ns = parser.parse_args(argv)
    return TrainConfig(
        data=ns.data,
        output=ns.output,
        base_model=ns.base_model,
        epochs=ns.epochs,
        lora_rank=ns.lora_rank,
        batch_size=ns.batch_size,
        grad_accum=ns.grad_accum,
        lr=ns.lr,
        eval_split=ns.eval_split,
        seed=ns.seed,
    )


def load_records(path: Path) -> list[dict[str, object]]:
    """Load a ChatML JSONL file into a list of plain dicts.

    Kept pure (no tokenization) so unit tests don't need a real
    tokenizer to verify I/O.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def run(cfg: TrainConfig) -> None:  # pragma: no cover - requires GPU stack
    """Execute the training loop. Requires Unsloth + CUDA at import time."""
    # Heavy imports are deferred until run() so that --help / unit tests
    # don't need the WSL2 GPU stack installed.
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import TrainingArguments  # noqa: PLC0415
    from trl import SFTTrainer  # noqa: PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    records = load_records(cfg.data)

    def format_example(ex: dict[str, object]) -> dict[str, str]:
        msgs = ex["messages"]
        return {
            "text": tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            ),
        }

    dataset = Dataset.from_list(records).map(format_example)
    split = dataset.train_test_split(test_size=cfg.eval_split, seed=cfg.seed)

    training_args = TrainingArguments(
        output_dir=str(cfg.output / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )

    print(f"\nTraining {len(split['train'])} examples × {cfg.epochs} epochs.")
    print(f"Validation: {len(split['test'])} examples.")
    trainer.train()

    cfg.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output))
    tokenizer.save_pretrained(str(cfg.output))
    print(f"\nLoRA adapter saved: {cfg.output}")
    print("Next step: publish to HuggingFace as CCG-FAKTUM/lapua-llm-v3.")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    cfg = parse_args(argv)
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
