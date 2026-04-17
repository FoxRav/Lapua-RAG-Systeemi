# PaddleOCR – Local Environment Build Plan

## Current state
- Empty workspace at `F:\-DEV-\76.PaddleOCR\`
- Python 3.10.11 available globally
- CUDA toolkit 11.8 installed; NVIDIA driver 581.04 (CUDA 13 capable)
- GPU: RTX 4050 Laptop, 6 GB VRAM (Ada, SM 8.9)
- Git 2.50 installed

## Target state
- `F:\-DEV-\76.PaddleOCR\PaddleOCR\` full upstream clone (main)
- `F:\-DEV-\76.PaddleOCR\.venv\` isolated Python 3.10 venv
- `paddlepaddle-gpu` (CUDA 11.8 build) installed
- `paddleocr` installed editable from the clone with **all available extras**
  (doc-parser / ie / trans / vl / serving / etc.)
- Optional VL-side heavy deps installed (transformers, accelerate, sentencepiece, timm)
- `paddle.utils.run_check()` passes with GPU visible
- `paddleocr --version` works from the venv

## Files to change
- Create `tmp/plan.md` (this)
- Create `README.md` (activation + quick-start notes)
- Clone `PaddleOCR/` (do not modify)
- Create `.venv/` (git-ignored)

## Checklist
- [x] Write plan
- [ ] Clone upstream repo
- [ ] Create venv and activate it
- [ ] Upgrade pip/setuptools/wheel
- [ ] Install paddlepaddle-gpu (CUDA 11.8)
- [ ] Inspect pyproject extras, install editable with all extras
- [ ] Install VL-side deps (if not pulled by extras)
- [ ] `paddle.utils.run_check()` – confirm GPU
- [ ] `paddleocr --version` – confirm CLI
- [ ] Write README with activation instructions

## Notes / constraints
- CUDA 11.8 chosen to match installed toolkit. Paddle-gpu cu118 wheels are
  stable for Windows + Py3.10.
- 6 GB VRAM: PaddleOCR-VL-0.9B fits; heavier VLMs may OOM – fallback to CPU
  for VL if needed.
- Keep clone read-only; all edits go through extras/venv.
