"""Evaluation helpers: gold-set scorers + LLM-as-judge adapter."""

from lapua_rag.eval.judge import JudgeVerdict, judge_answer

__all__ = ["JudgeVerdict", "judge_answer"]
