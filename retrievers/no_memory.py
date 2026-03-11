"""Baseline: no retrieval, minimal prompt with boxed instruction."""

from math_retriever import MathRetriever


class NoMemory(MathRetriever):
    def build_prompt(self, problem: str) -> str:
        return f"{problem}\n\nPut your answer inside \\boxed{{}}."
