"""Baseline: random K problem-solution pairs from corpus as few-shot examples."""

import random

from math_retriever import MathRetriever

K = 3
MAX_SOLUTION_CHARS = 3000


class RandomFewshot(MathRetriever):
    """Prepend K random solved examples from the corpus."""

    def __init__(self, test_problems=None, k: int = K, seed: int = 0):
        super().__init__(test_problems)
        self.k = k
        self.rng = random.Random(seed)
        # Filter to rows with solutions
        self.corpus = self.corpus.filter(lambda x: x["solution"] is not None)

    def build_prompt(self, problem: str) -> str:
        preamble = "Solve the following math problem step by step. Put your answer inside \\boxed{}."
        reminder = "Remember to put your answer inside \\boxed{}."
        context = self._retrieve(problem)
        if context:
            return f"{preamble}\n\n{problem}\n\n{context}\n\n{reminder}"
        return f"{preamble}\n\n{problem}\n\n{reminder}"

    def _retrieve(self, problem: str) -> str:
        if len(self.corpus) == 0 or self.k == 0:
            return ""
        indices = self.rng.sample(range(len(self.corpus)), min(self.k, len(self.corpus)))
        parts = ["Here are some solved examples for reference:\n"]
        for i, idx in enumerate(indices, 1):
            ex = self.corpus[idx]
            sol = ex["solution"]
            if len(sol) > MAX_SOLUTION_CHARS:
                sol = sol[:MAX_SOLUTION_CHARS] + "\n[... truncated]"
            parts.append(f"Example {i}:\nProblem: {ex['problem']}\nSolution: {sol}")
            if ex.get("answer"):
                parts[-1] += f"\nAnswer: {ex['answer']}"
        return "\n\n".join(parts)
