"""Baseline: BM25 retrieval of top-K similar solved problems."""

from math_retriever import MathBM25, MathRetriever

K = 3
MAX_SOLUTION_CHARS = 3000


class BM25Retrieval(MathRetriever):
    """Retrieve top-K corpus problems most similar to the query via BM25."""

    def __init__(self, test_problems=None, k: int = K):
        super().__init__(test_problems)
        self.k = k
        # Filter to rows with solutions, build BM25 over problem text
        self.corpus = self.corpus.filter(lambda x: x["solution"] is not None)
        self.index = MathBM25(self.corpus["problem"])

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
        results = self.index.query(problem, k=self.k)
        parts = ["Here are some similar solved examples for reference:\n"]
        for i, (_, idx) in enumerate(results, 1):
            ex = self.corpus[idx]
            sol = ex["solution"]
            if len(sol) > MAX_SOLUTION_CHARS:
                sol = sol[:MAX_SOLUTION_CHARS] + "\n[... truncated]"
            parts.append(f"Example {i}:\nProblem: {ex['problem']}\nSolution: {sol}")
            if ex.get("answer"):
                parts[-1] += f"\nAnswer: {ex['answer']}"
        return "\n\n".join(parts)
