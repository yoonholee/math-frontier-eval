"""Proof vs answer routing with separate corpus subsets.

KEY INSIGHT: Proof problems ("Prove that...", "Show that...") need examples of
proof structure (lemmas, induction, contradiction).  Computation problems need
numerical/algebraic examples.  By routing each problem type to a corpus subset
filtered by `problem_type` metadata, we surface more relevant exemplars.

Routing:
- Proof problems  → proof_corpus  (problem_type in {'converted_proof', 'proof'})
                    filtered to difficulty >= 6.0; supplemented if < 1000 rows
- Computation     → answer_corpus (full corpus with solution + difficulty)

Each sub-index is a separate BM25 instance.  Difficulty-band reranking (±2.0)
is applied to both to match evo_difficulty_matched_bm25's strategy.
"""

import re
import sys

from math_retriever import MathBM25, MathRetriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SOLUTION_CHARS_PROOF = 600
MAX_SOLUTION_CHARS_ANSWER = 400
PRE_RETRIEVE_K = 20
TOP_K = 3
DIFF_WINDOW = 2.0
FALLBACK_THRESHOLD = 2
PROOF_MIN_DIFFICULTY = 6.0
PROOF_MIN_ROWS = 1000

# ---------------------------------------------------------------------------
# Proof detection regex (exported so tests can import it directly)
# ---------------------------------------------------------------------------

_PROOF_RE = re.compile(
    r'\bprove\b|\bshow\s+that\b|\bdemonstrate\b|\bverify\s+that\b|\bestablish\s+that\b|'
    r'find\s+all\b.{0,60}(and\s+)?(prove|show)\b|\bif\s+and\s+only\s+if\b|\biff\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Preambles
# ---------------------------------------------------------------------------

_PROOF_PREAMBLE = (
    "Solve the following math problem with a rigorous proof or complete justification. "
    "Show all steps clearly, including any lemmas, base cases, or key structural arguments."
)

_COMP_PREAMBLE = (
    "Solve the following math problem step by step. Put your answer inside \\boxed{}."
)

_COMP_REMINDER = "Remember to put your answer inside \\boxed{}."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_proof_problem(problem: str) -> bool:
    return bool(_PROOF_RE.search(problem))


def _is_proof_type(row: dict) -> bool:
    """Return True if corpus row metadata indicates a proof-type problem."""
    pt = row.get('problem_type') or ''
    ans = row.get('answer') or ''
    return (
        'proof' in pt.lower()
        or pt == 'converted_proof'
        or ans.strip().lower() == 'proof'
    )


def _estimate_difficulty(problem: str) -> float:
    """Keyword-heuristic difficulty estimate (mirrors evo_difficulty_matched_bm25)."""
    p = problem
    if re.search(r"\bIMO\b|USAMO|Putnam|EGMO|Olympiad", p):
        return 8.0
    if re.search(r"\bAIME\b|HMMT|AMC 12|Harvard-MIT", p):
        return 7.0
    if re.search(r"AMC 10|AMC 8|\bSMT\b", p):
        return 5.5
    return 6.5  # corpus median fallback


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class EvoProofAnswerSplit(MathRetriever):
    """BM25 retrieval with proof-vs-computation corpus routing.

    Builds two separate indexes:
    - proof_index: over proof-type problems (difficulty >= 6.0)
    - answer_index: over full corpus (solution + difficulty present)

    At query time, detects whether the problem is a proof query and routes
    accordingly.  Both sub-retrievers apply difficulty-band reranking (±2.0).
    """

    def __init__(self, test_problems=None):
        super().__init__(test_problems)

        # ---- answer corpus: same as evo_difficulty_matched_bm25 ----
        self.answer_corpus = self.corpus.filter(
            lambda x: x["solution"] is not None and x["difficulty"] is not None
        )
        self.answer_difficulties = self.answer_corpus["difficulty"]
        self.answer_index = MathBM25(self.answer_corpus["problem"])

        # ---- proof corpus: metadata-filtered, high difficulty ----
        proof_hard = self.corpus.filter(
            lambda x: (
                x["solution"] is not None
                and x["difficulty"] is not None
                and _is_proof_type(x)
                and x["difficulty"] >= PROOF_MIN_DIFFICULTY
            )
        )

        # Supplement if too few rows: add any row whose answer == "proof"
        if len(proof_hard) < PROOF_MIN_ROWS:
            supplement = self.corpus.filter(
                lambda x: (
                    x["solution"] is not None
                    and x["difficulty"] is not None
                    and not _is_proof_type(x)  # avoid double-counting
                    and (x.get('answer') or '').strip().lower() == 'proof'
                )
            )
            # Concatenate proof_hard + supplement
            import datasets
            if len(supplement) > 0:
                proof_hard = datasets.concatenate_datasets([proof_hard, supplement])

        self.proof_corpus = proof_hard
        self.proof_difficulties = self.proof_corpus["difficulty"]
        self.proof_index = MathBM25(self.proof_corpus["problem"])

        print(
            f"[EvoProofAnswerSplit] proof_corpus={len(self.proof_corpus)}, "
            f"answer_corpus={len(self.answer_corpus)}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ #

    def build_prompt(self, problem: str) -> str:
        proof_mode = _is_proof_problem(problem)

        if proof_mode:
            context = self._retrieve(
                problem,
                corpus=self.proof_corpus,
                index=self.proof_index,
                difficulties=self.proof_difficulties,
                max_sol_chars=MAX_SOLUTION_CHARS_PROOF,
            )
            if context:
                return f"{_PROOF_PREAMBLE}\n\n{problem}\n\n{context}"
            return f"{_PROOF_PREAMBLE}\n\n{problem}"
        else:
            context = self._retrieve(
                problem,
                corpus=self.answer_corpus,
                index=self.answer_index,
                difficulties=self.answer_difficulties,
                max_sol_chars=MAX_SOLUTION_CHARS_ANSWER,
            )
            if context:
                return f"{_COMP_PREAMBLE}\n\n{problem}\n\n{context}\n\n{_COMP_REMINDER}"
            return f"{_COMP_PREAMBLE}\n\n{problem}\n\n{_COMP_REMINDER}"

    # ------------------------------------------------------------------ #

    def _retrieve(
        self,
        problem: str,
        corpus,
        index: MathBM25,
        difficulties,
        max_sol_chars: int,
    ) -> str:
        if len(corpus) == 0:
            return ""

        est_diff = _estimate_difficulty(problem)

        # Step 1: pre-retrieve top-20 BM25 candidates
        results = index.query(problem, k=PRE_RETRIEVE_K)
        if not results:
            return ""

        # Step 2: filter to candidates within difficulty window
        filtered = [
            (score, idx)
            for score, idx in results
            if abs(difficulties[idx] - est_diff) <= DIFF_WINDOW
        ]

        # Step 3: fallback to raw BM25 ranking if too few pass
        if len(filtered) < FALLBACK_THRESHOLD:
            filtered = results

        # Step 4: take top-3 by BM25 score (already sorted descending)
        top = filtered[:TOP_K]

        # Build context string
        parts = ["Here are some similar solved examples for reference:\n"]
        for i, (_, idx) in enumerate(top, 1):
            ex = corpus[idx]
            sol = ex["solution"]
            if len(sol) > max_sol_chars:
                sol = sol[:max_sol_chars] + "\n[... truncated]"
            entry = f"Example {i}:\nProblem: {ex['problem']}\nSolution: {sol}"
            if ex.get("answer"):
                entry += f"\nAnswer: {ex['answer']}"
            parts.append(entry)

        return "\n\n".join(parts)
