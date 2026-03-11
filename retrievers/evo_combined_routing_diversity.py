"""Combined proof/computation routing + greedy Jaccard diversity selection.

Merges two top-performing systems:
- EvoProofAnswerSplit: dual-corpus routing (proof index + computation index)
- EvoDiversityRerank: greedy Jaccard diversity selection on retrieved candidates

KEY INSIGHT: Routing ensures each problem sees corpus examples from the most
relevant sub-corpus (proof-structure examples for proof problems; numerical/
algebraic examples for computation).  Diversity selection then ensures the
top-3 chosen examples are structurally distinct, giving the model a broader
range of strategies.

Pipeline (per query):
1. Detect proof vs computation via full regex (prove/show/demonstrate/verify/iff).
2. Route to proof_index (proof-type corpus, difficulty >= 6.0) or answer_index
   (full corpus with solution + difficulty).
3. BM25 top-20 retrieval from the branch-specific index.
4. ±2.0 difficulty-band filter with fallback if < 2 pass.
5. Greedy Jaccard diversity selection (threshold=0.5) on PROBLEM TEXT token sets
   to pick top-3.
6. Domain-adaptive solution lengths:
   - Proof branch: 600 chars
   - Computation branch: COMB=800 / GEO=300 / NT=400 / ALGEBRA=400 chars
"""

import re
import sys

import datasets

from math_retriever import MathBM25, MathRetriever, math_tokenize

# ── retrieval hyperparameters ─────────────────────────────────────────────────
PRE_RETRIEVE_K = 20
TOP_K = 3
DIFF_WINDOW = 2.0
FALLBACK_THRESHOLD = 2
PROOF_MIN_DIFFICULTY = 6.0
PROOF_MIN_ROWS = 1000

# ── diversity threshold ───────────────────────────────────────────────────────
JACCARD_SIM_THRESHOLD = 0.5

# ── domain-adaptive solution lengths ─────────────────────────────────────────
MAX_SOLUTION_CHARS_PROOF = 600
_LEN_COMP = {
    "COMB": 800,
    "GEO": 300,
    "NT": 400,
    "ALGEBRA": 400,
}

# ── proof detection regex (comprehensive, from evo_proof_answer_split) ────────
_PROOF_RE = re.compile(
    r'\bprove\b|\bshow\s+that\b|\bdemonstrate\b|\bverify\s+that\b|\bestablish\s+that\b|'
    r'find\s+all\b.{0,60}(and\s+)?(prove|show)\b|\bif\s+and\s+only\s+if\b|\biff\b',
    re.IGNORECASE
)

# ── domain detection regexes (from evo_diversity_rerank) ─────────────────────
_COMB_RE = re.compile(
    r"\b(combinatorics?|counting|arrangement|pigeonhole|permutation|combination"
    r"|bijection|path|lattice|tournament|graph|coloring|tiling|choose|binomial"
    r"|committee)\b",
    re.IGNORECASE,
)
_GEO_RE = re.compile(
    r"\b(triangle|circle|polygon|angle|perpendicular|parallel|circumscribed"
    r"|inscribed|tangent|chord|arc|radius|diameter|midpoint|centroid|incircle"
    r"|circumcircle|orthocenter|altitude|median)\b",
    re.IGNORECASE,
)
_NT_RE = re.compile(
    r"\b(prime|divisible|divisor|modulo|mod|congruent|gcd|lcm|floor|ceiling"
    r"|digit|integer|remainder|factor|coprime|euler|phi|fermat"
    r"|quadratic residue)\b",
    re.IGNORECASE,
)

# ── preambles ─────────────────────────────────────────────────────────────────
_PROOF_PREAMBLE = (
    "Solve the following math problem with a rigorous proof or complete justification. "
    "Show all steps clearly, including any lemmas, base cases, or key structural arguments."
)
_COMP_PREAMBLE = "Solve the following math problem step by step. Put your answer inside \\boxed{}."
_COMP_REMINDER = "Remember to put your answer inside \\boxed{}."


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_proof_problem(problem: str) -> bool:
    """Detect proof-style problems via comprehensive regex."""
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
    """Keyword-heuristic difficulty estimate."""
    p = problem  # case-sensitive for competition names
    if re.search(r"\bIMO\b|USAMO|Putnam|EGMO|Olympiad", p):
        return 8.0
    if re.search(r"\bAIME\b|HMMT|AMC 12|Harvard-MIT", p):
        return 7.0
    if re.search(r"AMC 10|AMC 8|\bSMT\b", p):
        return 5.5
    return 6.5  # corpus median fallback


def _detect_domain(problem: str) -> str:
    """Detect math domain. Priority: COMB > GEO > NT > ALGEBRA."""
    if _COMB_RE.search(problem):
        return "COMB"
    if _GEO_RE.search(problem):
        return "GEO"
    if _NT_RE.search(problem):
        return "NT"
    return "ALGEBRA"


def _jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _greedy_diversity_select(candidates: list, corpus, k: int = TOP_K) -> list:
    """Greedy Jaccard-based diversity selection.

    Given BM25 candidates (score, idx) sorted descending, greedily pick up to
    k examples such that no two selected examples have Jaccard similarity > 0.5
    on their PROBLEM TEXT token sets.

    The highest BM25-scoring candidate is always included first.
    """
    selected = []
    selected_tokens = []

    for score, idx in candidates:
        problem_text = corpus[idx]["problem"]
        tokens = set(math_tokenize(problem_text))

        if not selected:
            selected.append((score, idx))
            selected_tokens.append(tokens)
            continue

        too_similar = any(
            _jaccard_similarity(tokens, sel_tokens) > JACCARD_SIM_THRESHOLD
            for sel_tokens in selected_tokens
        )

        if not too_similar:
            selected.append((score, idx))
            selected_tokens.append(tokens)

        if len(selected) >= k:
            break

    return selected


# ── retriever ─────────────────────────────────────────────────────────────────

class EvoCombinedRoutingDiversity(MathRetriever):
    """Dual-corpus routing with greedy Jaccard diversity selection.

    Combines:
    - Proof/computation corpus routing (EvoProofAnswerSplit): two separate BM25
      indexes ensure each problem type retrieves from the most semantically
      aligned sub-corpus.
    - Greedy diversity reranking (EvoDiversityRerank): after BM25 + difficulty-
      band filtering in the chosen branch, Jaccard deduplication ensures the
      top-3 examples are structurally distinct.

    Solution lengths are domain-adaptive for computation and fixed for proof.
    """

    def __init__(self, test_problems=None):
        super().__init__(test_problems)

        # ── answer corpus: full corpus with solution + difficulty ──────────────
        self.answer_corpus = self.corpus.filter(
            lambda x: x["solution"] is not None and x["difficulty"] is not None
        )
        self.answer_difficulties = self.answer_corpus["difficulty"]
        self.answer_index = MathBM25(self.answer_corpus["problem"])

        # ── proof corpus: metadata-filtered, high difficulty ──────────────────
        proof_hard = self.corpus.filter(
            lambda x: (
                x["solution"] is not None
                and x["difficulty"] is not None
                and _is_proof_type(x)
                and x["difficulty"] >= PROOF_MIN_DIFFICULTY
            )
        )

        # Supplement if too few rows
        if len(proof_hard) < PROOF_MIN_ROWS:
            supplement = self.corpus.filter(
                lambda x: (
                    x["solution"] is not None
                    and x["difficulty"] is not None
                    and not _is_proof_type(x)
                    and (x.get('answer') or '').strip().lower() == 'proof'
                )
            )
            if len(supplement) > 0:
                proof_hard = datasets.concatenate_datasets([proof_hard, supplement])

        self.proof_corpus = proof_hard
        self.proof_difficulties = self.proof_corpus["difficulty"]
        self.proof_index = MathBM25(self.proof_corpus["problem"])

        print(
            f"[EvoCombinedRoutingDiversity] proof_corpus={len(self.proof_corpus)}, "
            f"answer_corpus={len(self.answer_corpus)}",
            file=sys.stderr,
        )

    # ── public interface ───────────────────────────────────────────────────────

    def build_prompt(self, problem: str) -> str:
        proof_mode = _is_proof_problem(problem)

        if proof_mode:
            context = self._retrieve_and_diversify(
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
            domain = _detect_domain(problem)
            max_sol_chars = _LEN_COMP[domain]
            context = self._retrieve_and_diversify(
                problem,
                corpus=self.answer_corpus,
                index=self.answer_index,
                difficulties=self.answer_difficulties,
                max_sol_chars=max_sol_chars,
            )
            if context:
                return f"{_COMP_PREAMBLE}\n\n{problem}\n\n{context}\n\n{_COMP_REMINDER}"
            return f"{_COMP_PREAMBLE}\n\n{problem}\n\n{_COMP_REMINDER}"

    # ── internal pipeline ──────────────────────────────────────────────────────

    def _retrieve_and_diversify(
        self,
        problem: str,
        corpus,
        index: MathBM25,
        difficulties,
        max_sol_chars: int,
    ) -> str:
        """BM25 top-20 → difficulty-band filter → greedy diversity → top-3."""
        if len(corpus) == 0:
            return ""

        est_diff = _estimate_difficulty(problem)

        # Step 1: BM25 pre-retrieval
        results = index.query(problem, k=PRE_RETRIEVE_K)
        if not results:
            return ""

        # Step 2: difficulty-band filter (±2.0)
        filtered = [
            (score, idx)
            for score, idx in results
            if abs(difficulties[idx] - est_diff) <= DIFF_WINDOW
        ]

        # Step 3: fallback to raw BM25 if too few pass
        if len(filtered) < FALLBACK_THRESHOLD:
            filtered = results

        # Step 4: greedy Jaccard diversity selection (threshold=0.5)
        top = _greedy_diversity_select(filtered, corpus, k=TOP_K)

        # Step 5: build context with domain-adaptive truncation
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
