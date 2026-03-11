"""Proof/answer routing with OR-max dual-query BM25 and Jaccard diversity.

Combines the proof-vs-computation corpus routing from evo_proof_answer_split
with two proven components from evo_book_augmented_geo:

1. OR-max dual-query BM25: secondary query using extracted math phrases,
   fused by element-wise max of BM25 scores.
2. Greedy Jaccard diversity: selects top-K examples that are mutually
   dissimilar (Jaccard < DIVERSITY_THRESHOLD) rather than the raw top-K.

Routing:
- Proof problems  → proof_corpus  (problem_type in {'converted_proof', 'proof'})
                    difficulty >= 6.0; supplemented if < 1000 rows
- Computation     → answer_corpus (full corpus with solution + difficulty)
                    domain-adaptive solution lengths (COMB=800, GEO=300, etc.)
"""

import re
import sys

from math_retriever import MathBM25, MathRetriever, math_tokenize

# ── retrieval hyperparameters ─────────────────────────────────────────────────
PRE_RETRIEVE_K = 20
TOP_K = 3
DIFF_WINDOW = 2.0
FALLBACK_THRESHOLD = 2
DIVERSITY_THRESHOLD = 0.5
MIN_SECONDARY_TERMS = 3
PROOF_MIN_DIFFICULTY = 6.0
PROOF_MIN_ROWS = 1000

# ── domain-adaptive solution lengths ─────────────────────────────────────────
_LEN = {
    "COMB": 800,
    "GEO": 300,
    "NT": 400,
    "ALGEBRA": 400,
    "PROOF": 600,
}

# ── proof detection regex ─────────────────────────────────────────────────────
_PROOF_RE = re.compile(
    r'\bprove\b|\bshow\s+that\b|\bdemonstrate\b|\bverify\s+that\b|\bestablish\s+that\b|'
    r'find\s+all\b.{0,60}(and\s+)?(prove|show)\b|\bif\s+and\s+only\s+if\b|\biff\b',
    re.IGNORECASE
)

# ── domain detection regexes ──────────────────────────────────────────────────
_COMB_RE = re.compile(
    r"\b(combinatorics?|counting|arrangement|pigeonhole|permutation|combination"
    r"|bijection|path|lattice|tournament|graph|coloring|tiling|choose|binomial"
    r"|committee|select|arrange|ways)\b",
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

# ── named mathematical terms for secondary query extraction ───────────────────
_NAMED_TERMS = re.compile(
    r"\b("
    r"Fermat|Euler|Cauchy|Pigeonhole|Polya|P[oó]lya|Lagrange|Bezout|B[eé]zout"
    r"|Vieta|AM-GM|AM.GM|Cauchy-Schwarz|Cauchy.Schwarz|Chebyshev|Stirling"
    r"|Ramsey|Wilson|Lucas|Hensel|Dirichlet|Gauss|Legendre|Jacobi"
    r"|incenter|circumcenter|orthocenter|centroid"
    r"|incircle|circumcircle|circumradius|inradius|excircle"
    r"|angle\s+bisector|altitude|symmedian|radical\s+axis|power\s+of\s+a\s+point"
    r"|Ptolemy|Menelaus|Ceva|Simson|Euler\s+line"
    r"|arithmetic\s+progression|geometric\s+progression|Fibonacci|recurrence"
    r"|polynomial|quadratic|cubic|binomial|multinomial"
    r"|totient|quadratic\s+residue|Legendre\s+symbol|primitive\s+root"
    r")\b",
    re.IGNORECASE,
)
_NT_PHRASE_RE = re.compile(
    r"\b(prime\s+p|divisor|gcd|lcm|Euler\s+phi|totient)\b",
    re.IGNORECASE,
)
_DEFN_RE = re.compile(
    r"\b(?:let|denote|define)\s+([A-Z][a-z]*\s+\w+)",
    re.IGNORECASE,
)
_INLINE_MATH_RE = re.compile(
    r"\b([a-zA-Z]_[a-zA-Z0-9]+|[a-zA-Z]\([a-z]\)|[a-zA-Z]\^[0-9]+)\b"
)


# ── standalone helpers ────────────────────────────────────────────────────────

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
    """Keyword-heuristic difficulty estimate."""
    p = problem
    if re.search(r"\bIMO\b|USAMO|Putnam|EGMO|Olympiad", p):
        return 8.0
    if re.search(r"\bAIME\b|HMMT|AMC 12|Harvard-MIT", p):
        return 7.0
    if re.search(r"AMC 10|AMC 8|\bSMT\b", p):
        return 5.5
    return 6.5  # corpus median fallback


def _detect_domain(problem: str) -> str:
    if _COMB_RE.search(problem):
        return "COMB"
    if _GEO_RE.search(problem):
        return "GEO"
    if _NT_RE.search(problem):
        return "NT"
    return "ALGEBRA"


def _extract_math_phrases(problem: str) -> str:
    """Extract named math terms and inline math patterns for secondary BM25 query."""
    terms = []
    for m in _NAMED_TERMS.finditer(problem):
        terms.append(m.group(0))
    for m in _NT_PHRASE_RE.finditer(problem):
        terms.append(m.group(0))
    for m in _DEFN_RE.finditer(problem):
        terms.append(m.group(1))
    for m in _INLINE_MATH_RE.finditer(problem):
        terms.append(m.group(1))
    seen = set()
    unique = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return " ".join(unique)


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _or_max_retrieve(problem: str, index: MathBM25, difficulties, diff_window: float) -> list:
    """Run dual-query OR-max BM25 then difficulty-filter.

    Returns list of (fused_score, idx) sorted descending, after difficulty filter.
    """
    primary = {idx: score for score, idx in index.query(problem, k=PRE_RETRIEVE_K)}

    secondary: dict = {}
    phrase_query = _extract_math_phrases(problem)
    if phrase_query and len(phrase_query.split()) >= MIN_SECONDARY_TERMS:
        secondary = {idx: score for score, idx in index.query(phrase_query, k=PRE_RETRIEVE_K)}

    all_idx = set(primary) | set(secondary)
    if not all_idx:
        return []

    fused = [
        (max(primary.get(idx, 0.0), secondary.get(idx, 0.0)), idx)
        for idx in all_idx
    ]
    fused.sort(key=lambda x: x[0], reverse=True)

    est_diff = _estimate_difficulty(problem)
    filtered = [
        (score, idx)
        for score, idx in fused
        if abs(difficulties[idx] - est_diff) <= diff_window
    ]
    if len(filtered) < FALLBACK_THRESHOLD:
        filtered = fused

    return filtered


# ── retriever class ───────────────────────────────────────────────────────────

class EvoProofSplitOrMaxDiversity(MathRetriever):
    """Proof/computation routing with OR-max dual-query BM25 and Jaccard diversity.

    Builds two BM25 indexes:
      - answer_index: full corpus with solutions + difficulty
      - proof_index: proof-type corpus, high difficulty

    At query time:
      - Proof problems: OR-max retrieval on proof_index, greedy Jaccard diversity
      - Computation problems: OR-max retrieval on answer_index, greedy Jaccard
        diversity with domain-adaptive solution length caps
    """

    def __init__(self, test_problems=None):
        super().__init__(test_problems)

        # ---- answer corpus ----
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
                import datasets
                proof_hard = datasets.concatenate_datasets([proof_hard, supplement])

        self.proof_corpus = proof_hard
        self.proof_difficulties = self.proof_corpus["difficulty"]
        self.proof_index = MathBM25(self.proof_corpus["problem"])

        print(
            f"[EvoProofSplitOrMaxDiversity] proof_corpus={len(self.proof_corpus)}, "
            f"answer_corpus={len(self.answer_corpus)}",
            file=sys.stderr,
        )

    def build_prompt(self, problem: str) -> str:
        if _is_proof_problem(problem):
            candidates = _or_max_retrieve(
                problem, self.proof_index, self.proof_difficulties, DIFF_WINDOW
            )
            context = self._greedy_diversity_corpus(
                candidates, self.proof_corpus, _LEN["PROOF"]
            )
            if context:
                return f"{_PROOF_PREAMBLE}\n\n{problem}\n\n{context}"
            return f"{_PROOF_PREAMBLE}\n\n{problem}"
        else:
            domain = _detect_domain(problem)
            max_len = _LEN[domain]
            candidates = _or_max_retrieve(
                problem, self.answer_index, self.answer_difficulties, DIFF_WINDOW
            )
            context = self._greedy_diversity_corpus(
                candidates, self.answer_corpus, max_len
            )
            if context:
                return f"{_COMP_PREAMBLE}\n\n{problem}\n\n{context}\n\n{_COMP_REMINDER}"
            return f"{_COMP_PREAMBLE}\n\n{problem}\n\n{_COMP_REMINDER}"

    def _greedy_diversity_corpus(
        self, candidates: list, corpus, max_len: int
    ) -> str:
        """Greedy Jaccard diversity selection from BM25 candidates."""
        selected: list = []
        selected_texts: list = []

        for score, idx in candidates:
            text = corpus[idx]["problem"]
            if not any(_jaccard(text, t) >= DIVERSITY_THRESHOLD for t in selected_texts):
                selected.append((score, idx))
                selected_texts.append(text)
            if len(selected) >= TOP_K:
                break

        if not selected:
            return ""

        parts = ["Here are some similar solved examples for reference:\n"]
        for i, (_, idx) in enumerate(selected, 1):
            ex = corpus[idx]
            sol = ex["solution"]
            if len(sol) > max_len:
                sol = sol[:max_len] + "\n[... truncated]"
            entry = f"Example {i}:\nProblem: {ex['problem']}\nSolution: {sol}"
            if ex.get("answer"):
                entry += f"\nAnswer: {ex['answer']}"
            parts.append(entry)

        return "\n\n".join(parts)
