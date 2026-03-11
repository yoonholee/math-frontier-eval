"""Proof/answer routing with DeepMath hard sub-index injected for algebra/NT.

Extends the three-branch routing architecture from evo_proof_split_or_max_diversity
with a fourth DeepMath hard sub-index as a parallel OR-max source for algebra and
NT answer branches.

KEY NEW MECHANISM: For algebra/NT/combinatorics answer queries, the retriever
performs OR-max fusion between the main answer_index AND a deepmath_hard_index
(DeepMath corpus filtered to difficulty >= 7.0, Algebra/NT topics). This injects
hard, competition-style worked examples that are concentrated in DeepMath's
high-difficulty tail and may be under-represented in the main answer_index BM25 ranking.
Geometry answer queries use only the regular answer_index (DeepMath provides
minimal geometry gains based on prior iterations).

Pipeline:
1. Proof detection → proof_index branch (unchanged from base).
2. Domain detection (GEO / COMB / NT / ALGEBRA).
3. For proof branch: OR-max dual-query BM25 on proof_index, Jaccard diversity.
4. For GEO answer branch: OR-max dual-query on answer_index only, Jaccard diversity.
5. For Algebra/NT/COMB answer branch: OR-max dual-query on answer_index,
   PLUS parallel OR-max fusion with deepmath_hard_index, then Jaccard diversity.
6. Domain-adaptive solution length caps (same as base).
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
DEEPMATH_MIN_DIFFICULTY = 7.0

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


def _is_deepmath_hard_alg_nt(row: dict) -> bool:
    """Return True if row is a DeepMath hard Algebra/NT problem."""
    if row.get("dataset") != "DeepMath":
        return False
    diff = row.get("difficulty")
    if diff is None or diff < DEEPMATH_MIN_DIFFICULTY:
        return False
    if row.get("solution") is None:
        return False
    topic = row.get("topic") or ""
    # Allow None/empty topics (36% fill rate) + Algebra + Number Theory
    return topic in ("Algebra", "Number Theory", None, "")


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


def _or_max_fuse_index_results(results1: list, results2: list) -> list:
    """Fuse two retrieval result lists by element-wise max of scores.

    For docs appearing in both lists, take the max score.
    For docs in only one list, use that score.
    Returns sorted list of (score, doc_id) descending.

    Note: doc_ids are LOCAL indices within each corpus (not global).
    results1 uses corpus1 local ids, results2 uses corpus2 local ids.
    We tag each doc_id with its source to avoid collisions.
    """
    scores: dict = {}
    for score, doc_id in results1:
        key = ("src1", doc_id)
        scores[key] = max(scores.get(key, 0.0), score)
    for score, doc_id in results2:
        key = ("src2", doc_id)
        scores[key] = max(scores.get(key, 0.0), score)
    return sorted(scores.items(), key=lambda x: -x[1])


# ── retriever class ───────────────────────────────────────────────────────────

class EvoDeepMathHardAugment(MathRetriever):
    """Proof/computation routing with DeepMath hard sub-index for algebra/NT.

    Builds four indexes:
      - answer_index: full corpus with solutions + difficulty
      - proof_index: proof-type corpus, high difficulty
      - deepmath_hard_index: DeepMath corpus filtered to difficulty >= 7.0
        and topic in (Algebra, Number Theory, None/empty)

    At query time:
      - Proof problems: OR-max retrieval on proof_index, greedy Jaccard diversity
      - GEO answer queries: OR-max on answer_index only (no DeepMath augment)
      - Algebra/NT/COMB answer queries: OR-max on answer_index, THEN fuse with
        OR-max on deepmath_hard_index; Jaccard diversity over the merged pool
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

        # ---- DeepMath hard sub-corpus: algebra/NT, difficulty >= 7.0 ----
        self.deepmath_hard_corpus = self.corpus.filter(_is_deepmath_hard_alg_nt)
        self.deepmath_hard_difficulties = self.deepmath_hard_corpus["difficulty"]
        self.deepmath_hard_index = MathBM25(self.deepmath_hard_corpus["problem"])

        print(
            f"[EvoDeepMathHardAugment] proof_corpus={len(self.proof_corpus)}, "
            f"answer_corpus={len(self.answer_corpus)}, "
            f"deepmath_hard_corpus={len(self.deepmath_hard_corpus)}",
            file=sys.stderr,
        )

    def build_prompt(self, problem: str) -> str:
        # ---- Proof branch ----
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

        # ---- Answer branch ----
        domain = _detect_domain(problem)
        max_len = _LEN[domain]

        # Main answer index retrieval
        answer_candidates = _or_max_retrieve(
            problem, self.answer_index, self.answer_difficulties, DIFF_WINDOW
        )

        if domain == "GEO":
            # GEO: use only the main answer_index (DeepMath geo gains minimal)
            context = self._greedy_diversity_corpus(
                answer_candidates, self.answer_corpus, max_len
            )
        else:
            # Algebra/NT/COMB: OR-max fuse with DeepMath hard sub-index
            deepmath_candidates = _or_max_retrieve(
                problem, self.deepmath_hard_index,
                self.deepmath_hard_difficulties, DIFF_WINDOW
            )

            # Fuse results from both sources (tagged by source to avoid id collisions)
            fused = _or_max_fuse_index_results(answer_candidates, deepmath_candidates)

            # Build a unified candidate list with (score, source, local_idx)
            context = self._greedy_diversity_fused(
                fused, max_len
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
                selected.append((score, idx, corpus))
                selected_texts.append(text)
            if len(selected) >= TOP_K:
                break

        if not selected:
            return ""

        parts = ["Here are some similar solved examples for reference:\n"]
        for i, (_, idx, corp) in enumerate(selected, 1):
            ex = corp[idx]
            sol = ex["solution"]
            if len(sol) > max_len:
                sol = sol[:max_len] + "\n[... truncated]"
            entry = f"Example {i}:\nProblem: {ex['problem']}\nSolution: {sol}"
            if ex.get("answer"):
                entry += f"\nAnswer: {ex['answer']}"
            parts.append(entry)

        return "\n\n".join(parts)

    def _greedy_diversity_fused(
        self, fused: list, max_len: int
    ) -> str:
        """Greedy Jaccard diversity over fused (answer_index + deepmath_hard_index) candidates.

        fused is a list of ((source, local_idx), score) from _or_max_fuse_index_results.
        source is 'src1' (answer_corpus) or 'src2' (deepmath_hard_corpus).
        """
        selected: list = []
        selected_texts: list = []

        for (source, local_idx), score in fused:
            if source == "src1":
                corpus = self.answer_corpus
            else:
                corpus = self.deepmath_hard_corpus

            text = corpus[local_idx]["problem"]
            if not any(_jaccard(text, t) >= DIVERSITY_THRESHOLD for t in selected_texts):
                selected.append((score, source, local_idx))
                selected_texts.append(text)
            if len(selected) >= TOP_K:
                break

        if not selected:
            # ungated fallback: take top-2 from answer_corpus directly
            fallback_candidates = self.answer_index.query("", k=2)
            if not fallback_candidates:
                return ""
            selected = [(s, "src1", idx) for s, idx in fallback_candidates[:2]]

        parts = ["Here are some similar solved examples for reference:\n"]
        for i, (_, source, local_idx) in enumerate(selected, 1):
            corpus = self.answer_corpus if source == "src1" else self.deepmath_hard_corpus
            ex = corpus[local_idx]
            sol = ex["solution"]
            if sol is None:
                continue
            if len(sol) > max_len:
                sol = sol[:max_len] + "\n[... truncated]"
            entry = f"Example {i}:\nProblem: {ex['problem']}\nSolution: {sol}"
            if ex.get("answer"):
                entry += f"\nAnswer: {ex['answer']}"
            parts.append(entry)

        if len(parts) == 1:
            return ""

        return "\n\n".join(parts)
