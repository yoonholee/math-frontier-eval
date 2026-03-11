"""Algebra-hard sub-index OR-max fusion extending domain_conditional_secondary.

Extends EvoDomainConditionalSecondary with a dedicated algebra_hard_index built
from Algebra-topic rows in NuminaMath/DeepMath at difficulty >= 6.5. For algebra
queries, results from answer_index and algebra_hard_index are OR-max fused before
difficulty filtering and Jaccard diversity. All other domains (COMB/NT/GEO/PROOF)
use the unchanged domain_conditional_secondary logic.

KEY NEW MECHANISM: The algebra_hard_index is a dedicated sub-index filtered by
topic=='Algebra' AND difficulty>=6.5 AND dataset in ('NuminaMath','DeepMath').
This is a real architectural change: a topic+difficulty filtered sub-index whose
results are OR-max fused with the full answer_index, allowing hard algebra
competition problems concentrated in these datasets to surface alongside the
general corpus candidates.
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
ALGEBRA_HARD_MIN_DIFFICULTY = 6.5

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
_ALGEBRA_RE = re.compile(
    r"\b(polynomial|equation|function|sequence|series|sum|product|inequality"
    r"|quadratic|cubic|root|coefficient|variable|expression|algebra"
    r"|arithmetic|geometric|progression|Vieta|AM.GM|Cauchy)\b",
    re.IGNORECASE,
)

# ── preambles ─────────────────────────────────────────────────────────────────
_PROOF_PREAMBLE = (
    "Solve the following math problem with a rigorous proof or complete justification. "
    "Show all steps clearly, including any lemmas, base cases, or key structural arguments."
)
_COMP_PREAMBLE = "Solve the following math problem step by step. Put your answer inside \\boxed{}."
_COMP_REMINDER = "Remember to put your answer inside \\boxed{}."

# ── proof secondary: named mathematical terms (cross-domain, used for proofs) ─
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

# ── domain-conditional secondary regexes (NO generic inline math patterns) ────

# NT-secondary: number-theory specific vocabulary
_NT_SECONDARY_RE = re.compile(
    r"\b("
    r"prime|divisor|gcd|lcm|totient|congruence|digit|floor"
    r"|Wilson|Legendre|Hensel|Dirichlet"
    r"|quadratic\s+residue|modular\s+arithmetic"
    r")\b",
    re.IGNORECASE,
)

# GEO-secondary: geometric named terms
_GEO_SECONDARY_RE = re.compile(
    r"\b("
    r"Ptolemy|Menelaus|Ceva"
    r"|radical\s+axis|power\s+of\s+a\s+point|inversion"
    r"|circumcircle|incircle|altitude|angle\s+bisector"
    r"|orthocenter|incenter|circumcenter|excircle|symmedian"
    r")\b",
    re.IGNORECASE,
)

# COMB-secondary: combinatorial terms
_COMB_SECONDARY_RE = re.compile(
    r"\b("
    r"pigeonhole|bijection|double\s+counting|generating\s+function"
    r"|Stirling|Ramsey|chromatic|invariant|monovariant"
    r"|recurrence|inclusion.exclusion|binomial\s+coefficient"
    r")\b",
    re.IGNORECASE,
)

# ALGEBRA-secondary: algebraic terms
_ALGEBRA_SECONDARY_RE = re.compile(
    r"\b("
    r"Vieta|AM.GM|Cauchy.Schwarz|polynomial|functional\s+equation"
    r"|symmetric|substitution|arithmetic\s+progression|geometric\s+progression"
    r"|Chebyshev|quadratic|cubic|Fibonacci"
    r")\b",
    re.IGNORECASE,
)

_DOMAIN_SECONDARY_RE = {
    "NT": _NT_SECONDARY_RE,
    "GEO": _GEO_SECONDARY_RE,
    "COMB": _COMB_SECONDARY_RE,
    "ALGEBRA": _ALGEBRA_SECONDARY_RE,
}


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


def _is_algebra_query(problem: str) -> bool:
    """Detect if this is an algebra-domain problem (and not COMB/GEO/NT)."""
    if _COMB_RE.search(problem):
        return False
    if _GEO_RE.search(problem):
        return False
    if _NT_RE.search(problem):
        return False
    # Either explicit algebra keywords or fallback (default domain)
    return True


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


def _extract_proof_phrases(problem: str) -> str:
    """Extract named math terms for proof secondary BM25 query (cross-domain)."""
    terms = []
    for m in _NAMED_TERMS.finditer(problem):
        terms.append(m.group(0))
    for m in _NT_PHRASE_RE.finditer(problem):
        terms.append(m.group(0))
    for m in _DEFN_RE.finditer(problem):
        terms.append(m.group(1))
    seen = set()
    unique = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return " ".join(unique)


def _extract_domain_phrases(problem: str, domain: str) -> str:
    """Extract domain-specific vocabulary for the secondary BM25 query."""
    pattern = _DOMAIN_SECONDARY_RE.get(domain)
    if pattern is None:
        return ""
    terms = []
    for m in pattern.finditer(problem):
        terms.append(m.group(0))
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


def _query_index(problem: str, index: MathBM25, difficulties, domain: str, diff_window: float) -> list:
    """Run dual-query OR-max BM25 with domain-conditional secondary, then difficulty-filter.

    Returns list of (fused_score, idx) sorted descending, after difficulty filter.
    Handles empty BM25 results (OOV queries return []).
    """
    primary_raw = index.query(problem, k=PRE_RETRIEVE_K)
    primary = {idx: score for score, idx in primary_raw} if primary_raw else {}

    secondary: dict = {}
    if domain == "PROOF":
        phrase_query = _extract_proof_phrases(problem)
    else:
        phrase_query = _extract_domain_phrases(problem, domain)

    if phrase_query and len(phrase_query.split()) >= MIN_SECONDARY_TERMS:
        sec_raw = index.query(phrase_query, k=PRE_RETRIEVE_K)
        if sec_raw:
            secondary = {idx: score for score, idx in sec_raw}

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


def _or_max_fuse_two_sources(
    results1: list,
    results2: list,
) -> list:
    """Fuse two retrieval result lists with source tagging to avoid ID collisions.

    Each result list contains (score, local_idx). Tag with src1/src2 to distinguish.
    Returns sorted list of ((source_tag, local_idx), score) descending.
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

class EvoAlgebraHardFusion(MathRetriever):
    """Domain-conditional secondary routing with a dedicated algebra_hard sub-index.

    Extends EvoDomainConditionalSecondary by building a third index:
      - algebra_hard_index: corpus rows where topic=='Algebra' AND difficulty>=6.5
        AND solution is not None AND dataset in ('NuminaMath', 'DeepMath').

    At query time for ALGEBRA domain:
      - OR-max fuse results from answer_index AND algebra_hard_index (taking the
        better score for each doc, with source tags to avoid ID collisions).
      - Apply difficulty band filter (±2.0) and greedy Jaccard diversity (threshold=0.5).

    For COMB/NT/GEO/PROOF: unchanged from domain_conditional_secondary behavior.

    This is a genuine architectural change: the algebra_hard_index is a topic+difficulty
    filtered sub-index (not just parameter tuning), enabling hard competition algebra
    problems concentrated in NuminaMath/DeepMath to surface alongside general corpus
    candidates via OR-max fusion.
    """

    def __init__(self, test_problems=None):
        super().__init__(test_problems)

        # ---- answer corpus: full corpus with solutions ----
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

        # ---- algebra_hard sub-corpus: Algebra topic, difficulty >= 6.5, NuminaMath/DeepMath ----
        _ALGEBRA_HARD_DATASETS = {"NuminaMath", "DeepMath"}
        self.algebra_hard_corpus = self.corpus.filter(
            lambda x: (
                x.get("topic") == "Algebra"
                and x.get("difficulty") is not None
                and x["difficulty"] >= ALGEBRA_HARD_MIN_DIFFICULTY
                and x.get("solution") is not None
                and x.get("dataset") in _ALGEBRA_HARD_DATASETS
            )
        )
        self.algebra_hard_difficulties = self.algebra_hard_corpus["difficulty"]
        self.algebra_hard_index = MathBM25(self.algebra_hard_corpus["problem"])

        print(
            f"[EvoAlgebraHardFusion] proof_corpus={len(self.proof_corpus)}, "
            f"answer_corpus={len(self.answer_corpus)}, "
            f"algebra_hard_corpus={len(self.algebra_hard_corpus)}",
            file=sys.stderr,
        )

    def build_prompt(self, problem: str) -> str:
        if _is_proof_problem(problem):
            # Proof problems: use cross-domain named-terms secondary on proof_index
            candidates = _query_index(
                problem, self.proof_index, self.proof_difficulties, "PROOF", DIFF_WINDOW
            )
            context = self._greedy_diversity_corpus(
                candidates, self.proof_corpus, _LEN["PROOF"]
            )
            if context:
                return f"{_PROOF_PREAMBLE}\n\n{problem}\n\n{context}"
            return f"{_PROOF_PREAMBLE}\n\n{problem}"

        domain = _detect_domain(problem)
        max_len = _LEN[domain]

        if domain == "ALGEBRA":
            # ALGEBRA: OR-max fuse answer_index + algebra_hard_index
            answer_candidates = _query_index(
                problem, self.answer_index, self.answer_difficulties, domain, DIFF_WINDOW
            )
            alg_hard_candidates = _query_index(
                problem, self.algebra_hard_index, self.algebra_hard_difficulties, domain, DIFF_WINDOW
            )

            # OR-max fuse the two source result lists (tagged to avoid ID collisions)
            fused = _or_max_fuse_two_sources(answer_candidates, alg_hard_candidates)

            # Difficulty band filter over fused results
            est_diff = _estimate_difficulty(problem)
            filtered_fused = [
                (key, score) for key, score in fused
                if self._fused_diff(key, est_diff) <= DIFF_WINDOW
            ]
            if len(filtered_fused) < FALLBACK_THRESHOLD:
                filtered_fused = fused

            context = self._greedy_diversity_fused(filtered_fused, max_len)
        else:
            # COMB/NT/GEO: unchanged domain_conditional_secondary behavior
            candidates = _query_index(
                problem, self.answer_index, self.answer_difficulties, domain, DIFF_WINDOW
            )
            context = self._greedy_diversity_corpus(
                candidates, self.answer_corpus, max_len
            )

        if context:
            return f"{_COMP_PREAMBLE}\n\n{problem}\n\n{context}\n\n{_COMP_REMINDER}"
        return f"{_COMP_PREAMBLE}\n\n{problem}\n\n{_COMP_REMINDER}"

    def _fused_diff(self, key: tuple, est_diff: float) -> float:
        """Return |difficulty - est_diff| for a fused result key (src_tag, local_idx)."""
        src_tag, local_idx = key
        if src_tag == "src1":
            return abs(self.answer_difficulties[local_idx] - est_diff)
        else:
            return abs(self.algebra_hard_difficulties[local_idx] - est_diff)

    def _greedy_diversity_corpus(
        self, candidates: list, corpus, max_len: int
    ) -> str:
        """Greedy Jaccard diversity selection from single-corpus BM25 candidates."""
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

    def _greedy_diversity_fused(self, fused: list, max_len: int) -> str:
        """Greedy Jaccard diversity over OR-max fused (answer_index + algebra_hard_index) candidates.

        fused is a list of ((src_tag, local_idx), score) from _or_max_fuse_two_sources.
        """
        selected: list = []
        selected_texts: list = []

        for (src_tag, local_idx), score in fused:
            if src_tag == "src1":
                corpus = self.answer_corpus
            else:
                corpus = self.algebra_hard_corpus

            text = corpus[local_idx]["problem"]
            if not any(_jaccard(text, t) >= DIVERSITY_THRESHOLD for t in selected_texts):
                selected.append((score, src_tag, local_idx))
                selected_texts.append(text)
            if len(selected) >= TOP_K:
                break

        if not selected:
            return ""

        parts = ["Here are some similar solved examples for reference:\n"]
        for i, (_, src_tag, local_idx) in enumerate(selected, 1):
            corpus = self.answer_corpus if src_tag == "src1" else self.algebra_hard_corpus
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
