"""Domain-conditional secondary query BM25 with proof/answer routing and Jaccard diversity.

Extends evo_proof_split_or_max_diversity by replacing the shared _extract_math_phrases
secondary query (which fires on all domains) with four domain-conditional secondary
query extractors.

The original secondary query extracted generic inline math patterns (a_i, f(x), x^2)
that create cross-domain noise. This candidate removes those generic patterns and instead
applies domain-specific vocabulary extraction per detected domain.

Domain-conditional secondary queries:
- NT:      number-theory terms (prime, divisor, gcd, lcm, totient, congruence, etc.)
- GEO:     geometric named terms (Ptolemy, Menelaus, radical axis, etc.)
- COMB:    combinatorial terms (pigeonhole, bijection, generating function, etc.)
- ALGEBRA: algebraic terms (Vieta, AM-GM, polynomial, functional equation, etc.)

For proof problems: the existing named-terms secondary is used (proofs are cross-domain).
All other hyperparameters unchanged from base (DIFF_WINDOW=2.0, DIVERSITY_THRESHOLD=0.5,
TOP_K=3, etc.).
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
    """Extract domain-specific vocabulary for the secondary BM25 query.

    Uses domain-conditional regexes that avoid generic inline math patterns
    (a_i, f(x), x^2) which create cross-domain noise.
    """
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


def _or_max_retrieve(problem: str, index: MathBM25, difficulties, diff_window: float) -> list:
    """Run dual-query OR-max BM25 then difficulty-filter (proof variant).

    Uses the cross-domain named-terms secondary query for proof problems.
    Returns list of (fused_score, idx) sorted descending, after difficulty filter.
    """
    primary = {idx: score for score, idx in index.query(problem, k=PRE_RETRIEVE_K)}

    secondary: dict = {}
    phrase_query = _extract_proof_phrases(problem)
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


def _or_max_retrieve_domain(
    problem: str,
    index: MathBM25,
    difficulties,
    domain: str,
    diff_window: float,
) -> list:
    """Run dual-query OR-max BM25 then difficulty-filter (domain-conditional variant).

    Uses the domain-specific secondary query rather than generic inline math patterns.
    Returns list of (fused_score, idx) sorted descending, after difficulty filter.
    """
    primary = {idx: score for score, idx in index.query(problem, k=PRE_RETRIEVE_K)}

    secondary: dict = {}
    phrase_query = _extract_domain_phrases(problem, domain)
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

class EvoDomainConditionalSecondary(MathRetriever):
    """Proof/computation routing with domain-conditional secondary BM25 and Jaccard diversity.

    Extends EvoProofSplitOrMaxDiversity by replacing the shared secondary query
    (which included generic inline math patterns like a_i, f(x), x^2 that create
    cross-domain noise) with four domain-conditional secondary query extractors.

    Builds two BM25 indexes:
      - answer_index: full corpus with solutions + difficulty
      - proof_index: proof-type corpus, high difficulty

    At query time:
      - Proof problems: OR-max retrieval on proof_index using cross-domain named-terms
        secondary, greedy Jaccard diversity
      - Computation problems: OR-max retrieval on answer_index using domain-conditional
        secondary vocabulary, greedy Jaccard diversity with domain-adaptive solution
        length caps
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
            f"[EvoDomainConditionalSecondary] proof_corpus={len(self.proof_corpus)}, "
            f"answer_corpus={len(self.answer_corpus)}",
            file=sys.stderr,
        )

    def build_prompt(self, problem: str) -> str:
        if _is_proof_problem(problem):
            # Proof problems: use cross-domain named-terms secondary
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
            # Computation problems: use domain-conditional secondary
            domain = _detect_domain(problem)
            max_len = _LEN[domain]
            candidates = _or_max_retrieve_domain(
                problem, self.answer_index, self.answer_difficulties, domain, DIFF_WINDOW
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
