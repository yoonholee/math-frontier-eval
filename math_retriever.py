"""Base class and retrieval primitives for math retrieval systems.

MathRetriever: base class, contract is build_prompt(problem) -> str.
Primitives: normalize(), math_tokenize(), MathBM25, MathDense.
"""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenized
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parent


class MathRetriever(ABC):
    """Base class. Only contract: build_prompt(problem) -> str."""

    def __init__(self, test_problems: list[dict] = None):
        BOOKS_REPO = "yoonholee/olympiad-books-open-source"
        CORPUS_REPO = "yoonholee/math-corpus-combined"
        self.books = load_dataset(BOOKS_REPO, split="train")
        self.corpus = load_dataset(CORPUS_REPO, split="train")

        self.test_problems = test_problems or []

    @abstractmethod
    def build_prompt(self, problem: str) -> str:
        """Return the complete prompt for this problem."""


# ── Text normalization ────────────────────────────────────────────────

_RE_DISPLAY = re.compile(r"\\(?:displaystyle|textstyle|scriptstyle)\s*")
_RE_DELIM = re.compile(r"\\(?:left|right|big|Big|bigg|Bigg)([|().\[\]{}])")
_RE_DELIM2 = re.compile(r"\\(?:left|right|big|Big|bigg|Bigg)\b\s*")
_RE_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize LaTeX to canonical forms. ~0.07ms/doc."""
    s = _RE_DISPLAY.sub("", text)
    s = _RE_DELIM.sub(r"\1", s)
    s = _RE_DELIM2.sub("", s)
    s = s.replace("\\leqslant", "\\le").replace("\\geqslant", "\\ge")
    s = s.replace("\\leq", "\\le").replace("\\geq", "\\ge")
    s = s.replace("\\neq", "\\ne")
    s = s.replace("\\lvert", "|").replace("\\rvert", "|")
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    s = s.replace("\\operatorname", "\\mathrm")
    s = s.replace("\\mathbb", "\\mathbb").replace("\\mathbf", "\\mathbf")
    s = _RE_WS.sub(" ", s).strip()
    return s


# ── Math-aware tokenizer ─────────────────────────────────────────────

# Captures: \commands, ^{...}, _{...}, words, numbers, single chars
_MATH_TOKEN = re.compile(r"\\[a-zA-Z]+|[_^]\{[^}]*\}|[a-zA-Z]+|[0-9]+|\S")


def math_tokenize(text: str) -> list[str]:
    r"""Tokenize preserving LaTeX commands (\frac, \sum), superscripts, subscripts."""
    return _MATH_TOKEN.findall(text.lower())


# ── BM25 with math tokenizer ─────────────────────────────────────────


def _make_tokenized(docs_tokens: list[list[str]]) -> Tokenized:
    """Build bm25s Tokenized from pre-tokenized docs. Preserves backslashes."""
    vocab = {}
    ids = []
    for doc in docs_tokens:
        doc_ids = []
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            doc_ids.append(vocab[tok])
        ids.append(doc_ids)
    return Tokenized(ids=ids, vocab=vocab)


class MathBM25:
    """BM25 index with math-aware tokenizer.

    Usage:
        # From strings:
        idx = MathBM25(["problem 1 with \\frac{a}{b}", ...])

        # From dicts with custom text function:
        idx = MathBM25(corpus, text_fn=lambda ex: ex["problem"])

        # From HF Dataset:
        idx = MathBM25(dataset, text_fn=lambda ex: ex["problem"])

        # With stable doc IDs:
        idx = MathBM25(texts, doc_ids=[100, 200, 300])

        results = idx.query("find \\frac", k=3)  # [(score, doc_id), ...]

        # Persistence:
        idx.save("index_dir")
        idx = MathBM25.load("index_dir")
    """

    def __init__(self, documents, *, doc_ids=None, text_fn=None):
        """Build a BM25 index.

        Args:
            documents: list[str], list[dict], or HF Dataset.
            doc_ids: optional list of stable IDs (ints). If None, uses 0..n-1.
            text_fn: callable to extract text from each element. Required if
                     documents are dicts/Dataset rows. Ignored for list[str].
        """
        # Extract texts
        if text_fn is not None:
            texts = [text_fn(d) for d in documents]
        else:
            texts = list(documents)

        self.n = len(texts)
        self.doc_ids = doc_ids if doc_ids is not None else list(range(self.n))

        tokenized = [math_tokenize(normalize(t)) for t in texts]
        self._toks = _make_tokenized(tokenized)
        self._bm25 = bm25s.BM25()
        self._bm25.index(self._toks)

    def query(self, text: str, k: int = 3) -> list[tuple[float, int]]:
        """Return [(score, doc_id), ...] sorted by score descending.

        Returns stable doc_ids (not internal indices). Empty list if query
        has no known tokens.
        """
        q_toks = math_tokenize(normalize(text))
        vocab = self._toks.vocab
        q_ids = [vocab[t] for t in q_toks if t in vocab]
        if not q_ids:
            return []
        qt = Tokenized(ids=[q_ids], vocab=vocab)
        doc_indices, scores = self._bm25.retrieve(qt, k=min(k, self.n))
        return [
            (float(scores[0, r]), self.doc_ids[int(doc_indices[0, r])])
            for r in range(scores.shape[1])
        ]

    def save(self, path):
        """Save index to directory (bm25s index + metadata)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._bm25.save(path / "bm25s", corpus=None)
        meta = {
            "n": self.n,
            "doc_ids": self.doc_ids,
            "vocab": {k: int(v) for k, v in self._toks.vocab.items()},
        }
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path):
        """Load a saved index."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        obj = cls.__new__(cls)
        obj.n = meta["n"]
        obj.doc_ids = meta["doc_ids"]
        obj._bm25 = bm25s.BM25.load(path / "bm25s", load_corpus=False)
        obj._toks = Tokenized(ids=[], vocab=meta["vocab"])
        return obj
