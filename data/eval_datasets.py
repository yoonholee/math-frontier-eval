"""Evaluation datasets used by benchmark/evolve workflows.

Supported eval datasets:
    aime             - AIME 2025 (30 problems)
    hmmt_feb         - HMMT Feb 2025 (30 problems)
    hmmt_nov         - HMMT Nov 2025 (30 problems)
    smt              - SMT 2025 (53 problems)
    brumo            - BruMO 2025 (30 problems)
    cmimc            - CMIMC 2025 (40 problems)
    arxivmath_1225   - ArXivMath Dec 2025 (17 problems)
    arxivmath_0126   - ArXivMath Jan 2026 (23 problems)
    usamo            - USAMO 2025 (6 proof problems)
    imo_answerbench  - IMO-AnswerBench v2 (400 problems)
    imo_proofbench   - IMO-ProofBench (60 problems)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from grading import load_imoproofbench

ROOT = Path(__file__).resolve().parent


def _clean_answer(answer: str) -> str:
    """Strip LaTeX \\$ delimiters and trailing punctuation from answers."""
    answer = answer.strip()
    # Remove surrounding $...$ (single pair)
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    elif answer.startswith("$") and answer.endswith("$."):
        answer = answer[1:-2].strip()
    # Remove all remaining $ formatting markers
    return answer.replace("$", "").strip()


def _load_hf_answer_dataset(repo: str) -> list[dict]:
    """Load a MathArena HF dataset with an answer field."""
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    problems = []
    for r in ds:
        pt = r.get("problem_type", None)
        if isinstance(pt, list):
            pt = pt[0] if pt else ""
        source = (pt or "").lower().replace(" ", "_") or "unknown"
        problems.append(
            {
                "problem": r["problem"],
                "groundtruth": _clean_answer(str(r["answer"])),
                "source": source,
            }
        )
    return problems


def load_usamo() -> list[dict]:
    """Load USAMO 2025 proof tasks with grading rubrics."""
    from datasets import load_dataset

    ds = load_dataset("MathArena/usamo_2025", split="train")
    problems = []
    for r in ds:
        guidelines = "\n".join(
            f"- ({item['points']}pt) {item['desc']}" for item in r["grading_scheme"]
        )
        problems.append(
            {
                "problem": r["problem"].strip(),
                "groundtruth": None,
                "solution": r["sample_solution"],
                "grading_guidelines": guidelines,
                "source": "usamo_2025",
            }
        )
    return problems


def load_imo_answerbench() -> list[dict]:
    """Load IMO-AnswerBench v2 (short-answer olympiad problems)."""
    csv_path = ROOT / "imobench" / "answerbench_v2.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    problems = []
    for r in rows:
        cat = r["Category"].lower().replace(" ", "_")
        if cat == "functional_equation":
            cat = "algebra"
        answer = _clean_answer(r["Short Answer"])
        problems.append(
            {
                "problem": r["Problem"].strip(),
                "groundtruth": answer,
                "source": cat,
                "problem_id": r["Problem ID"],
                "subcategory": r.get("Subcategory", ""),
            }
        )
    return problems


def load_imo_proofbench() -> list[dict]:
    """Load IMO-ProofBench (all problems, answer + proof variants)."""
    rows = load_imoproofbench()
    problems = []
    for r in rows:
        raw_answer = (r.get("answer") or "").strip()
        cat = r.get("category", "unknown").lower().replace(" ", "_")
        level = r.get("level", "unknown")
        if raw_answer:
            problems.append(
                {
                    "problem": r["problem"].strip(),
                    "groundtruth": _clean_answer(raw_answer),
                    "source": level,
                    "problem_id": r.get("question_id", ""),
                    "category": cat,
                }
            )
        else:
            problems.append(
                {
                    "problem": r["problem"].strip(),
                    "groundtruth": None,
                    "solution": r.get("solution", ""),
                    "grading_guidelines": r.get("grading_guidelines", ""),
                    "source": level,
                    "problem_id": r.get("question_id", ""),
                    "category": cat,
                }
            )
    return problems


EVAL_DATASETS: dict[str, Callable[[], list[dict]]] = {
    "aime": lambda: _load_hf_answer_dataset("MathArena/aime_2025"),
    "hmmt_feb": lambda: _load_hf_answer_dataset("MathArena/hmmt_feb_2025"),
    "hmmt_nov": lambda: _load_hf_answer_dataset("MathArena/hmmt_nov_2025"),
    "smt": lambda: _load_hf_answer_dataset("MathArena/smt_2025"),
    "brumo": lambda: _load_hf_answer_dataset("MathArena/brumo_2025"),
    "cmimc": lambda: _load_hf_answer_dataset("MathArena/cmimc_2025"),
    "arxivmath_1225": lambda: _load_hf_answer_dataset("MathArena/arxivmath-1225"),
    "arxivmath_0126": lambda: _load_hf_answer_dataset("MathArena/arxivmath-0126"),
    "usamo": load_usamo,
    "imo_answerbench": load_imo_answerbench,
    "imo_proofbench": load_imo_proofbench,
}

TEST_DATASET_NAMES = [
    "aime",
    "hmmt_feb",
    "hmmt_nov",
    "smt",
    "brumo",
    "cmimc",
    "arxivmath_1225",
    "arxivmath_0126",
]


def load_eval_problems(dataset: str) -> list[dict]:
    """Load eval problems by dataset name."""
    if dataset not in EVAL_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: {list(EVAL_DATASETS)}"
        )
    return EVAL_DATASETS[dataset]()


def load_test_problems() -> list[dict]:
    """Load all answer-style benchmark problems used for retriever prefiltering."""
    problems = []
    for name in TEST_DATASET_NAMES:
        problems.extend(EVAL_DATASETS[name]())
    return problems
