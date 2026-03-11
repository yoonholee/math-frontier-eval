"""Snapshot tests for the pre-built prompt dataset.

These tests verify that:
1. The prompt dataset loads correctly and contains the expected prompts.
2. Prompt hashes match values recorded when the dataset was built (2026-03-11).
3. Dataset sizes and domain counts are correct.

The prompt dataset is loaded from prompts_dataset.parquet (local cache)
or from HuggingFace (yoonholee/math-frontier-prompts) if not cached.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import _problem_id, load_prompt_dataset
from data.eval_datasets import load_eval_problems

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def prompt_ds():
    return load_prompt_dataset()


@pytest.fixture(scope="session")
def sample_problems():
    """Three fixed problems drawn from the test datasets."""
    return {
        "cmimc_0": load_eval_problems("cmimc")[0],
        "imo_answerbench_7": load_eval_problems("imo_answerbench")[7],
        "imo_proofbench_0": load_eval_problems("imo_proofbench")[0],
    }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _get_prompt(prompt_ds, retriever_name, problem_dict):
    pid = _problem_id(problem_dict)
    return prompt_ds.get((retriever_name, pid))


# ---------------------------------------------------------------------------
# Snapshot data (captured 2026-03-10)
#
# Problem texts (first 80 chars):
#   cmimc_0:          "Four runners are preparing to begin a $1$-mile race..."
#   imo_answerbench_7: "Let $x_0, x_1, \ldots$ be a sequence of real numbers..."
#   imo_proofbench_0:  "Determine all functions $f: \mathbb{Z} \rightarrow..."
# ---------------------------------------------------------------------------

SNAPSHOTS = {
    "no_memory": {
        "cmimc_0":          {"n_examples": 0, "prompt_len": 491,  "sha": "5cabd5a66e1cff2d"},
        "imo_answerbench_7": {"n_examples": 0, "prompt_len": 310,  "sha": "99165363aa7a2952"},
        "imo_proofbench_0":  {"n_examples": 0, "prompt_len": 174,  "sha": "830b83b4b70760af"},
    },
    "evo_geo_solution_indexed": {
        "cmimc_0":          {"n_examples": 3, "prompt_len": 3764, "sha": "1350f287049c80c6"},
        "imo_answerbench_7": {"n_examples": 3, "prompt_len": 2658, "sha": "e67430124f7fa4ab"},
        "imo_proofbench_0":  {"n_examples": 3, "prompt_len": 2268, "sha": "a571c6d22ecb789d"},
    },
    "evo_combined_routing_diversity": {
        "cmimc_0":          {"n_examples": 3, "prompt_len": 3764, "sha": "1350f287049c80c6"},
        "imo_answerbench_7": {"n_examples": 3, "prompt_len": 2658, "sha": "e67430124f7fa4ab"},
        "imo_proofbench_0":  {"n_examples": 2, "prompt_len": 1890, "sha": "585ee8bad9c7c5dc"},
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("retriever_name", list(SNAPSHOTS))
def test_prompt_snapshot(retriever_name, sample_problems, prompt_ds):
    """Prompts in dataset must match hashes recorded at build time (2026-03-11)."""
    for label, expected in SNAPSHOTS[retriever_name].items():
        prompt = _get_prompt(prompt_ds, retriever_name, sample_problems[label])
        assert prompt is not None, f"No prompt found for {retriever_name}/{label}"
        n_ex = prompt.count("Example ")
        sha = _sha(prompt)
        assert n_ex == expected["n_examples"], (
            f"{retriever_name}/{label}: expected {expected['n_examples']} examples, got {n_ex}"
        )
        assert len(prompt) == expected["prompt_len"], (
            f"{retriever_name}/{label}: expected prompt_len={expected['prompt_len']}, got {len(prompt)}"
        )
        assert sha == expected["sha"], (
            f"{retriever_name}/{label}: prompt hash changed. "
            f"Expected {expected['sha']}, got {sha}. "
            "Rebuild the dataset if retriever logic changed intentionally."
        )


def test_no_memory_has_no_examples(sample_problems, prompt_ds):
    """no_memory prompts must never contain injected examples."""
    for label, p in sample_problems.items():
        prompt = _get_prompt(prompt_ds, "no_memory", p)
        assert prompt is not None
        assert "Example " not in prompt, f"no_memory injected examples for {label}"


def test_frontier_retrievers_inject_examples(sample_problems, prompt_ds):
    """All frontier retrievers must have at least one example for each problem."""
    frontier = [
        "evo_geo_solution_indexed",
        "evo_proof_split_or_max_diversity",
        "evo_geo_proof_curated_index",
        "evo_openmath_geo_proof_branch",
        "evo_domain_conditional_secondary",
        "evo_deepmath_hard_augment",
        "evo_proof_answer_split",
        "evo_combined_routing_diversity",
        "evo_algebra_hard_fusion",
    ]
    for ret_name in frontier:
        for label, p in sample_problems.items():
            prompt = _get_prompt(prompt_ds, ret_name, p)
            assert prompt is not None, f"No prompt for {ret_name}/{label}"
            assert "Example " in prompt, (
                f"{ret_name} has no examples for {label}"
            )


def test_datasets_correct_sizes():
    """Test datasets contain expected problem counts."""
    expected = {
        "cmimc": 40,
        "usamo": 6,
        "imo_answerbench": 400,
        "imo_proofbench": 60,
    }
    for ds, n in expected.items():
        problems = load_eval_problems(ds)
        assert len(problems) == n, f"{ds}: expected {n} problems, got {len(problems)}"


def test_answerbench_domain_counts():
    """IMO-AnswerBench must have exactly 100 problems per domain after merging functional_equation."""
    from collections import Counter
    problems = load_eval_problems("imo_answerbench")
    counts = Counter(p["source"] for p in problems)
    for domain in ["algebra", "combinatorics", "geometry", "number_theory"]:
        assert counts[domain] == 100, (
            f"Expected 100 {domain} problems, got {counts[domain]}"
        )
    assert "functional_equation" not in counts, (
        "functional_equation should be merged into algebra"
    )


def test_proofbench_has_level_and_category():
    """IMO-ProofBench problems must have level and category fields."""
    problems = load_eval_problems("imo_proofbench")
    levels = {"IMO-easy", "IMO-medium", "IMO-hard", "pre-IMO"}
    cats = {"algebra", "combinatorics", "geometry", "number_theory"}
    for p in problems:
        assert p.get("source") in levels, f"Bad level: {p.get('source')}"
        assert p.get("category") in cats, f"Bad category: {p.get('category')}"
