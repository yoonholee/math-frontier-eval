"""Snapshot tests for frontier retriever prompt construction.

These tests characterize the retrieval behavior of each frontier retriever
on fixed input strings. They do NOT call an LLM — only verify that
build_prompt() returns the expected structure (number of examples, prompt
hash). If a retriever change breaks a snapshot, update the expected values
intentionally after verifying the new behavior is correct.

Corpus loads from HuggingFace (math-corpus-combined). Requires network
access or a pre-populated HF_HOME cache.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import load_retriever
from data.eval_datasets import load_eval_problems, load_test_problems

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_problems():
    return load_test_problems()


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


def _prompt(retriever_name, problem_dict, test_problems):
    r = load_retriever(retriever_name, test_problems=test_problems)
    return r.build_prompt(problem_dict["problem"])


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
def test_prompt_snapshot(retriever_name, sample_problems, test_problems):
    """Each retriever must produce exact prompt lengths and hashes for fixed inputs."""
    for label, expected in SNAPSHOTS[retriever_name].items():
        prompt = _prompt(retriever_name, sample_problems[label], test_problems)
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
            "Update snapshot if change is intentional."
        )


def test_no_memory_has_no_examples(sample_problems, test_problems):
    """no_memory retriever must never inject any examples."""
    r = load_retriever("no_memory", test_problems=test_problems)
    for label, p in sample_problems.items():
        prompt = r.build_prompt(p["problem"])
        assert "Example " not in prompt, f"no_memory injected examples for {label}"


def test_frontier_retrievers_inject_examples(sample_problems, test_problems):
    """All frontier retrievers must inject at least one example for each problem."""
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
        r = load_retriever(ret_name, test_problems=test_problems)
        for label, p in sample_problems.items():
            prompt = r.build_prompt(p["problem"])
            assert "Example " in prompt, (
                f"{ret_name} produced no examples for {label}"
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
