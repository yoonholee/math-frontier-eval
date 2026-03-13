# math-frontier-eval

Benchmark for evaluating retrieval-augmented math reasoning across multiple LLMs.

Retrievers are evolved on a held-out validation set and evaluated here on four test benchmarks. Prompts are pre-computed and stored on HuggingFace — no corpus or retriever code runs at eval time.

## Setup

```bash
uv sync
```

Requires access to [`yoonholee/math-frontier-prompts`](https://huggingface.co/datasets/yoonholee/math-frontier-prompts) (private). Log in with `huggingface-cli login` or set `HF_TOKEN`.

For local vLLM endpoints, set `LOCAL_BASE_URL` (default: `http://iris-hgx-2:30000/v1`).

## Usage

```bash
# Run all retrievers on all datasets
uv run benchmark.py run

# Single retriever
uv run benchmark.py run --retriever no_memory

# Single dataset
uv run benchmark.py run --dataset imo_answerbench

# Specific model
uv run benchmark.py run --model openai/gpt-4o

# Smoke test (1 problem, 1 sample, tiny generation)
uv run benchmark.py run --debug
```

Results are saved to `results/<model>/<retriever>/n<k>.json` and a summary table is printed at the end.

## Models

| Alias | Model |
|-------|-------|
| `gpt20b` | `local/openai/gpt-oss-20b` (vLLM) |
| `gpt120b` | `local/openai/gpt-oss-120b` (vLLM) |
| — | `openai/gpt-4o`, `openai/o3-mini`, ... |
| — | `anthropic/claude-opus-4-6`, ... |

Model prefix determines backend: `local/openai/` → vLLM, `openai/` → OpenAI API, `anthropic/` → Anthropic API.

## Retrievers

**Baselines**
- `no_memory` — problem only, no examples
- `bm25_retrieval` — BM25 retrieval from the math corpus
- `random_fewshot` — 3 random examples

**Frontier** (evolved on val set, evaluated here)
- `evo_geo_solution_indexed` — overall leader
- `evo_proof_split_or_max_diversity`
- `evo_geo_proof_curated_index`
- `evo_openmath_geo_proof_branch`
- `evo_domain_conditional_secondary`
- `evo_deepmath_hard_augment` — geometry champion
- `evo_proof_answer_split` — number theory champion
- `evo_combined_routing_diversity` — algebra champion
- `evo_algebra_hard_fusion` — combinatorics champion

## Datasets

| Dataset | Problems | Type |
|---------|----------|------|
| CMIMC | 40 | Answer (competition) |
| USAMO | 6 | Proof (olympiad) |
| IMO-AnswerBench | 400 | Answer (100 × 4 domains) |
| IMO-ProofBench | 60 | Proof (4 levels × 4 domains) |

Answer problems are graded with symbolic equivalence. Proof problems are graded by an LLM judge on a 0–7 scale.

## Prompt Dataset

Pre-computed prompts live at [`yoonholee/math-frontier-prompts`](https://huggingface.co/datasets/yoonholee/math-frontier-prompts), one subset per retriever.

```python
from datasets import load_dataset

# Load one retriever
ds = load_dataset("yoonholee/math-frontier-prompts", "evo_geo_solution_indexed", split="train")

# Fields: retriever, problem_id, dataset, source, category,
#         prompt, groundtruth, solution, grading_guidelines
```

To rebuild after changing retriever logic:

```bash
cd ../math  # monorepo
uv run build_prompt_dataset.py --rebuild --push
```

## Tests

```bash
uv run pytest
```

Snapshot tests verify prompt hashes for 3 retrievers × 3 problems. Full dataset integrity checks (sizes, domain counts, required fields) are included. Tests run in ~5 seconds using the cached parquet.
