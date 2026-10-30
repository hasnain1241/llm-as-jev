# llm-as-jev

Colab-friendly reproduction of "LLM-as-Jev" for `Qwen/Qwen3-0.6B` (full fine-tune, no LoRA).
A causal LLM becomes a decision model that outputs a probability distribution over numbered options.

- Code and notebook: this repo (never data or weights)
- Processed data, reference-stat caches, eval results: https://huggingface.co/datasets/Hasnain1241/llm-as-jev-dataset
- Checkpoints and final model: https://huggingface.co/Hasnain1241/llm-as-jev

## How it works

Prompt: `State: ... / Options: [1] ... [K] / Select one option...`, chat template with thinking disabled, then the prefill `Best answer: [`.
Candidate k is the suffix `k]`. Score s_k = sum of suffix token log-probs, P(k) = softmax(s).
Scoring caches the prompt KV and teacher-forces equal-length suffixes in parallel (works up to K = 151).

Loss: `L = L_tree + lambda * (L_mass + L_out + L_pos)`, lambda = 1, with anchors to the frozen base model
using top-64 truncation plus a tail bucket. The base statistics are computed once (stage 3) and cached, so
training holds one model only. Off-path anchor nodes are evaluated as extra rows over the prompt
(plain sequences, no custom attention masks), so training costs about 4x the tokens of a single pass.

## Run (Colab)

Open `notebooks/colab_run.ipynb` and run top to bottom. Set the Colab secret `my_hf_token`.
Stages: 1 smoke tests, 2 data + training-free baseline, 3 reference cache, 4 fine-tune, 5 eval + push.
Each stage checks the HF repos first and skips or resumes. Training saves to Drive (exact resume) and
pushes `checkpoint-latest` to the model repo (weights only; optimizer moments reset if resumed from HF).
Compute-unit figures in the notebook are rough estimates from assumed per-GPU rates.

## Data (Reason-9k)

1,000 examples from each of: CLINC150 (plus, 151 options), MASSIVE en-US, Bitext customer support,
ReClor, LogiQA 2.0, CosmosQA (about 16% "none of the above" gold), PIQA, aNLI, ProofWriter.
Option order is shuffled once per example. Examples sharing a 13-word n-gram with any eval item are
dropped; examples over 8192 tokens are dropped, never truncated. The dataset IDs actually used (and any
substitution, for example if ReClor is gated) are recorded in `processed/dataset_sources.json` in the dataset repo.

## Try it

The last section of the notebook loads the base and fine-tuned models from the Hub and prints each model's
probability distribution for any state and options you type (`llm_as_jev/demo.py`). Example in Python:

```python
from llm_as_jev import demo
demo.show(models, tok, "I lost my card", ["card arrival", "lost or stolen card", "exchange rate"])
```

## Scope of this reproduction

Implemented: the prompt/answer interface with trie scoring, the tree-restricted likelihood with mass, out and
position anchors (top-64 truncation, cached base statistics), Reason-9k construction with 13-gram decontamination,
and the evaluation suite (accuracy, ECE, Brier, text-mode check) for Qwen3-0.6B only.
Not done: the 4B model, JevBench, multiple seeds, and any numeric comparison against the paper's reported figures.
The results below show the method's qualitative effect (much better calibration, large in-domain gain, small
external accuracy gain) at this scale.

## Results

<!-- RESULTS_START -->
| Metric | Training-free | Fine-tuned |
|---|---|---|
| External macro accuracy (6 sets) | 0.5531 | 0.5729 |
| External macro ECE | 0.2624 | 0.1178 |
| External macro Brier | 0.6523 | 0.5303 |
| Banking77 accuracy | 0.1847 | 0.6026 |
| Banking77 ECE | 0.4597 | 0.1201 |
| Text-mode parse failures | 0 | 0 |
| Text-mode samples | 350 | 350 |

External sets: BoolQ 500, MMLU 1200, MMLU-Pro 800, ARC-C 600, WinoGrande 500, SciQ 400 (seeded subsamples).
Single run, one seed. Per-set numbers are in `results/` of the dataset repo. MMLU-Pro moved from 0.219 to 0.208,
which is within sampling noise for n = 800. Dataset sources used: ReClor came from `metaeval/reclor` (no substitution needed).
<!-- RESULTS_END -->
