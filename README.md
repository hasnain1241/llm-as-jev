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

## JevBench public items

Evaluation only: the 231 public items of JevBench v1.4.2.2 (easy 48, standard 72, hard 111) are scored with the same trie
scorer and never used for training or checkpoint selection. Conversion (`llm_as_jev/jevbench.py`): the question's
instructions are prepended to the item state, options keep the canonical `labels` order, and each option is written as
`label: rubric description` (yes/no items use the true/false descriptions; score items use the level descriptions).
This is not the official JevBench score, which also includes calibration, speed and cost axes and held-out items,
so the numbers below do not compare to its leaderboard.

<!-- JEVBENCH_START -->
| Metric | Base | Fine-tuned |
|---|---|---|
| Overall accuracy (231 items) | 0.541 | 0.554 |
| Easy accuracy (48) | 0.938 | 0.979 |
| Standard accuracy (72) | 0.653 | 0.583 |
| Hard accuracy (111) | 0.297 | 0.351 |
| Accuracy, prompts over 1500 tokens (37) | 0.162 | 0.270 |
| ECE (15 bins) | 0.299 | 0.170 |
| Brier | 0.690 | 0.569 |
| Yes predictions on yes/no items (74) | 37 | 44 |
| Chance accuracy (uniform guess) | 0.318 | 0.318 |

Benchmark commit `fcc6927` (tag v1.4.2.2). Gold "yes" count on the 74 yes/no items is 35. No Reason-9k training item shares a
13-word n-gram with these items.

How to read it: with 231 items the sampling error on accuracy is about 3 points, so the accuracy differences (3 more items
correct overall, 5 fewer on the standard tier, 6 more on the hard tier) are not distinguishable from noise. The calibration
gain (ECE 0.30 to 0.17, Brier 0.69 to 0.57) is the consistent effect, and matches the external-set results. Both models are
near chance on the yes/no items (57% and 55%), and the fine-tuned model leans toward "yes" (44 predictions against 35 gold).
The hard tier is near chance for the base model, so a 0.6B model is not solving these items.
<!-- JEVBENCH_END -->

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
