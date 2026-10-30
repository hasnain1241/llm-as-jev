# llm-as-jev

Colab-friendly reproduction of "LLM-as-Jev" for `Qwen/Qwen3-0.6B` (full fine-tune, no LoRA).
A causal LLM becomes a decision model that outputs a probability distribution over numbered options.

- Code and notebook: this repo (never data or weights)
- Processed data, reference-stat caches, eval results: https://huggingface.co/datasets/Hasnain1241/llm-as-jev-dataset
- Checkpoints and final model: https://huggingface.co/Hasnain1241/llm-as-jev

## Summary of findings

Single run, one seed, 0.6B model only.

- Calibration improved clearly and consistently. On the six external sets macro ECE fell from 0.262 to 0.118 and Brier from 0.652 to 0.530. On the 231 public JevBench items ECE fell from 0.299 to 0.170 and Brier from 0.690 to 0.569.
- In-domain intent classification improved a lot: Banking77 accuracy 0.185 to 0.603 (77 options, not in the training mix).
- External accuracy changed little (macro 0.553 to 0.573). JevBench accuracy moved by a few items either way (0.541 to 0.554), which is within sampling noise.
- The model is not a strong reasoner at this size: the base model is near chance on the JevBench hard tier, and both models are near chance on its yes/no items.
- Text mode works: after the `Best answer: [` prefill, greedy generation parsed as a valid id in all 350 checked items, for both models.

## How it works

Prompt: `State: ... / Options: [1] ... [K] / Select one option...`, chat template with thinking disabled, then the prefill `Best answer: [`.
Candidate k is the suffix `k]`. Score s_k = sum of suffix token log-probs, P(k) = softmax(s).
Scoring caches the prompt KV and teacher-forces equal-length suffixes in parallel (works up to K = 151).

Loss: `L = L_tree + lambda * (L_mass + L_out + L_pos)`, lambda = 1, with anchors to the frozen base model
using top-64 truncation plus a tail bucket. The base statistics are computed once (stage 3) and cached, so
training holds one model only. Off-path anchor nodes are evaluated as extra rows over the prompt
(plain sequences, no custom attention masks), so training costs about 4x the tokens of a single pass.

Training: AdamW, lr 5e-6 with 10% warmup and linear decay to 10%, weight decay 0.01, grad clip 1.0, global batch 32
(gradient accumulation), gradient checkpointing, fp32 master weights with bf16 (A100/L4) or fp16 (T4) autocast, seed 20260922.
A health check halts training if the root legal log-mass drops more than 0.5 nats from the base model.

## Data (Reason-9k)

1,000 examples from each of: CLINC150 (plus, 151 options), MASSIVE en-US, Bitext customer support,
ReClor, LogiQA 2.0, CosmosQA (about 16% "none of the above" gold), PIQA, aNLI, ProofWriter.
Option order is shuffled once per example. Examples sharing a 13-word n-gram with any eval item are
dropped; examples over 8192 tokens are dropped, never truncated.

Sources actually used (from `processed/dataset_sources.json`, no substitutions were needed): `clinc/clinc_oos` (plus),
`AmazonScience/massive` (en-US), `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, `metaeval/reclor`,
`baber/logiqa2`, `allenai/cosmos_qa`, `ybisk/piqa`, `allenai/art`, `tasksource/proofwriter`.

## Results

External evaluation: BoolQ 500, MMLU 1200, MMLU-Pro 800, ARC-C 600, WinoGrande 500, SciQ 400 (seeded subsamples,
all options kept), plus Banking77 test (3,080). ECE uses 15 bins and no temperature scaling.

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
<!-- RESULTS_END -->

Per set (accuracy, training-free to fine-tuned):

| Set | Accuracy | ECE |
|---|---|---|
| BoolQ | 0.692 to 0.732 | 0.171 to 0.081 |
| MMLU | 0.473 to 0.485 | 0.315 to 0.085 |
| MMLU-Pro | 0.219 to 0.208 | 0.432 to 0.232 |
| ARC-C | 0.538 to 0.552 | 0.317 to 0.106 |
| WinoGrande | 0.472 to 0.516 | 0.295 to 0.161 |
| SciQ | 0.925 to 0.945 | 0.045 to 0.043 |
| Banking77 | 0.185 to 0.603 | 0.460 to 0.120 |

The MMLU-Pro change is within sampling noise for n = 800. The text-mode check generated 32 tokens after the prefill for
50 items per set; the generated id matched the scored argmax in 340 of 350 cases (training-free) and 338 of 350 (fine-tuned).

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

## Comparison with the paper (JevBench public items)

Paper figures were supplied by the repo owner and have not been independently verified here. I have not confirmed
the paper's model size or exact protocol for these columns, so check they match before reading this as like for like.
"Yours" is this repo (Qwen3-0.6B, one run, one seed).

| Metric | Paper base | Yours base | Paper FT (Reason-9k) | Yours FT |
|---|---|---|---|---|
| Overall accuracy | 56.7% | 54.1% | 55.0% | 55.4% |
| Hard (of 111) | 37 (33.3%) | 33 (29.7%) | 36 (32.4%) | 39 (35.1%) |
| Long items correct (of 37) | 7 | 6 | 9 | 10 |
| ECE | 0.278 | 0.299 | 0.118 | 0.170 |
| "Yes" predictions (of 74 yes/no items) | 34 | 37 | 41 | 44 |

What lines up: ECE falls after fine-tuning in both (paper 0.278 to 0.118, here 0.299 to 0.170), long-item correct count
rises (7 to 9, 6 to 10), and "yes" predictions rise by 7 in both (34 to 41, 37 to 44). What differs: the ECE reduction here is
smaller, and overall accuracy fell slightly in the paper (56.7% to 55.0%) but rose slightly here (54.1% to 55.4%). With 231
items, differences of a few items are within noise, so the accuracy and hard-tier gaps should not be read as real differences.

## Known limitations

- One seed, one run, no confidence intervals. Small accuracy differences should not be read as real.
- Only Qwen3-0.6B. The 4B model was not run. The paper comparison above uses figures supplied by the repo owner and was not independently verified.
- Hand-tried examples (not a measured result) showed the fine-tuned model answering "Unknown" on short logic questions where the
  premises settle the answer, for example "Anne is kind. Kind people are happy. Is Anne happy?". The cause is not confirmed. It may
  be a prior from the ProofWriter part of the training mix, and the label distribution of that slice has not been checked.
- On JevBench the fine-tuned model leans toward "yes" on yes/no items (44 predictions against 35 gold yes). The paper's table shows the same direction (34 to 41).
- The JevBench conversion (label plus rubric description per option, instructions prepended) is my design choice and affects the scores.
- Option-order sensitivity was tried on a single question only (no flips). It has not been measured at scale.

## Run (Colab)

Open `notebooks/colab_run.ipynb` and run top to bottom. Set the Colab secret `my_hf_token`. A GPU is required
(L4 is a good default; A100 is faster; T4 works with fp16).
Stages: 1 smoke tests, 2 data + training-free baseline, 3 reference cache, 4 fine-tune, 5 eval + push.
Then the optional sections: demo (type your own state and options), JevBench public items, and a final test run.
Each stage checks the HF repos first and skips or resumes. Training saves to Drive (exact resume) and
pushes `checkpoint-latest` to the model repo (weights only; optimizer moments reset if resumed from HF).
After 10 steps training prints a projected compute-unit total and stops if it exceeds 15.
Compute-unit figures in the notebook are rough estimates from assumed per-GPU rates.
Stage 5 edits `README.md` inside the Colab clone only; if a later `git pull` complains, run `git checkout -- README.md` first.

## Try it

```python
from llm_as_jev import demo
demo.show(models, tok, "I lost my card", ["card arrival", "lost or stolen card", "exchange rate"])
demo.show(models, tok, query, banking77_labels, top=5)   # many options: show the top 5
```

`models` is a dict of name to model (base and fine-tuned, loaded from the Hub); the notebook's demo section sets it up.

## Repo layout

| Path | Purpose |
|---|---|
| `llm_as_jev/prompt.py` | prompt template, suffix trie, candidate suffix tokenization |
| `llm_as_jev/scoring.py` | training-free scoring (KV cache, parallel suffixes), text-mode generation |
| `llm_as_jev/loss.py` | tree loss, mass/out/position anchors, top-64 reference stats |
| `llm_as_jev/train.py` | reference cache build, training loop, checkpoints, budget and health checks |
| `llm_as_jev/data.py` | Reason-9k and eval-set builders, 13-gram decontamination |
| `llm_as_jev/eval.py` | accuracy, ECE, Brier, text-mode check |
| `llm_as_jev/jevbench.py` | JevBench public-item converter and evaluation |
| `llm_as_jev/hub.py`, `auth.py` | Hugging Face pushes and pulls, token handling (never printed) |
| `llm_as_jev/demo.py` | interactive distributions for base vs fine-tuned |
| `configs/qwen3_0p6b.yaml` | all hyperparameters |
| `tests/` | trie, scoring vs naive, loss, token safety, JevBench converter |
| `notebooks/colab_run.ipynb` | the full pipeline |

Artifacts in the dataset repo: `processed/reason9k.jsonl`, `processed/eval_sets.json`, `processed/dataset_sources.json`,
`cache/ref_stats_top64.pt`, `results/baseline_training_free.json`, `results/finetuned.json`, `results/jevbench.json`.
