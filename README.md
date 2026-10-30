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

## Results

<!-- RESULTS_START -->
Not run yet. Stage 5 of the notebook fills this table.
<!-- RESULTS_END -->
