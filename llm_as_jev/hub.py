"""Hugging Face Hub helpers: datasets, caches, results JSON, checkpoints, final model."""
import json
import os

from .auth import get_hf_token

DATASET_REPO = "Hasnain1241/llm-as-jev-dataset"
MODEL_REPO = "Hasnain1241/llm-as-jev"
CKPT_DIR = "checkpoint-latest"


def api():
    from huggingface_hub import HfApi

    return HfApi(token=get_hf_token())


def ensure_repos():
    a = api()
    a.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
    a.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)


def _has(repo, repo_type, path):
    return api().file_exists(repo_id=repo, filename=path, repo_type=repo_type)


def dataset_has(path):
    return _has(DATASET_REPO, "dataset", path)


def model_has(path):
    return _has(MODEL_REPO, "model", path)


def push_dataset_file(local_path, path_in_repo, message=None):
    api().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=message or f"Add {path_in_repo}",
    )


def pull_dataset_file(path_in_repo, local_dir="data"):
    """Download a file from the dataset repo; returns the local path or None if absent."""
    from huggingface_hub import hf_hub_download

    if not dataset_has(path_in_repo):
        return None
    return hf_hub_download(
        repo_id=DATASET_REPO,
        filename=path_in_repo,
        repo_type="dataset",
        local_dir=local_dir,
        token=get_hf_token(),
    )


def push_json(obj, path_in_repo, tmp_dir="outputs"):
    os.makedirs(tmp_dir, exist_ok=True)
    local = os.path.join(tmp_dir, os.path.basename(path_in_repo))
    with open(local, "w") as f:
        json.dump(obj, f, indent=2)
    push_dataset_file(local, path_in_repo)


def pull_json(path_in_repo, local_dir="data"):
    p = pull_dataset_file(path_in_repo, local_dir)
    if p is None:
        return None
    with open(p) as f:
        return json.load(f)


def push_model_folder(folder, path_in_repo="", message="Upload model", replace=False):
    kwargs = {"delete_patterns": "*"} if replace and path_in_repo else {}
    api().upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo or None,
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message=message,
        **kwargs,
    )


def push_checkpoint(folder, step):
    push_model_folder(folder, CKPT_DIR, f"Checkpoint at step {step}", replace=True)


def pull_checkpoint(local_dir):
    """Download checkpoint-latest from the model repo; returns its local folder or None."""
    from huggingface_hub import snapshot_download

    if not model_has(f"{CKPT_DIR}/trainer_state.json"):
        return None
    root = snapshot_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        allow_patterns=[f"{CKPT_DIR}/*"],
        local_dir=local_dir,
        token=get_hf_token(),
    )
    return os.path.join(root, CKPT_DIR)


def build_model_card(results_table_md):
    return f"""---
base_model: Qwen/Qwen3-0.6B
library_name: transformers
license: apache-2.0
tags: [decision-model, multiple-choice, llm-as-jev]
---

# LLM-as-Jev (Qwen3-0.6B, full fine-tune)

Qwen3-0.6B fine-tuned to output a probability distribution over numbered options.
The prompt lists options as `[1] ... [K]` and the reply is prefilled with `Best answer: [`.
Candidate k is scored as the log-probability of the suffix `k]`; probabilities are the softmax of those scores.

Training: Reason-9k (9 sets x 1,000 examples), tree-restricted likelihood plus
mass/out/position anchors to the frozen base model (top-64 truncation), AdamW, lr 5e-6, global batch 32.
Code: https://github.com/hasnain1241/llm-as-jev

## Results

{results_table_md}
"""


def push_final_model(folder, card_md):
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(card_md)
    push_model_folder(folder, "", "Final fine-tuned model", replace=False)
