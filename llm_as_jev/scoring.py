"""Training-free scoring: s_k = sum log p(suffix_k | prompt), P = softmax(s)."""
import copy
import re

import torch
import torch.nn.functional as F


def _expand_cache(past, n):
    cache = copy.deepcopy(past)
    cache.batch_repeat_interleave(n)
    return cache


@torch.no_grad()
def score_candidates(model, enc, chunk=32):
    """Log-score of every candidate. One prompt forward, then suffixes[:-1] are
    teacher-forced in parallel against the cached prompt, batched by equal length."""
    dev = next(model.parameters()).device
    prompt = torch.tensor([enc.prompt_ids], device=dev)
    out = model(prompt, use_cache=True)
    past = out.past_key_values
    root_lp = F.log_softmax(out.logits[0, -1].float(), -1)
    k_total = len(enc.suffixes)
    scores = torch.zeros(k_total, device=dev)
    groups = {}
    for i, s in enumerate(enc.suffixes):
        scores[i] += root_lp[s[0]]
        if len(s) > 1:
            groups.setdefault(len(s), []).append(i)
    for length, idxs in groups.items():
        for c in range(0, len(idxs), chunk):
            part = idxs[c : c + chunk]
            inp = torch.tensor([enc.suffixes[i][:-1] for i in part], device=dev)
            tgt = torch.tensor([enc.suffixes[i][1:] for i in part], device=dev)
            cache = _expand_cache(past, len(part))
            logits = model(inp, past_key_values=cache, use_cache=True).logits.float()
            lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores[torch.tensor(part, device=dev)] += lp.sum(-1)
    return scores


def probs_from_scores(scores):
    return torch.softmax(scores, -1)


@torch.no_grad()
def naive_scores(model, enc):
    """Reference implementation: one full forward per candidate (for tests)."""
    dev = next(model.parameters()).device
    n = len(enc.prompt_ids)
    res = []
    for s in enc.suffixes:
        ids = torch.tensor([enc.prompt_ids + list(s)], device=dev)
        lp = F.log_softmax(model(ids).logits[0].float(), -1)
        res.append(sum(lp[n - 1 + j, s[j]] for j in range(len(s))))
    return torch.stack(res)


_ID_RE = re.compile(r"^\s*(\d+)\s*\]")


@torch.no_grad()
def generate_text_mode(model, tokenizer, enc, max_new=32):
    """Greedy continuation after the prefill. Returns (text, parsed id or None)."""
    dev = next(model.parameters()).device
    ids = torch.tensor([enc.prompt_ids], device=dev)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
    m = _ID_RE.match(text)
    return text, (int(m.group(1)) if m else None)
