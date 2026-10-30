"""Evaluation: accuracy, ECE (15 bins, no temperature scaling), Brier, and a text-mode check."""
import random
import time

import torch

from .prompt import SuffixCache, encode_item
from .scoring import generate_text_mode, probs_from_scores, score_candidates

EXTERNAL = ["boolq", "mmlu", "mmlu_pro", "arc_c", "winogrande", "sciq"]


def calibration_metrics(probs, labels, bins=15):
    """probs: list of 1D float tensors/lists, labels: list[int]."""
    conf, correct, brier = [], [], []
    for p, y in zip(probs, labels):
        p = torch.as_tensor(p, dtype=torch.float64)
        c, a = p.max(0)
        conf.append(c.item())
        correct.append(float(a.item() == y))
        onehot = torch.zeros_like(p)
        onehot[y] = 1.0
        brier.append(((p - onehot) ** 2).sum().item())
    conf_t, cor_t = torch.tensor(conf), torch.tensor(correct)
    ece = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf_t > lo) & (conf_t <= hi) if b > 0 else (conf_t >= lo) & (conf_t <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(cor_t[m].mean().item() - conf_t[m].mean().item())
    return {"acc": cor_t.mean().item(), "ece": ece, "brier": sum(brier) / len(brier), "n": len(labels)}


def score_items(model, tok, items, chunk=32, log=print, name=""):
    cache = SuffixCache()
    probs, encs = [], []
    t0 = time.time()
    for i, it in enumerate(items):
        enc = encode_item(tok, it["state"], it["options"], cache)
        s = score_candidates(model, enc, chunk)
        if not torch.isfinite(s).all():
            raise FloatingPointError(
                "Non-finite scores. On T4 set eval_dtype: float32 in the config and rerun."
            )
        probs.append(probs_from_scores(s).cpu())
        encs.append(enc)
        if (i + 1) % 500 == 0:
            log(f"  {name}: {i + 1}/{len(items)}  {time.time() - t0:.0f}s")
    return probs, encs


def text_mode_check(model, tok, items, probs, encs, n, seed):
    rng = random.Random(seed)
    idx = rng.sample(range(len(items)), min(n, len(items)))
    fail = agree = 0
    for i in idx:
        _, gid = generate_text_mode(model, tok, encs[i])
        k = len(items[i]["options"])
        if gid is None or not (1 <= gid <= k):
            fail += 1
        elif gid - 1 == int(probs[i].argmax()):
            agree += 1
    parsed = len(idx) - fail
    return {"n": len(idx), "parse_failures": fail, "argmax_match": agree, "match_rate_of_parsed": agree / parsed if parsed else None}


def run_eval(model, tok, eval_sets, text_n=50, seed=20260922, log=print, jevbench=None):
    model.eval()
    res, sets = {}, dict(eval_sets)
    if jevbench:
        sets["jevbench"] = jevbench
    tm = {"n": 0, "parse_failures": 0, "argmax_match": 0}
    for name, items in sets.items():
        t0 = time.time()
        probs, encs = score_items(model, tok, items, log=log, name=name)
        m = calibration_metrics(probs, [it["label"] for it in items])
        t = text_mode_check(model, tok, items, probs, encs, text_n, seed)
        m["text_mode"] = t
        for k in tm:
            tm[k] += t[k]
        res[name] = m
        log(f"{name}: acc={m['acc']:.4f} ece={m['ece']:.4f} brier={m['brier']:.4f} "
            f"text_fail={t['parse_failures']}/{t['n']} ({time.time() - t0:.0f}s)")
    ext = [res[k] for k in EXTERNAL if k in res]
    res["external_macro"] = {k: sum(r[k] for r in ext) / len(ext) for k in ("acc", "ece", "brier")}
    res["text_mode_total"] = tm
    return res


def results_table(base, ft):
    def row(label, a, b, fmt="{:.4f}"):
        return f"| {label} | {fmt.format(a)} | {fmt.format(b)} |"

    lines = [
        "| Metric | Training-free | Fine-tuned |",
        "|---|---|---|",
        row("External macro accuracy (6 sets)", base["external_macro"]["acc"], ft["external_macro"]["acc"]),
        row("External macro ECE", base["external_macro"]["ece"], ft["external_macro"]["ece"]),
        row("External macro Brier", base["external_macro"]["brier"], ft["external_macro"]["brier"]),
        row("Banking77 accuracy", base["banking77"]["acc"], ft["banking77"]["acc"]),
        row("Banking77 ECE", base["banking77"]["ece"], ft["banking77"]["ece"]),
        row("Text-mode parse failures", base["text_mode_total"]["parse_failures"], ft["text_mode_total"]["parse_failures"], "{}"),
        row("Text-mode samples", base["text_mode_total"]["n"], ft["text_mode_total"]["n"], "{}"),
    ]
    return "\n".join(lines)
