"""JevBench public-item evaluation (evaluation only, never training or checkpoint selection).

Item schema (datasets/public/{easy,original,hard}.jsonl, v1.4.2.2):
  id, family, group, split, state, labels (list of option names), expected (gold),
  question = {type: choice | noul | score, instructions: str, criteria: dict or list}
    choice: criteria maps label -> description, expected is a label string
    noul:   labels are ["no", "yes"], criteria has "true"/"false" descriptions
    score:  labels are "0".."3"/"4", criteria is a list of level descriptions, expected is an int
"""
import ast
import os
import subprocess

from .data import eval_ngram_set, is_contaminated
from .eval import calibration_metrics, score_items

URL = "https://github.com/fstandhartinger/jevbench"
TAG = "v1.4.2.2"
# report name -> file name (the middle tier is called "original" on disk)
TIERS = {"easy": "easy", "standard": "original", "hard": "hard"}
LONG_TOKENS = 1500


def fetch_jevbench(dest="data/jevbench", tag=TAG):
    """Clone the benchmark (tag first, then main). Returns {"tag", "commit"}. Never runs its code."""
    used = tag
    if not os.path.isdir(os.path.join(dest, ".git")):
        r = subprocess.run(["git", "clone", "--depth", "1", "--branch", tag, URL, dest], capture_output=True)
        if r.returncode != 0:
            used = "main"
            subprocess.run(["git", "clone", "--depth", "1", URL, dest], capture_output=True, check=True)
    commit = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"tag": used, "commit": commit}


def convert_item(row, tier):
    """JevBench row -> our item. Option order is the canonical `labels` order; each option shows the
    label plus its rubric description so the model sees what the label means."""
    q = row["question"]
    if isinstance(q, str):
        q = ast.literal_eval(q)
    labels = [str(l) for l in row["labels"]]
    kind, crit = q["type"], q["criteria"]
    if kind == "choice":
        descs = [crit[l] for l in labels]
        gold = labels.index(str(row["expected"]))
    elif kind == "noul":
        assert set(labels) == {"yes", "no"}, labels
        descs = [crit["true"] if l == "yes" else crit["false"] for l in labels]
        gold = labels.index(str(row["expected"]))
    elif kind == "score":
        descs = list(crit)
        gold = labels.index(str(row["expected"]))
    else:
        raise ValueError(f"unknown item type {kind!r} in {row.get('id')}")
    assert len(descs) == len(labels), row.get("id")
    return {
        "source": "jevbench",
        "state": f"{q['instructions']}\n\n{row['state']}",
        "options": [f"{l}: {d}" for l, d in zip(labels, descs)],
        "label": gold,
        "meta": {"id": row["id"], "tier": tier, "family": row["family"], "type": kind,
                 "yes_index": labels.index("yes") if kind == "noul" else None},
    }


def load_public(root="data/jevbench"):
    import json

    items = []
    for tier, fname in TIERS.items():
        with open(os.path.join(root, "datasets", "public", f"{fname}.jsonl")) as f:
            items += [convert_item(json.loads(l), tier) for l in f if l.strip()]
    return items


def contamination_overlap(train_items, jev_items, n=13):
    """Count training items sharing a 13-word n-gram with any JevBench item."""
    grams = eval_ngram_set({"jevbench": jev_items}, n)
    return sum(is_contaminated(x, grams, n) for x in train_items)


def _subset(items, probs, encs, idx):
    if not idx:
        return {"n": 0}
    m = calibration_metrics([probs[i] for i in idx], [items[i]["label"] for i in idx])
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


def summarize(items, probs, encs):
    labels = [it["label"] for it in items]
    pred = [int(p.argmax()) for p in probs]
    out = {"overall": _subset(items, probs, encs, list(range(len(items))))}
    out["tiers"] = {t: _subset(items, probs, encs, [i for i, it in enumerate(items) if it["meta"]["tier"] == t]) for t in TIERS}
    out["long_over_%d_tokens" % LONG_TOKENS] = _subset(items, probs, encs, [i for i, e in enumerate(encs) if len(e.prompt_ids) > LONG_TOKENS])
    yn = [i for i, it in enumerate(items) if it["meta"]["type"] == "noul"]
    yes_pred = sum(pred[i] == items[i]["meta"]["yes_index"] for i in yn)
    out["yes_no"] = {"n": len(yn), "yes_predictions": yes_pred, "gold_yes": sum(labels[i] == items[i]["meta"]["yes_index"] for i in yn),
                     "acc": round(sum(pred[i] == labels[i] for i in yn) / max(1, len(yn)), 4)}
    out["chance_acc"] = round(sum(1 / len(it["options"]) for it in items) / len(items), 4)
    return out


def evaluate(models, tok, items, train_items=None, version=None, log=print):
    """models: dict name -> model. Returns a JSON-serialisable results dict."""
    res = {"version": version, "n_items": len(items), "models": {}}
    if train_items is not None:
        res["train_items_overlapping_jevbench_13gram"] = contamination_overlap(train_items, items)
    for name, m in models.items():
        m.eval()
        probs, encs = score_items(m, tok, items, log=log, name=f"jevbench/{name}")
        res["models"][name] = summarize(items, probs, encs)
        log(f"{name}: acc={res['models'][name]['overall']['acc']:.4f}")
    res["n_long_items"] = sum(1 for e in encs if len(e.prompt_ids) > LONG_TOKENS)
    return res


def markdown(res):
    names = list(res["models"])
    head = "| Metric | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)
    g = lambda f: " | ".join(f(res["models"][n]) for n in names)
    rows = [
        ("Overall accuracy (231)", lambda m: f"{m['overall']['acc']:.3f}"),
        ("Easy accuracy", lambda m: f"{m['tiers']['easy']['acc']:.3f} (n={m['tiers']['easy']['n']})"),
        ("Standard accuracy", lambda m: f"{m['tiers']['standard']['acc']:.3f} (n={m['tiers']['standard']['n']})"),
        ("Hard accuracy", lambda m: f"{m['tiers']['hard']['acc']:.3f} (n={m['tiers']['hard']['n']})"),
        (f"Accuracy, prompts over {LONG_TOKENS} tokens", lambda m: (lambda s: f"{s['acc']:.3f} (n={s['n']})" if s["n"] else "n/a")(m[f"long_over_{LONG_TOKENS}_tokens"])),
        ("ECE (15 bins)", lambda m: f"{m['overall']['ece']:.3f}"),
        ("Yes predictions on yes/no items", lambda m: f"{m['yes_no']['yes_predictions']} of {m['yes_no']['n']} (gold yes: {m['yes_no']['gold_yes']})"),
        ("Chance accuracy (uniform guess)", lambda m: f"{m['chance_acc']:.3f}"),
    ]
    lines = [head] + [f"| {label} | {g(f)} |" for label, f in rows]
    return "\n".join(lines)
