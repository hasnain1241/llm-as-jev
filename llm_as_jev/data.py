"""Reason-9k construction and external eval sets.

Every item is {"source", "state", "options": [str], "label": int (0-based)}.
Loaders try several dataset IDs in order and record which one worked in SOURCES.
"""
import json
import random
import re

SOURCES = {}  # name -> {"id": ..., "note": ...}

_TRAIN_N = 1000
_WORD = re.compile(r"\w+")


# ---------------------------------------------------------------- generic loading
def load_first(name, candidates, split):
    """candidates: list of (repo_id, config or None). Tries plain, parquet branch, remote code."""
    from datasets import load_dataset

    errors = []
    for repo, cfg in candidates:
        attempts = [{}, {"revision": "refs/convert/parquet"}, {"trust_remote_code": True}]
        for extra in attempts:
            try:
                ds = load_dataset(repo, cfg, split=split, **extra) if cfg else load_dataset(repo, split=split, **extra)
                SOURCES[name] = {"id": repo, "config": cfg, "split": split, "extra": list(extra)}
                return ds
            except Exception as e:  # noqa: BLE001
                errors.append(f"{repo}/{cfg}/{list(extra)}: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError(f"No working source for {name}:\n" + "\n".join(errors))


def _names(ds, col):
    f = ds.features[col]
    return list(f.names) if hasattr(f, "names") else None


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _item(source, state, options, label):
    return {"source": source, "state": _clean(state), "options": [_clean(o) for o in options], "label": int(label)}


# ---------------------------------------------------------------- intent sets
def _intent_items(name, ds, text_col, intent_col, names=None):
    names = names or _names(ds, intent_col)
    if names is None:
        names = sorted(set(ds[intent_col]))
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for r in ds:
        v = r[intent_col]
        lab = v if isinstance(v, int) else idx[v]
        out.append(_item(name, r[text_col], names, lab))
    return out


def load_clinc(split="train"):
    ds = load_first("clinc150", [("clinc/clinc_oos", "plus"), ("DeepPavlov/clinc_oos", "plus")], split)
    return _intent_items("clinc150", ds, "text", "intent")


def load_massive(split="train"):
    ds = load_first("massive", [("AmazonScience/massive", "en-US")], split)
    return _intent_items("massive", ds, "utt", "intent")


def load_bitext(split="train"):
    ds = load_first("bitext", [("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None)], split)
    return _intent_items("bitext", ds, "instruction", "intent")


# ---------------------------------------------------------------- generic MC normalizer
def _first(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    raise KeyError(f"none of {keys} in {list(row)}")


def _mc_item(name, row):
    ctx = _first(row, ["context", "text", "passage", "article"]) if any(k in row for k in ["context", "text", "passage", "article"]) else ""
    q = _first(row, ["question", "query"])
    opts = _first(row, ["options", "answers", "choices", "candidates"])
    lab = _first(row, ["label", "correct_option", "answer", "answer_index"])
    if isinstance(lab, str):
        lab = "ABCD".index(lab) if lab in list("ABCD") else int(lab)
    state = f"{ctx}\nQuestion: {q}" if ctx else q
    return _item(name, state, opts, lab)


def load_reclor():
    cands = [("metaeval/reclor", None), ("voidful/ReClor", None), ("community-datasets/reclor", None)]
    try:
        ds = load_first("reclor", cands, "train")
        return [_mc_item("reclor", r) for r in ds]
    except RuntimeError:
        # ReClor is gated/unavailable: substitute LogiQA (v1), also logical reasoning MC.
        ds = load_first("reclor", [("lucasmccabe/logiqa", None)], "train")
        SOURCES["reclor"]["note"] = "SUBSTITUTE: ReClor unavailable, using LogiQA v1"
        return [_mc_item("reclor", r) for r in ds]


def load_logiqa2():
    ds = load_first("logiqa2", [("baber/logiqa2", None), ("lucasmccabe/logiqa", None)], "train")
    return [_mc_item("logiqa2", r) for r in ds]


def load_cosmosqa():
    ds = load_first("cosmosqa", [("allenai/cosmos_qa", None)], "train")
    out = []
    for r in ds:
        opts = [r[f"answer{i}"] for i in range(4)]
        out.append(_item("cosmosqa", f"{r['context']}\nQuestion: {r['question']}", opts, r["label"]))
    return out


def load_piqa():
    ds = load_first("piqa", [("ybisk/piqa", None), ("baber/piqa", None), ("nthngdy/piqa", None)], "train")
    return [_item("piqa", r["goal"], [r["sol1"], r["sol2"]], r["label"]) for r in ds]


def load_anli():
    ds = load_first("anli", [("allenai/art", None)], "train")
    out = []
    for r in ds:
        state = f"Observation 1: {r['observation_1']}\nObservation 2: {r['observation_2']}\nWhich hypothesis best explains the observations?"
        out.append(_item("anli", state, [r["hypothesis_1"], r["hypothesis_2"]], int(r["label"]) - 1))
    return out


def load_proofwriter():
    ds = load_first("proofwriter", [("tasksource/proofwriter", None), ("theoxo/proofwriter-deduction-with-fol", None)], "train")
    opts = ["True", "False", "Unknown"]
    out = []
    for r in ds:
        theory = _first(r, ["theory", "context", "text"])
        q = _first(r, ["question", "hypothesis", "statement"])
        a = str(_first(r, ["answer", "label"])).strip().lower()
        a = {"n/a": "unknown", "unk": "unknown"}.get(a, a)
        if a not in ("true", "false", "unknown"):
            continue
        out.append(_item("proofwriter", f"{theory}\nStatement: {q}", opts, ["true", "false", "unknown"].index(a)))
    return out


def upsample_none_of_above(items, n, rng, frac=0.16):
    """CosmosQA: ~frac of the sample has 'none of the above' as the gold answer."""
    is_nota = [x["options"][x["label"]].lower().startswith("none of the above") for x in items]
    nota = [x for x, f in zip(items, is_nota) if f]
    rest = [x for x, f in zip(items, is_nota) if not f]
    k = int(round(n * frac))
    pick = [rng.choice(nota) for _ in range(k)] if len(nota) < k else rng.sample(nota, k)
    return pick + rng.sample(rest, n - k)


# ---------------------------------------------------------------- eval sets
def _sub(items, n, seed):
    rng = random.Random(seed)
    return rng.sample(items, n) if len(items) > n else list(items)


def load_eval_sets(seed=20260922):
    """Fixed seeded subsamples; all options retained."""
    sets = {}
    ds = load_first("boolq", [("google/boolq", None)], "validation")
    sets["boolq"] = _sub([_item("boolq", f"{r['passage']}\nQuestion: {r['question']}?", ["Yes", "No"], 0 if r["answer"] else 1) for r in ds], 500, seed)

    ds = load_first("mmlu", [("cais/mmlu", "all")], "test")
    sets["mmlu"] = _sub([_item("mmlu", f"[{r['subject'].replace('_', ' ')}] {r['question']}", r["choices"], r["answer"]) for r in ds], 1200, seed)

    ds = load_first("mmlu_pro", [("TIGER-Lab/MMLU-Pro", None)], "test")
    sets["mmlu_pro"] = _sub([_item("mmlu_pro", r["question"], r["options"], r["answer_index"]) for r in ds], 800, seed)

    ds = load_first("arc_c", [("allenai/ai2_arc", "ARC-Challenge")], "test")
    sets["arc_c"] = _sub([_item("arc_c", r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])) for r in ds], 600, seed)

    ds = load_first("winogrande", [("allenai/winogrande", "winogrande_xl")], "validation")
    sets["winogrande"] = _sub([_item("winogrande", "Fill in the blank (_): " + r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1) for r in ds], 500, seed)

    ds = load_first("sciq", [("allenai/sciq", None)], "test")
    rng = random.Random(seed)
    sciq = []
    for r in ds:
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = list(range(4))
        rng.shuffle(order)
        state = (r["support"] + "\n" if r.get("support") else "") + "Question: " + r["question"]
        sciq.append(_item("sciq", state, [opts[i] for i in order], order.index(0)))
    sets["sciq"] = _sub(sciq, 400, seed)

    ds = load_first("banking77", [("PolyAI/banking77", None), ("mteb/banking77", None)], "test")
    names = _names(ds, "label")
    if names is None:
        m = {r["label"]: r["label_text"] for r in ds}
        names = [m[i] for i in range(len(m))]
    sets["banking77"] = [_item("banking77", r["text"], [n.replace("_", " ") for n in names], r["label"]) for r in ds]
    return sets


def load_jevbench(path):
    items = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            items.append(_item("jevbench", r["state"], r["options"], r["label"]))
    return items


# ---------------------------------------------------------------- contamination
def ngrams(text, n=13):
    w = [x.lower() for x in _WORD.findall(text)]
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def eval_ngram_set(eval_sets, n=13):
    grams = set()
    for items in eval_sets.values():
        for it in items:
            grams |= ngrams(it["state"] + " " + " ".join(it["options"]), n)
    return grams


def is_contaminated(item, grams, n=13):
    return bool(ngrams(item["state"] + " " + " ".join(item["options"]), n) & grams)


# ---------------------------------------------------------------- Reason-9k
LOADERS = {
    "clinc150": load_clinc,
    "massive": load_massive,
    "bitext": load_bitext,
    "reclor": load_reclor,
    "logiqa2": load_logiqa2,
    "cosmosqa": load_cosmosqa,
    "piqa": load_piqa,
    "anli": load_anli,
    "proofwriter": load_proofwriter,
}


def shuffle_options(item, rng):
    order = list(range(len(item["options"])))
    rng.shuffle(order)
    return {**item, "options": [item["options"][i] for i in order], "label": order.index(item["label"])}


def build_reason9k(tokenizer, eval_sets, seed=20260922, per_set=_TRAIN_N, max_len=8192, log=print):
    """Return the list of training items. Order shuffles are applied per example.
    Long examples (> max_len tokens) are dropped, never truncated."""
    from .prompt import build_prompt

    grams = eval_ngram_set(eval_sets)
    out = []
    for name, loader in LOADERS.items():
        rng = random.Random(f"{seed}-{name}")
        pool = loader()
        if name == "cosmosqa":
            pool = upsample_none_of_above(pool, min(len(pool), per_set * 3), rng)
            pool = [x for x in pool]
        rng.shuffle(pool)
        taken, dropped_c, dropped_l = [], 0, 0
        for it in pool:
            if len(taken) >= per_set:
                break
            if is_contaminated(it, grams):
                dropped_c += 1
                continue
            it = shuffle_options(it, rng)
            n_tok = len(tokenizer(build_prompt(tokenizer, it["state"], it["options"]), add_special_tokens=False)["input_ids"])
            if n_tok + 8 > max_len:
                dropped_l += 1
                continue
            taken.append(it)
        log(f"{name}: {len(taken)} kept (pool {len(pool)}, contaminated {dropped_c}, too long {dropped_l})")
        out += taken
    return out


def write_jsonl(items, path):
    with open(path, "w") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"id": i, **it}) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
