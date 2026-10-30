import math
import os
import subprocess

import pytest
import torch

from llm_as_jev.eval import calibration_metrics
from llm_as_jev.loss import (build_instance, combine, ctx_losses, ctx_reference, node_losses,
                             node_reference, run_instances)
from llm_as_jev.prompt import SuffixCache, SuffixTrie, encode_item, format_user_message
from llm_as_jev.scoring import naive_scores, probs_from_scores, score_candidates
from llm_as_jev.train import lr_multiplier, make_micro_batches

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def tiny(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096)
    return Qwen3ForCausalLM(cfg).float().to(DEV).eval()


def opts(k):
    return [f"option {i}" for i in range(1, k + 1)]


# ---------------------------------------------------------------- trie / prompt
def test_trie_basic():
    t = SuffixTrie([(1, 9), (2, 3, 9), (2, 4, 9)])
    assert t.legal(()) == [1, 2]
    assert t.legal((2,)) == [3, 4]
    assert t.legal((1,)) == [9]
    assert t.legal((2, 3)) == [9]
    assert t.depth == 3
    with pytest.raises(AssertionError):
        SuffixTrie([(1,), (1, 2)])


@pytest.mark.parametrize("k", [2, 9, 10, 77, 151])
def test_suffix_tokens_and_depth(tok, k):
    enc = encode_item(tok, "s", opts(k))
    assert len(enc.suffixes) == k
    assert enc.trie.depth == int(math.log10(k)) + 2
    assert tok.decode(enc.suffixes[k - 1]) == f"{k}]"
    assert tok.decode(enc.prompt_ids).endswith("Best answer: [")
    assert len(set(enc.suffixes)) == k


def test_prompt_format_and_no_thinking(tok):
    assert format_user_message("x", ["a", "b"]).startswith("State: x\nOptions:\n[1] a\n[2] b\nSelect one option.")
    enc = encode_item(tok, "x", ["a", "b"])
    text = tok.decode(enc.prompt_ids)
    assert "<think>" in text and "</think>" in text  # empty think block = thinking disabled
    assert enc.reply_start < len(enc.prompt_ids) - 1


# ---------------------------------------------------------------- scoring
@pytest.mark.parametrize("k", [3, 12, 40])
def test_parallel_matches_naive(tok, tiny, k):
    enc = encode_item(tok, "Pick the best", opts(k))
    fast = score_candidates(tiny, enc, chunk=7)
    slow = naive_scores(tiny, enc)
    assert torch.allclose(fast, slow, atol=1e-4), (fast - slow).abs().max()
    assert abs(probs_from_scores(fast).sum().item() - 1) < 1e-5


# ---------------------------------------------------------------- loss
def _forward(tiny, tok, insts):
    return run_instances(tiny, insts, tok.pad_token_id, DEV)


def test_loss_anchors_vanish_at_reference(tok, tiny):
    items = [{"state": "q1", "options": opts(12), "label": 5}, {"state": "q2", "options": opts(3), "label": 0}]
    insts = [build_instance(tok, it, 1, i, SuffixCache()) for i, it in enumerate(items)]
    with torch.no_grad():
        out = _forward(tiny, tok, insts)
        nref = {k: v.to(DEV) for k, v in node_reference(out["node_z"], out["legal"]).items()}
        cref = {k: v.to(DEV) for k, v in ctx_reference(out["ctx_z"]).items()}
        tree, km, ko = node_losses(out["node_z"], out["legal"], out["gold"], nref, DEV)
        kc = ctx_losses(out["ctx_z"], cref)
        total, parts = combine(tree, km, ko, kc, out["node_ex"], out["node_w"], out["ctx_ex"], len(insts))
    assert torch.isfinite(total).all()
    assert parts["mass"].abs().max() < 1e-4 and parts["out"].abs().max() < 1e-4 and parts["pos"].abs().max() < 1e-4
    assert (parts["tree"] > 0).all()


def test_tree_loss_equals_manual(tok, tiny):
    it = {"state": "q", "options": opts(12), "label": 10}
    inst = build_instance(tok, it, 1, 0, SuffixCache())
    enc = encode_item(tok, "q", opts(12))
    gold = enc.suffixes[10]
    with torch.no_grad():
        out = _forward(tiny, tok, [inst])
        nref = {k: v.to(DEV) for k, v in node_reference(out["node_z"], out["legal"]).items()}
        tree, _, _ = node_losses(out["node_z"], out["legal"], out["gold"], nref, DEV)
        manual = 0.0
        ids = torch.tensor([enc.prompt_ids + list(gold)], device=DEV)
        lp = torch.log_softmax(tiny(ids).logits[0].float(), -1)
        n = len(enc.prompt_ids)
        for j in range(len(gold)):
            legal = enc.trie.legal(gold[:j])
            lm = torch.logsumexp(lp[n - 1 + j, legal], 0)
            manual += -(lp[n - 1 + j, gold[j]] - lm).item()
    assert abs(tree[: inst.n_gold].sum().item() - manual) < 1e-4


def test_instance_structure(tok):
    it = {"state": "q", "options": opts(151), "label": 150}
    inst = build_instance(tok, it, 7, 0, SuffixCache())
    assert inst.n_gold == 4 and len(inst.off_prefixes) == 3
    assert abs(sum(inst.node_w) - 1.0) < 1e-9
    gold = tuple(inst.rows[0][len(inst.rows[0]) - 4:])
    assert all(p != gold[: len(p)] for p in inst.off_prefixes)
    again = build_instance(tok, it, 7, 0, SuffixCache(), off_prefixes=inst.off_prefixes)
    assert again.rows == inst.rows


def test_loss_has_gradient(tok, tiny):
    it = {"state": "q", "options": opts(5), "label": 2}
    inst = build_instance(tok, it, 1, 0, SuffixCache())
    with torch.no_grad():
        out = _forward(tiny, tok, [inst])
        nref = {k: v.to(DEV) for k, v in node_reference(out["node_z"], out["legal"]).items()}
        cref = {k: v.to(DEV) for k, v in ctx_reference(out["ctx_z"]).items()}
    tiny.zero_grad()
    out = _forward(tiny, tok, [inst])
    tree, km, ko = node_losses(out["node_z"], out["legal"], out["gold"], nref, DEV)
    total, _ = combine(tree, km, ko, ctx_losses(out["ctx_z"], cref), out["node_ex"], out["node_w"], out["ctx_ex"], 1)
    total.sum().backward()
    assert tiny.model.embed_tokens.weight.grad is not None
    tiny.zero_grad()


# ---------------------------------------------------------------- misc
def test_lr_schedule():
    total = 100
    assert lr_multiplier(0, total) == pytest.approx(0.1)
    assert lr_multiplier(9, total) == pytest.approx(1.0)
    assert lr_multiplier(10, total) == pytest.approx(1.0)
    assert lr_multiplier(total, total) == pytest.approx(0.1)


def test_micro_batches_cover_and_respect_budget(tok):
    items = [{"state": "x " * (5 + 40 * i), "options": opts(4), "label": 0} for i in range(10)]
    insts = [build_instance(tok, it, 1, i, SuffixCache()) for i, it in enumerate(items)]
    mbs = make_micro_batches(insts, 2000, 4)
    assert sorted(i for mb in mbs for i in mb) == list(range(10))
    for mb in mbs:
        rows = sum(len(insts[i].rows) for i in mb)
        assert len(mb) <= 4
        if len(mb) > 1:
            assert rows * max(len(r) for i in mb for r in insts[i].rows) <= 2000


def test_calibration_metrics():
    m = calibration_metrics([[1.0, 0.0], [0.5, 0.5]], [0, 1])
    assert m["acc"] == 1.0 or m["acc"] == 0.5
    perfect = calibration_metrics([[1.0, 0.0], [0.0, 1.0]], [0, 1])
    assert perfect["acc"] == 1 and perfect["ece"] == 0 and perfect["brier"] == 0


def test_token_never_printed(capsys, monkeypatch):
    from llm_as_jev import auth

    monkeypatch.setenv("my_hf_token", "hf_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE12")
    tok_val = auth.get_hf_token()
    assert tok_val.startswith("hf_FAKE")
    captured = capsys.readouterr()
    assert "FAKEFAKE" not in captured.out + captured.err


def test_no_token_in_tracked_files():
    import re

    root = os.path.dirname(os.path.dirname(__file__))
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True).stdout.split()
    real = os.environ.get("my_hf_token")
    pat = re.compile(r"hf_[A-Za-z0-9]{30,}")
    for f in files:
        p = os.path.join(root, f)
        if not os.path.isfile(p):
            continue
        text = open(p, errors="ignore").read()
        assert not pat.search(text), f"token-like string in {f}"
        if real:
            assert real not in text, f"token present in {f}"
