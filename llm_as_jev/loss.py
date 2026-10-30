"""Tree-restricted likelihood plus top-64 anchors to the frozen base model.

An Instance expands one example into rows (plain sequences):
  row 0       : prompt + full gold suffix (gold-path nodes, context positions)
  rows 1..3   : prompt + off-path trie prefix (one off-path node each)
Reference stats (from the frozen model) share the same node ordering:
  nodes = gold-path nodes (j = 0..m-1), then off-path nodes.
"""
import random
from dataclasses import dataclass, field

import torch

from .prompt import encode_item

TOP = 64
N_OFF = 3
EPS = 1e-12


@dataclass
class Instance:
    rows: list  # list of token-id lists
    node_row: list  # per node: row index
    node_pos: list  # per node: token index within the row
    node_legal: list  # per node: legal token ids
    gold_next: list  # per gold node: gold token id
    n_gold: int
    ctx_pos: list  # [(row, pos)] reply start, position after ']'
    node_w: list  # per node anchor weight
    off_prefixes: list = field(default_factory=list)


def build_instance(tokenizer, item, seed, idx, cache=None, off_prefixes=None, enc=None):
    enc = enc or encode_item(tokenizer, item["state"], item["options"], cache)
    n = len(enc.prompt_ids)
    gold = enc.suffixes[item["label"]]
    m = len(gold)
    gold_prefixes = {gold[:j] for j in range(m)}
    if off_prefixes is None:
        cands = sorted(p for p in enc.trie.internal_nodes() if len(p) > 0 and p not in gold_prefixes)
        rng = random.Random(f"{seed}-off-{idx}")
        off_prefixes = rng.sample(cands, min(N_OFF, len(cands)))
    off_prefixes = [tuple(p) for p in off_prefixes]
    rows = [enc.prompt_ids + list(gold)]
    node_row = [0] * m
    node_pos = [n - 1 + j for j in range(m)]
    node_legal = [enc.trie.legal(gold[:j]) for j in range(m)]
    for p in off_prefixes:
        rows.append(enc.prompt_ids + list(p))
        node_row.append(len(rows) - 1)
        node_pos.append(n + len(p) - 1)
        node_legal.append(enc.trie.legal(p))
    n_off = len(off_prefixes)
    w_gold = (0.5 if n_off else 1.0) / m
    w_off = 0.5 / n_off if n_off else 0.0
    return Instance(
        rows=rows,
        node_row=node_row,
        node_pos=node_pos,
        node_legal=node_legal,
        gold_next=list(gold),
        n_gold=m,
        ctx_pos=[(0, enc.reply_start), (0, n + m - 1)],
        node_w=[w_gold] * m + [w_off] * n_off,
        off_prefixes=off_prefixes,
    )


# ------------------------------------------------------------------ reference stats
def _legal_mask(legal_lists, vocab, device):
    width = max(len(x) for x in legal_lists)
    idx = torch.full((len(legal_lists), width), vocab, dtype=torch.long)
    for i, l in enumerate(legal_lists):
        idx[i, : len(l)] = torch.tensor(l)
    mask = torch.zeros(len(legal_lists), vocab + 1, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask[:, :vocab].to(device)


@torch.no_grad()
def node_reference(z, legal_lists):
    """Stats of the base model at nodes. z: (N, V) logits."""
    z = z.float()
    lm = _legal_mask(legal_lists, z.shape[-1], z.device)
    logZ = torch.logsumexp(z, -1)
    leg = torch.logsumexp(z.masked_fill(~lm, float("-inf")), -1) - logZ
    ill = torch.logsumexp(z.masked_fill(lm, float("-inf")), -1) - logZ
    top_z, ids = z.masked_fill(lm, float("-inf")).topk(TOP, -1)
    return {
        "m0": leg.exp().double().cpu(),
        "ill0": ill.exp().double().cpu(),
        "ids": ids.int().cpu(),
        "p": (top_z - logZ[:, None]).exp().cpu(),
    }


@torch.no_grad()
def ctx_reference(z):
    z = z.float()
    logZ = torch.logsumexp(z, -1)
    top_z, ids = z.topk(TOP, -1)
    return {"ids": ids.int().cpu(), "p": (top_z - logZ[:, None]).exp().cpu()}


# ------------------------------------------------------------------ losses
def _kl_trunc(p0, r0, logq, logr):
    """KL(P0 || Q) with P0 = (p0 over kept tokens, r0 tail bucket). logq/logr from the model."""
    kl = (p0 * (torch.log(p0.clamp_min(EPS)) - logq)).sum(-1)
    tail = r0 * (torch.log(r0.clamp_min(EPS)) - logr)
    return kl + torch.where(r0 > 0, tail, torch.zeros_like(tail))


def node_losses(z, legal_lists, gold_tok, ref, dev):
    """z: (N, V) model logits at nodes.
    gold_tok: (N,) gold token (-1 for off-path nodes). ref: dict of tensors on dev.
    Returns per-node (tree_nll, kl_mass, kl_out)."""
    z = z.float()
    V = z.shape[-1]
    lm = _legal_mask(legal_lists, V, dev)
    logZ = torch.logsumexp(z, -1)
    lse_leg = torch.logsumexp(z.masked_fill(~lm, float("-inf")), -1)
    z_ill = z.masked_fill(lm, float("-inf"))
    lse_ill = torch.logsumexp(z_ill, -1)
    # tree: -log( p(gold) / legal_mass )
    zg = z.gather(1, gold_tok.clamp_min(0)[:, None]).squeeze(1)
    tree = (lse_leg - zg) * (gold_tok >= 0)
    # Bernoulli KL on legal mass
    m0, i0 = ref["m0"].float(), ref["ill0"].float()
    logm, logi = lse_leg - logZ, lse_ill - logZ
    kl_mass = m0 * (torch.log(m0.clamp_min(EPS)) - logm) + i0 * (torch.log(i0.clamp_min(EPS)) - logi)
    # KL over illegal-renormalized distributions
    q0 = ref["p"] / i0.clamp_min(EPS)[:, None]
    r0 = (1 - q0.sum(-1)).clamp_min(0)
    ids = ref["ids"].long()
    logq = z_ill.gather(1, ids) - lse_ill[:, None]
    tail = z_ill.scatter(1, ids, float("-inf"))
    logr = torch.logsumexp(tail, -1) - lse_ill
    kl_out = _kl_trunc(q0, r0, logq, logr)
    return tree, kl_mass, kl_out


def ctx_losses(z, ref):
    z = z.float()
    logZ = torch.logsumexp(z, -1)
    ids = ref["ids"].long()
    logq = z.gather(1, ids) - logZ[:, None]
    logr = torch.logsumexp(z.scatter(1, ids, float("-inf")), -1) - logZ
    p0 = ref["p"]
    r0 = (1 - p0.sum(-1)).clamp_min(0)
    return _kl_trunc(p0, r0, logq, logr)


def combine(tree, kl_mass, kl_out, kl_ctx, node_ex, node_w, ctx_ex, n_ex, lam=1.0):
    """Per-example reduction. kl_ctx has 2 entries per example (weight 0.5 each)."""
    z = tree.new_zeros(n_ex)
    l_tree = z.index_add(0, node_ex, tree)
    l_mass = z.index_add(0, node_ex, node_w * kl_mass)
    l_out = z.index_add(0, node_ex, node_w * kl_out)
    l_pos = z.index_add(0, ctx_ex, 0.5 * kl_ctx)
    total = l_tree + lam * (l_mass + l_out + l_pos)
    return total, {"tree": l_tree, "mass": l_mass, "out": l_out, "pos": l_pos}


# ------------------------------------------------------------------ batched forward
def run_instances(model, insts, pad_id, device):
    """Forward all rows of the given instances; return logits at nodes and ctx positions.
    The LM head is applied only at the selected positions (never over the full sequence)."""
    rows, owner = [], []
    for b, inst in enumerate(insts):
        for r in inst.rows:
            rows.append(r)
            owner.append(b)
    row_base = []
    c = 0
    for inst in insts:
        row_base.append(c)
        c += len(inst.rows)
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    att = torch.zeros(len(rows), T, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        att[i, : len(r)] = 1
    h = model.model(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
    n_r, n_p, node_ex, legal, gold = [], [], [], [], []
    c_r, c_p, ctx_ex = [], [], []
    for b, inst in enumerate(insts):
        for k in range(len(inst.node_row)):
            n_r.append(row_base[b] + inst.node_row[k])
            n_p.append(inst.node_pos[k])
            node_ex.append(b)
            legal.append(inst.node_legal[k])
            gold.append(inst.gold_next[k] if k < inst.n_gold else -1)
        for (r, p) in inst.ctx_pos:
            c_r.append(row_base[b] + r)
            c_p.append(p)
            ctx_ex.append(b)
    t = lambda x: torch.tensor(x, device=device)
    node_z = model.lm_head(h[t(n_r), t(n_p)])
    ctx_z = model.lm_head(h[t(c_r), t(c_p)])
    return {
        "node_z": node_z,
        "ctx_z": ctx_z,
        "legal": legal,
        "gold": t(gold),
        "node_ex": t(node_ex),
        "ctx_ex": t(ctx_ex),
        "node_w": torch.tensor([w for inst in insts for w in inst.node_w], device=device),
    }
