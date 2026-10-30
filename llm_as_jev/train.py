"""Reference-stat caching and the full fine-tune loop (single GPU, Colab friendly)."""
import copy
import json
import math
import os
import random
import shutil
import time

import torch

from . import hub
from .loss import (build_instance, combine, ctx_losses, ctx_reference, node_losses,
                   node_reference, run_instances)
from .prompt import SuffixCache

# Rough Colab compute-unit burn per hour (varies by region/time; used only for estimates).
CU_PER_HOUR = {"T4": 2.0, "L4": 4.8, "A100": 11.8, "H100": 18.0, "V100": 5.0}
DRIVE_DIR = "/content/drive/MyDrive/llm-as-jev"


class BudgetExceeded(RuntimeError):
    pass


class HealthCheckFailed(RuntimeError):
    pass


def load_config(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ hardware plan
def gpu_plan():
    """Pick precision, token budget per micro-batch and CU rate from the visible GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (Colab: Runtime > Change runtime type).")
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    bf16 = torch.cuda.is_bf16_supported() and "T4" not in name
    key = next((k for k in CU_PER_HOUR if k in name), None)
    plan = {
        "name": name,
        "mem_gb": round(mem, 1),
        "amp_dtype": torch.bfloat16 if bf16 else torch.float16,
        "eval_dtype": torch.bfloat16 if bf16 else torch.float16,
        "use_scaler": not bf16,
        "cu_per_hour": CU_PER_HOUR.get(key, 5.0),
        # padded tokens per micro-batch (all rows of the micro-batch); fp32 master weights + Adam use ~10 GB
        "token_budget": 4096 if mem < 20 else 8192 if mem < 30 else 16384 if mem < 60 else 32768,
        "max_micro": 8 if mem < 20 else 16,
    }
    return plan


def mem_str():
    return f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak"


# ------------------------------------------------------------------ model
def load_model(name, dtype, device="cuda"):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return m.to(device)


def load_tokenizer(name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# ------------------------------------------------------------------ batching
def make_micro_batches(insts, token_budget, max_micro):
    """Greedy length-sorted grouping so padded tokens stay under the budget."""
    order = sorted(range(len(insts)), key=lambda i: -max(len(r) for r in insts[i].rows))
    out, cur, cur_rows, cur_max = [], [], 0, 0
    for i in order:
        L = max(len(r) for r in insts[i].rows)
        rows = cur_rows + len(insts[i].rows)
        if cur and (rows * max(cur_max, L) > token_budget or len(cur) >= max_micro):
            out.append(cur)
            cur, cur_rows, cur_max = [], 0, 0
            rows = len(insts[i].rows)
        cur.append(i)
        cur_rows, cur_max = rows, max(cur_max, L)
    if cur:
        out.append(cur)
    return out


def build_instances(tok, items, seed, off_map=None):
    cache = SuffixCache()
    out = []
    for i, it in enumerate(items):
        off = off_map[i] if off_map is not None else None
        out.append(build_instance(tok, it, seed, i, cache, off_prefixes=off))
    return out


def stack_ref(refs):
    """refs: per-example reference dicts, in the same order as run_instances nodes/ctx."""
    cat = lambda k: torch.cat([r[k] for r in refs])
    return cat("m0"), cat("ill0"), cat("ids"), cat("p"), cat("cids"), cat("cp")


def ref_to_device(refs, dev):
    m0, ill0, ids, p, cids, cp = stack_ref(refs)
    return (
        {"m0": m0.to(dev), "ill0": ill0.to(dev), "ids": ids.to(dev), "p": p.to(dev)},
        {"ids": cids.to(dev), "p": cp.to(dev)},
    )


# ------------------------------------------------------------------ reference cache
@torch.no_grad()
def build_reference_cache(model, tok, items, seed, plan, log=print):
    """One pass with the frozen model. Returns a list of per-example dicts (CPU tensors)."""
    model.eval()
    dev = next(model.parameters()).device
    insts = build_instances(tok, items, seed)
    refs = [None] * len(items)
    t0 = time.time()
    batches = make_micro_batches(insts, plan["token_budget"], plan["max_micro"])
    for bi, idxs in enumerate(batches):
        sub = [insts[i] for i in idxs]
        with torch.autocast("cuda", dtype=plan["amp_dtype"]):
            out = run_instances(model, sub, tok.pad_token_id, dev)
        nref = node_reference(out["node_z"], out["legal"])
        cref = ctx_reference(out["ctx_z"])
        n0 = c0 = 0
        for j, i in enumerate(idxs):
            k = len(insts[i].node_row)
            refs[i] = {
                "off": [list(p) for p in insts[i].off_prefixes],
                "m0": nref["m0"][n0 : n0 + k].clone(),
                "ill0": nref["ill0"][n0 : n0 + k].clone(),
                "ids": nref["ids"][n0 : n0 + k].clone(),
                "p": nref["p"][n0 : n0 + k].clone(),
                "cids": cref["ids"][c0 : c0 + 2].clone(),
                "cp": cref["p"][c0 : c0 + 2].clone(),
            }
            n0 += k
            c0 += 2
        if bi % 50 == 0:
            log(f"ref cache: batch {bi + 1}/{len(batches)}  {time.time() - t0:.0f}s")
    return refs


# ------------------------------------------------------------------ loss on a micro-batch
def batch_loss(model, insts, refs, tok, plan, lam):
    dev = next(model.parameters()).device
    with torch.autocast("cuda", dtype=plan["amp_dtype"]):
        out = run_instances(model, insts, tok.pad_token_id, dev)
    nref, cref = ref_to_device(refs, dev)
    tree, kl_mass, kl_out = node_losses(out["node_z"], out["legal"], out["gold"], nref, dev)
    kl_ctx = ctx_losses(out["ctx_z"], cref)
    total, parts = combine(tree, kl_mass, kl_out, kl_ctx, out["node_ex"], out["node_w"], out["ctx_ex"], len(insts), lam)
    return total, parts


@torch.no_grad()
def root_logmass(model, insts, tok, plan):
    """Model log legal-mass at the root node for each instance (instances need off=[])."""
    dev = next(model.parameters()).device
    with torch.autocast("cuda", dtype=plan["amp_dtype"]):
        out = run_instances(model, insts, tok.pad_token_id, dev)
    z = out["node_z"].float()
    first, c = [], 0
    for inst in insts:
        first.append(c)
        c += len(inst.node_row)
    first = torch.tensor(first, device=dev)
    zr = z[first]
    leg = [insts[i].node_legal[0] for i in range(len(insts))]
    from .loss import _legal_mask

    lm = _legal_mask(leg, z.shape[-1], dev)
    return torch.logsumexp(zr.masked_fill(~lm, float("-inf")), -1) - torch.logsumexp(zr, -1)


# ------------------------------------------------------------------ checkpoints
def _bf16_copy_save(model, folder, tok=None):
    cpu = copy.deepcopy(model).to("cpu", dtype=torch.bfloat16)
    cpu.config.use_cache = True
    cpu.save_pretrained(folder, safe_serialization=True)
    if tok is not None:
        tok.save_pretrained(folder)
    del cpu


def save_drive(model, opt, sched, scaler, step, cfg, root=DRIVE_DIR):
    if not os.path.isdir("/content/drive/MyDrive"):
        return
    tmp, final = os.path.join(root, "state.tmp"), os.path.join(root, "state")
    os.makedirs(tmp, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(tmp, "model.pt"))
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler else None}, os.path.join(tmp, "opt.pt"))
    with open(os.path.join(tmp, "trainer_state.json"), "w") as f:
        json.dump({"step": step}, f)
    if os.path.isdir(final):
        shutil.rmtree(final)
    os.rename(tmp, final)


def push_hf(model, tok, step, total, done=False, work="outputs/hf_ckpt"):
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    _bf16_copy_save(model, work, tok)
    with open(os.path.join(work, "trainer_state.json"), "w") as f:
        json.dump({"step": step, "total_steps": total, "completed": done}, f)
    hub.push_checkpoint(work, step)


# ------------------------------------------------------------------ main loop
def lr_multiplier(step, total, warm_frac=0.1, final_frac=0.1):
    warm = max(1, math.ceil(warm_frac * total))
    if step < warm:
        return (step + 1) / warm
    return 1.0 - (1.0 - final_frac) * (step - warm) / max(1, total - warm)


def train(cfg, items, refs, tok, plan, log=print):
    """Full fine-tune with resume from Drive (exact) or HF (weights only)."""
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    dev = "cuda"
    model = load_model(cfg["model"], torch.float32, dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.999), fused=True,
    )
    gb = cfg["global_batch"]
    n = len(items)
    total = math.ceil(n / gb) * cfg.get("epochs", 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, total, cfg["warmup_frac"], cfg["final_lr_frac"]))
    scaler = torch.amp.GradScaler("cuda") if plan["use_scaler"] else None

    order = list(range(n))
    random.Random(seed).shuffle(order)
    insts = build_instances(tok, items, seed, off_map=[r["off"] for r in refs])
    dev_idx = order[-cfg["health_dev_size"]:]
    dev_insts = [build_instance(tok, items[i], seed, i, off_prefixes=[]) for i in dev_idx]
    dev_base = torch.stack([torch.log(refs[i]["m0"][0]) for i in dev_idx]).float()

    step = 0
    drive_state = os.path.join(DRIVE_DIR, "state")
    if os.path.isfile(os.path.join(drive_state, "trainer_state.json")):
        model.load_state_dict(torch.load(os.path.join(drive_state, "model.pt"), map_location=dev))
        st = torch.load(os.path.join(drive_state, "opt.pt"), map_location=dev)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if scaler and st["scaler"]:
            scaler.load_state_dict(st["scaler"])
        step = json.load(open(os.path.join(drive_state, "trainer_state.json")))["step"]
        log(f"Resumed exactly from Drive at step {step}")
    else:
        ck = hub.pull_checkpoint("outputs/hf_resume")
        if ck:
            from transformers import AutoModelForCausalLM

            src = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32)
            model.load_state_dict(src.state_dict())
            del src
            step = json.load(open(os.path.join(ck, "trainer_state.json")))["step"]
            for _ in range(step):
                sched.step()
            log(f"Resumed weights from HF checkpoint at step {step} (optimizer moments reset)")
    if step >= total:
        log("Training already complete.")
        return model, total

    start_step, t_start = step, time.time()
    hist = []
    while step < total:
        t0 = time.time()
        idx = order[(step * gb) % n : (step * gb) % n + gb]
        mbs = make_micro_batches([insts[i] for i in idx], plan["token_budget"], plan["max_micro"])
        agg = {"tree": 0.0, "mass": 0.0, "out": 0.0, "pos": 0.0}
        for mb in mbs:
            sel = [idx[j] for j in mb]
            total_l, parts = batch_loss(model, [insts[i] for i in sel], [refs[i] for i in sel], tok, plan, cfg["lam"])
            loss = total_l.sum() / len(idx)
            (scaler.scale(loss) if scaler else loss).backward()
            for k in agg:
                agg[k] += parts[k].sum().item() / len(idx)
        if scaler:
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if not math.isfinite(gn):
            log(f"warning: non-finite grad norm at step {step}")
        hist.append(agg)
        if step % 5 == 0 or step == 1:
            log(f"step {step}/{total} " + " ".join(f"{k}={v:.4f}" for k, v in agg.items())
                + f" gnorm={gn:.2f} lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.1f}s/step {mem_str()}")

        if step - start_step == cfg["projection_step"]:
            per = (time.time() - t_start - 0) / (step - start_step)
            proj = plan["cu_per_hour"] * per * total / 3600
            log(f"Projected training compute: {per:.1f}s/step x {total} steps = {per * total / 60:.0f} min "
                f"~ {proj:.1f} compute units ({plan['name']}, ~{plan['cu_per_hour']} CU/h)")
            if proj > cfg["budget_units"] and not cfg.get("approve_over_budget", False):
                save_drive(model, opt, sched, scaler, step, cfg)
                raise BudgetExceeded(f"Projected {proj:.1f} CU exceeds the {cfg['budget_units']} CU budget. "
                                     "State saved; set approve_over_budget=true to continue.")

        if step % cfg["health_every"] == 0:
            model.eval()
            lm = root_logmass(model, dev_insts, tok, plan).cpu()
            model.train()
            drop = (dev_base - lm).mean().item()
            log(f"health: root legal log-mass drop vs base = {drop:.3f} nats")
            if drop > cfg["health_max_drop"]:
                save_drive(model, opt, sched, scaler, step, cfg)
                raise HealthCheckFailed(f"Root legal log-mass dropped {drop:.2f} nats (> {cfg['health_max_drop']}).")

        if step % cfg["drive_every"] == 0 and step < total:
            save_drive(model, opt, sched, scaler, step, cfg)
        if step % cfg["hf_push_every"] == 0 and step < total:
            model.eval()
            push_hf(model, tok, step, total)
            model.train()

    save_drive(model, opt, sched, scaler, step, cfg)
    model.eval()
    push_hf(model, tok, step, total, done=True)
    log(f"Training done in {(time.time() - t_start) / 60:.1f} min")
    return model, total
