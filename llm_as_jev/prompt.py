"""Prompt/answer interface and the suffix trie over candidate answers."""
from dataclasses import dataclass

PREFILL = "Best answer: ["
INSTRUCTION = "Select one option. Answer only with its bracketed numeric identifier."


def format_user_message(state, options):
    lines = [f"State: {state}", "Options:"]
    lines += [f"[{i}] {o}" for i, o in enumerate(options, 1)]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


def template_text(tokenizer, state, options):
    """Chat template with thinking disabled, ending at the assistant reply start."""
    msgs = [{"role": "user", "content": format_user_message(state, options)}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_prompt(tokenizer, state, options):
    return template_text(tokenizer, state, options) + PREFILL


class SuffixTrie:
    """Trie over candidate suffix token sequences (each ends in the ']' token)."""

    def __init__(self, seqs):
        self.seqs = [tuple(s) for s in seqs]
        self.children = {}
        for s in self.seqs:
            for j in range(len(s)):
                self.children.setdefault(s[:j], set()).add(s[j])
        self.children = {k: sorted(v) for k, v in self.children.items()}
        for s in self.seqs:  # prefix-free: no candidate may end at an internal node
            assert s not in self.children, "candidate suffixes must be prefix-free"
        self.depth = max(len(s) for s in self.seqs)

    def legal(self, prefix):
        return self.children.get(tuple(prefix), [])

    def internal_nodes(self):
        return list(self.children.keys())


@dataclass
class Encoded:
    prompt_ids: list
    suffixes: list  # suffixes[k-1] = token tuple for candidate k
    reply_start: int  # index of the last template token (predicts the reply's first token)
    trie: SuffixTrie


class SuffixCache:
    """Caches suffix tokens per K. The first prompt seen for a K tokenizes every full
    string (prompt + suffix); later prompts re-verify the prefix property on the
    first and last candidate only, which keeps large-K datasets affordable."""

    def __init__(self):
        self.by_k = {}


def _suffixes_from_full(tokenizer, prompt, prompt_ids, k_total):
    texts = [prompt + f"{k}]" for k in range(1, k_total + 1)]
    full = tokenizer(texts, add_special_tokens=False)["input_ids"]
    n = len(prompt_ids)
    out = []
    for ids in full:
        assert ids[:n] == prompt_ids, "prompt tokens are not a prefix of the full tokenization"
        out.append(tuple(ids[n:]))
    return out


def encode_item(tokenizer, state, options, cache=None):
    k_total = len(options)
    prompt = build_prompt(tokenizer, state, options)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tmpl_ids = tokenizer(template_text(tokenizer, state, options), add_special_tokens=False)["input_ids"]
    assert prompt_ids[: len(tmpl_ids)] == tmpl_ids, "template tokens are not a prefix of the prompt"
    if cache is not None and k_total in cache.by_k:
        sfx = cache.by_k[k_total]
        for k in {1, k_total}:
            ids = tokenizer(prompt + f"{k}]", add_special_tokens=False)["input_ids"]
            assert ids[: len(prompt_ids)] == prompt_ids and tuple(ids[len(prompt_ids):]) == sfx[k - 1]
    else:
        sfx = _suffixes_from_full(tokenizer, prompt, prompt_ids, k_total)
        if cache is not None:
            cache.by_k[k_total] = sfx
    return Encoded(prompt_ids, sfx, len(tmpl_ids) - 1, SuffixTrie(sfx))
