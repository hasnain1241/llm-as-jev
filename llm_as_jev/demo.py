"""Interactive demo: probability distribution over options, base vs fine-tuned."""
import torch

from .prompt import encode_item
from .scoring import probs_from_scores, score_candidates


@torch.no_grad()
def predict(model, tok, state, options):
    """Return a list of probabilities, one per option (sums to 1)."""
    enc = encode_item(tok, state, options)
    return probs_from_scores(score_candidates(model, enc)).cpu().tolist()


def show(models, tok, state, options, width=30):
    """models: dict name -> model. Prints one bar chart per model."""
    print(f"State: {state}")
    for name, m in models.items():
        p = predict(m, tok, state, options)
        best = max(range(len(p)), key=p.__getitem__)
        print(f"\n{name}")
        for i, (o, x) in enumerate(zip(options, p), 1):
            bar = "#" * round(x * width)
            print(f"  [{i}] {x:6.1%} {bar:<{width}} {o}{'  <- best' if i - 1 == best else ''}")


def ask(models, tok):
    """Prompt for a state and options in the notebook, then show the distributions."""
    state = input("State: ").strip()
    options = []
    print("Enter options one per line, empty line to finish:")
    while True:
        o = input(f"  option {len(options) + 1}: ").strip()
        if not o:
            break
        options.append(o)
    if len(options) < 2:
        print("Need at least 2 options.")
        return
    show(models, tok, state, options)
