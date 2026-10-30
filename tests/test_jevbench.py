import pytest

from llm_as_jev.jevbench import contamination_overlap, convert_item, summarize


def row(kind, labels, expected, criteria, state="The text.", fam="f", rid="x-0"):
    return {"id": rid, "family": fam, "group": None, "split": "public", "state": state, "labels": labels,
            "expected": expected, "question": {"type": kind, "instructions": "Decide.", "criteria": criteria}}


def test_choice_keeps_canonical_order_and_gold():
    r = row("choice", ["b", "a", "c"], "c", {"a": "A desc", "b": "B desc", "c": "C desc"})
    it = convert_item(r, "easy")
    assert it["options"] == ["b: B desc", "a: A desc", "c: C desc"]
    assert it["label"] == 2
    assert it["state"] == "Decide.\n\nThe text."
    assert it["meta"]["tier"] == "easy"


def test_noul_maps_yes_no_to_true_false_descriptions():
    r = row("noul", ["no", "yes"], "yes", {"true": "all met", "false": "something missing"})
    it = convert_item(r, "standard")
    assert it["options"] == ["no: something missing", "yes: all met"]
    assert it["label"] == 1 and it["meta"]["yes_index"] == 1


def test_score_items_use_int_expected_and_list_criteria():
    r = row("score", ["0", "1", "2", "3"], 2, ["none", "minor", "major", "severe"])
    it = convert_item(r, "standard")
    assert it["label"] == 2
    assert it["options"][3] == "3: severe"


def test_question_given_as_string_is_parsed():
    r = row("choice", ["a", "b"], "a", {"a": "x", "b": "y"})
    r["question"] = str(r["question"])
    assert convert_item(r, "hard")["label"] == 0


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        convert_item(row("weird", ["a"], "a", {"a": "x"}), "easy")


def test_summarize_counts_yes_predictions():
    import torch

    class E:  # stands in for Encoded: only prompt_ids length is used
        def __init__(self, n):
            self.prompt_ids = [0] * n

    items = [convert_item(row("noul", ["no", "yes"], g, {"true": "t", "false": "f"}, rid=f"i{i}"), t)
             for i, (g, t) in enumerate([("yes", "easy"), ("no", "standard"), ("yes", "hard")])]
    probs = [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.8]), torch.tensor([0.7, 0.3])]
    s = summarize(items, probs, [E(10), E(2000), E(10)])
    assert s["overall"]["acc"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["yes_no"] == {"n": 3, "yes_predictions": 2, "gold_yes": 2, "acc": pytest.approx(0.3333, abs=1e-3)}
    assert s["tiers"]["easy"]["acc"] == 1.0
    assert s["long_over_1500_tokens"]["n"] == 1


def test_contamination_overlap_detects_shared_13gram():
    long = " ".join(f"w{i}" for i in range(20))
    jev = [{"state": long, "options": ["a"], "label": 0}]
    train = [{"state": "intro " + long, "options": ["b"], "label": 0}, {"state": "unrelated short text", "options": ["c"], "label": 0}]
    assert contamination_overlap(train, jev) == 1
