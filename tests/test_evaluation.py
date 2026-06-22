"""Offline guard for the verdict evaluation: the curated labeled set must classify perfectly
(every sourced fraud -> 🔴, the restated control -> not 🔴). No network (live controls excluded)."""
import evaluate


def test_curated_cases_classify_correctly():
    results, _ = evaluate.evaluate(include_live=False)
    by = {r["name"]: r for r in results}
    # Every curated case scored (no errors).
    assert all(r["predicted"] is not None for r in results), \
        [r for r in results if r["predicted"] is None]
    # The four sourced frauds/distress are flagged 🔴; the restated control is not.
    assert by["Enron (FY2000)"]["predicted"] == "flag"
    assert by["WorldCom (FY2001)"]["predicted"] == "flag"
    assert by["Sunbeam (FY1997, cooked)"]["predicted"] == "flag"
    assert by["Lehman Brothers (FY2007)"]["predicted"] == "flag"
    assert by["Sunbeam (restated)"]["predicted"] == "clean"


def test_curated_precision_recall_are_perfect():
    _, m = evaluate.evaluate(include_live=False)
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["FP"] == 0 and m["FN"] == 0
