"""
Precision/recall evaluation of CASSANDRA's deterministic verdict.

The detection task is binary: does the engine raise a 🔴 **High Risk** verdict for a company that
should be flagged (real accounting fraud or severe distress), while leaving clean companies alone?
Prediction = (verdict Emoji == "RED"). 🟡/🟢/⚪ count as "not flagged".

Labeled set (small but real and sourced):
  Positives (should be 🔴): Enron, WorldCom, Sunbeam (cooked), Lehman          [curated CSV]
  Negatives (should NOT be 🔴): Sunbeam (restated control) + blue-chip controls [curated + live SEC]

The synthetic TEXTBOOK_DEMO is intentionally NOT scored here — it's a unit-test fixture, not a real
case, and counting it would inflate recall. Run:  python evaluate.py   (live controls need network).
"""
import logging

from forensics import ForensicEngine
from curated_cases import CuratedCaseLoader

logger = logging.getLogger(__name__)

# (display name, source, identifier, expected)  — expected in {"flag", "clean"}; kind is for display.
LABELED = [
    ("Enron (FY2000)",            "curated", "ENRON",            "flag",  "Fraud"),
    ("WorldCom (FY2001)",         "curated", "WORLDCOM",         "flag",  "Fraud"),
    ("Sunbeam (FY1997, cooked)",  "curated", "SUNBEAM",          "flag",  "Fraud"),
    ("Lehman Brothers (FY2007)",  "curated", "LEHMAN",           "flag",  "Distress"),
    ("Sunbeam (restated)",        "curated", "SUNBEAM_RESTATED", "clean", "Control"),
    ("Microsoft",                 "live",    "MSFT",             "clean", "Blue-chip"),
    ("Apple",                     "live",    "AAPL",             "clean", "Blue-chip"),
    ("Alphabet",                  "live",    "GOOGL",            "clean", "Blue-chip"),
    ("Amazon",                    "live",    "AMZN",             "clean", "Blue-chip"),
    ("Johnson & Johnson",         "live",    "JNJ",              "clean", "Blue-chip"),
    ("Coca-Cola",                 "live",    "KO",               "clean", "Blue-chip"),
    ("Walmart",                   "live",    "WMT",              "clean", "Blue-chip"),
    ("Nvidia",                    "live",    "NVDA",             "clean", "Blue-chip"),
]


def _live_annual_years(engine, facts):
    """Annual fiscal years (descending) present in the filing's reporting currency."""
    cur = engine._detect_currency(facts)
    pts = facts.get("facts", {}).get("us-gaap", {}).get("Assets", {}).get("units", {}).get(cur, [])
    years = {dp["fy"] for dp in pts
             if dp.get("form") in engine.ANNUAL_FORMS and dp.get("fp") == "FY" and dp.get("fy")}
    return sorted(years, reverse=True)


def _verdict_for(spec, engine, loader, sec):
    """Score one labeled case the same way app.analyze() does. Returns the verdict dict."""
    _name, source, ident, _exp, _kind = spec
    benford = None
    if source == "curated":
        yrs = loader.get_company_years(ident)
        keys = sorted(yrs)
        cur, pri = yrs[keys[-1]], yrs[keys[-2]]
    else:
        facts = sec.get_company_facts(ident)
        if not facts:
            raise RuntimeError(f"no SEC facts for {ident}")
        years = _live_annual_years(engine, facts)
        if len(years) < 2:
            raise RuntimeError(f"<2 annual years for {ident}")
        cur = engine.extract_financials(facts, years[0])
        pri = engine.extract_financials(facts, years[1])
        benford = engine.calculate_benford_deviation(facts, years[0])
    return engine.calculate_verdict(
        m_score=engine.calculate_m_score(cur, pri),
        z_score=engine.calculate_z_score(cur),
        sloan=engine.calculate_sloan_ratio(cur, pri),
        piotroski=engine.calculate_piotroski_f_score(cur, pri),
        leverage=engine.calculate_leverage(cur, pri),
        benford=benford,
    )


def evaluate(include_live=True):
    """Score every labeled case. Returns (results, metrics). Curated cases need no network."""
    engine = ForensicEngine()
    loader = CuratedCaseLoader()
    sec = None
    specs = [s for s in LABELED if include_live or s[1] == "curated"]
    if any(s[1] == "live" for s in specs):
        from sec_client import SECClient
        sec = SECClient()

    results = []
    for spec in specs:
        name, source, ident, expect, kind = spec
        try:
            v = _verdict_for(spec, engine, loader, sec)
            emoji = v.get("Emoji")
            predicted = "flag" if emoji == "RED" else "clean"
            flags = v.get("Flags") or []
            signal = flags[0] if (predicted == "flag" and flags) else "—"
            results.append({"name": name, "kind": kind, "expected": expect,
                            "verdict": f"{emoji} {v.get('Verdict')}", "predicted": predicted,
                            "correct": predicted == expect, "signal": signal})
        except Exception as e:                                  # network/data gap -> reported, not scored
            results.append({"name": name, "kind": kind, "expected": expect, "verdict": f"ERROR: {e}",
                            "predicted": None, "correct": None, "signal": "—"})
    return results, compute_metrics(results)


def compute_metrics(results):
    """Confusion matrix + precision/recall/F1/specificity/accuracy over the scored results."""
    scored = [r for r in results if r["predicted"] is not None]
    tp = sum(r["expected"] == "flag" and r["predicted"] == "flag" for r in scored)
    fn = sum(r["expected"] == "flag" and r["predicted"] == "clean" for r in scored)
    fp = sum(r["expected"] == "clean" and r["predicted"] == "flag" for r in scored)
    tn = sum(r["expected"] == "clean" and r["predicted"] == "clean" for r in scored)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    spec = tn / (tn + fp) if (tn + fp) else 1.0
    acc = (tp + tn) / len(scored) if scored else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "precision": prec, "recall": rec,
            "specificity": spec, "accuracy": acc, "f1": f1, "n": len(scored)}


def _print_report(results, m):
    print("\n## CASSANDRA — verdict evaluation (🔴 = flagged)\n")
    print(f"| {'Company':26} | {'Type':9} | {'Expected':8} | {'Verdict':16} | {'Pred':6} | "
          f"{'✓':1} | Key signal |")
    print(f"|{'-'*28}|{'-'*11}|{'-'*10}|{'-'*18}|{'-'*8}|{'-'*3}|{'-'*11}|")
    for r in results:
        ok = "✓" if r["correct"] else ("·" if r["correct"] is None else "✗")
        exp = "🔴 flag" if r["expected"] == "flag" else "clean"
        pred = "—" if r["predicted"] is None else r["predicted"]
        print(f"| {r['name']:26} | {r['kind']:9} | {exp:8} | {r['verdict']:16} | {pred:6} | "
              f"{ok:1} | {r['signal'][:40]} |")

    print("\n### Confusion matrix")
    print("                  Predicted 🔴   Predicted clean")
    print(f"  Actual flag        {m['TP']:^9}     {m['FN']:^9}")
    print(f"  Actual clean       {m['FP']:^9}     {m['TN']:^9}")

    print("\n### Metrics")
    print(f"  Precision   {m['precision']:.3f}   (of those flagged 🔴, how many were truly bad)")
    print(f"  Recall      {m['recall']:.3f}   (of the truly bad, how many we caught)")
    print(f"  Specificity {m['specificity']:.3f}   (of the clean, how many we left alone)")
    print(f"  F1          {m['f1']:.3f}")
    print(f"  Accuracy    {m['accuracy']:.3f}   (n={m['n']})")
    skipped = [r['name'] for r in results if r['predicted'] is None]
    if skipped:
        print(f"\n  ⚠ not scored (errors/network): {', '.join(skipped)}")
    print("\n  Note: small, curated, point-in-time set — illustrative of calibration, not a population"
          " estimate. Frauds are pre-XBRL classics (sourced CSV); negatives are live SEC blue-chips.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    import sys
    try:                                   # emoji-safe on the Windows console (cp1252)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    live = "--offline" not in sys.argv
    res, metrics = evaluate(include_live=live)
    _print_report(res, metrics)
