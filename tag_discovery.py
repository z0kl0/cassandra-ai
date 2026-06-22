"""
Dev-time XBRL tag-discovery aid.

The forensic engine is only as good as its `tag_map`: when a filer reports a concept under a tag
the map doesn't know (e.g. Alphabet folding finance-lease ROU assets into PP&E in FY2025, or a
filer splitting out its cost line), the value silently reads as 0 and can corrupt a score. This tool surfaces
those gaps for a given company-year so a human can extend `tag_map` deliberately:

  1. MISSING concepts the engine wanted but couldn't resolve this year.
  2. The largest UNMAPPED USD tags the filer reports (ranked by magnitude) — the likely homes
     for those missing concepts.
  3. For each missing concept, the closest unmapped tags by name similarity (a *suggestion*).

This is an authoring aid only. It never auto-maps anything at runtime — a human commits the
mapping, preserving the deterministic, auditable extraction the project depends on.

Usage:  python tag_discovery.py TICKER [YEAR]
"""
import difflib
import logging

from sec_client import SECClient
from forensics import ForensicEngine

logger = logging.getLogger(__name__)


def _annual_usd_value(engine: ForensicEngine, points: list, year: int):
    """A representative annual USD magnitude for `year` (for ranking), or None."""
    cands = [dp for dp in points
             if dp.get("form") in engine.ANNUAL_FORMS and dp.get("fp") == "FY"
             and (dp.get("fy") == year or dp.get("end", "").endswith("-12-31"))
             and str(year) in dp.get("end", "")]
    if not cands:
        return None
    best = max(cands, key=lambda dp: (dp.get("end", ""), dp.get("filed", "")))
    return float(best["val"])


def discover(facts: dict, year: int, engine: ForensicEngine = None, top_n: int = 25) -> dict:
    """Returns {coverage, missing, unmapped, suggestions} for one company-year."""
    engine = engine or ForensicEngine()
    mapped_tags = {t for tags in engine.tag_map.values() for t in tags}
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    unmapped = []  # (abs_value, value, tag, label)
    for tag, data in us_gaap.items():
        usd = data.get("units", {}).get("USD")
        if not usd:
            continue
        val = _annual_usd_value(engine, usd, year)
        if val is None or tag in mapped_tags:
            continue
        unmapped.append((abs(val), val, tag, data.get("label") or tag))
    unmapped.sort(reverse=True)

    # Concepts the engine wanted but did not resolve this year -> the actionable gaps.
    fin = engine.extract_financials(facts, year)
    missing = [c for c, prov in fin["_provenance"].items() if prov is None]

    # Suggest the closest unmapped tags per missing concept (name similarity to its known tags).
    suggestions = {}
    for concept in missing:
        cand_tags = engine.tag_map[concept]
        scored = [(max(difflib.SequenceMatcher(None, ct.lower(), tag.lower()).ratio()
                       for ct in cand_tags), tag, val)
                  for _, val, tag, _ in unmapped]
        scored.sort(reverse=True)
        suggestions[concept] = scored[:3]

    return {"coverage": fin["_coverage"], "missing": missing,
            "unmapped": unmapped[:top_n], "suggestions": suggestions}


def _fmt(v):
    return f"{v:,.0f}"


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "GOOGL"
    engine = ForensicEngine()
    facts = SECClient().get_company_facts(ticker)
    if not facts:
        print(f"No SEC facts for {ticker}."); sys.exit(1)

    if len(sys.argv) > 2:
        year = int(sys.argv[2])
    else:
        pts = facts["facts"]["us-gaap"].get("Assets", {}).get("units", {}).get("USD", [])
        yrs = [dp["fy"] for dp in pts if dp.get("form") in engine.ANNUAL_FORMS
               and dp.get("fp") == "FY" and dp.get("fy")]
        year = max(yrs) if yrs else None

    rep = discover(facts, year, engine)
    print(f"\n=== Tag discovery: {facts.get('entityName', ticker)}  FY{year} ===")
    print(f"Engine coverage: {rep['coverage']:.0%}  ({len(rep['missing'])} concept(s) unresolved)")

    if rep["missing"]:
        print("\nMISSING concepts (wanted but not resolved) + suggested tags to add:")
        for concept in rep["missing"]:
            print(f"  - {concept}")
            for score, tag, val in rep["suggestions"][concept]:
                if score > 0.55:  # only show plausible name matches
                    print(f"        ~{score:.2f}  {tag}  ({_fmt(val)})")

    print(f"\nLargest UNMAPPED USD tags reported in FY{year} (candidates the engine ignores):")
    for _, val, tag, label in rep["unmapped"]:
        print(f"  {_fmt(val):>18}  {tag}")
