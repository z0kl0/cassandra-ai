"""
Regression tests for the deterministic forensic engine.

Three layers:
  1. Model math vs. known/textbook values (synthetic + real sourced cases).
  2. Verdict aggregation (the classification the backtest scores).
  3. XBRL resolver + robustness unit tests (synthetic SEC facts, no network/LLM).

All tests are offline and deterministic: no SEC calls, no Ollama.
"""
import pytest

from forensics import ForensicEngine
from curated_cases import CuratedCaseLoader


@pytest.fixture(scope="module")
def eng():
    return ForensicEngine()


@pytest.fixture(scope="module")
def cases():
    return CuratedCaseLoader()


def _pair(cases, case_id):
    yrs = cases.get_company_years(case_id)
    keys = sorted(yrs)
    return yrs[keys[-1]], yrs[keys[-2]]  # current, prior


# --------------------------------------------------------------------------------------
# 1. Model math vs. known values
# --------------------------------------------------------------------------------------

def test_textbook_demo_known_scores(eng, cases):
    """The synthetic case has hand-verifiable M and Z scores; guards the core formulas."""
    cur, pri = _pair(cases, "TEXTBOOK_DEMO")
    assert eng.calculate_m_score(cur, pri)["M_Score"] == pytest.approx(-0.084, abs=1e-3)
    assert eng.calculate_z_score(cur)["Z_Score"] == pytest.approx(2.635, abs=1e-3)


def test_sunbeam_cooked_flags_manipulation(eng, cases):
    cur, pri = _pair(cases, "SUNBEAM")
    m = eng.calculate_m_score(cur, pri)
    assert m["M_Score"] == pytest.approx(-1.884, abs=1e-3)
    assert m["Is_Manipulator"] is True
    # Sloan accruals corroborate (net income positive, operating cash flow negative).
    assert eng.calculate_sloan_ratio(cur, pri)["Sloan_Ratio"] == pytest.approx(0.1073, abs=1e-3)


def test_sunbeam_restated_clears(eng, cases):
    """Removing the fraud should drop the M-Score below the -2.22 manipulation threshold."""
    cur, pri = _pair(cases, "SUNBEAM_RESTATED")
    m = eng.calculate_m_score(cur, pri)
    assert m["M_Score"] == pytest.approx(-2.46, abs=1e-2)
    assert m["Is_Manipulator"] is False


def test_enron_flags_manipulation(eng, cases):
    cur, pri = _pair(cases, "ENRON")
    m = eng.calculate_m_score(cur, pri)
    assert m["Is_Manipulator"] is True
    # Sales growth and gross-margin indices should be the dominant signals.
    assert m["Components"]["SGI"] > 2.0


def test_worldcom_is_a_known_m_score_blind_spot(eng, cases):
    """Expense-capitalization fraud is NOT caught by Beneish, but Z-Score flags distress.
    This documents the limitation that motivates the multi-model design."""
    cur, pri = _pair(cases, "WORLDCOM")
    assert eng.calculate_m_score(cur, pri)["Is_Manipulator"] is False
    assert eng.calculate_z_score(cur)["Status"] == "Distress"


def test_lehman_extreme_leverage_high_confidence(eng, cases):
    """Models needing a classified balance sheet are low-confidence for a bank, but the
    leverage flag (needs only assets + equity) catches it at high confidence."""
    cur, pri = _pair(cases, "LEHMAN")
    lev = eng.calculate_leverage(cur, pri)
    assert lev["Equity_Multiplier"] == pytest.approx(30.73, abs=1e-2)
    assert lev["Status"] == "Extreme Leverage"
    assert lev["Confidence"] == "High"
    assert lev["Trend"]["Direction"] == "rising"
    # The bank's partial data should surface as reduced coverage.
    assert cur["_coverage"] < 0.6


# --------------------------------------------------------------------------------------
# 2. Verdict aggregation
# --------------------------------------------------------------------------------------

def _verdict(eng, cur, pri):
    return eng.calculate_verdict(
        m_score=eng.calculate_m_score(cur, pri),
        z_score=eng.calculate_z_score(cur),
        sloan=eng.calculate_sloan_ratio(cur, pri),
        piotroski=eng.calculate_piotroski_f_score(cur, pri),
        leverage=eng.calculate_leverage(cur, pri),
    )


@pytest.mark.parametrize("case_id,expected_emoji", [
    ("SUNBEAM", "RED"),
    ("SUNBEAM_RESTATED", "GREEN"),
    ("ENRON", "RED"),
    ("WORLDCOM", "RED"),
    ("LEHMAN", "RED"),
    ("TEXTBOOK_DEMO", "RED"),
])
def test_verdict_per_case(eng, cases, case_id, expected_emoji):
    cur, pri = _pair(cases, case_id)
    assert _verdict(eng, cur, pri)["Emoji"] == expected_emoji


def test_cooked_vs_restated_verdict_flips(eng, cases):
    """The same company should go RED (fraud) -> GREEN (corrected) once the fraud is removed."""
    cooked_cur, cooked_pri = _pair(cases, "SUNBEAM")
    clean_cur, clean_pri = _pair(cases, "SUNBEAM_RESTATED")
    assert _verdict(eng, cooked_cur, cooked_pri)["Emoji"] == "RED"
    assert _verdict(eng, clean_cur, clean_pri)["Emoji"] == "GREEN"


def test_lehman_verdict_driven_by_leverage_not_invalid_zscore(eng, cases):
    """Lehman is RED, but the low-confidence Z-Score must not be what drives it."""
    cur, pri = _pair(cases, "LEHMAN")
    v = _verdict(eng, cur, pri)
    assert v["Emoji"] == "RED"
    assert any("leverage" in r.lower() for r in v["Reasons"])


def test_verdict_insufficient_data():
    eng = ForensicEngine()
    v = eng.calculate_verdict()  # nothing passed in
    assert v["Verdict"] == "Insufficient Data"
    assert v["Emoji"] == "WHITE"


# --------------------------------------------------------------------------------------
# 3. XBRL resolver + robustness (synthetic SEC companyfacts, no network)
# --------------------------------------------------------------------------------------

def _facts(tag, unit, points, taxonomy="us-gaap"):
    return {"facts": {taxonomy: {tag: {"units": {unit: points}}}}}


def test_resolver_prefers_annual_10k_over_quarterly(eng):
    facts = _facts("Assets", "USD", [
        {"end": "2023-09-30", "val": 900, "fy": 2023, "fp": "Q3", "form": "10-Q", "filed": "2023-10-20"},
        {"end": "2023-12-31", "val": 1000, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
    ])
    val, prov = eng._get_value_for_year(facts, "Assets", 2023)
    assert val == 1000
    assert "10-K" not in prov  # provenance reports tag/unit/end, value comes from the annual point


def test_resolver_accepts_20f_foreign_filer(eng):
    facts = _facts("Assets", "USD", [
        {"end": "2023-12-31", "val": 500, "fy": 2023, "fp": "FY", "form": "20-F", "filed": "2024-04-30"},
    ])
    val, _ = eng._get_value_for_year(facts, "Assets", 2023)
    assert val == 500


def test_resolver_picks_fiscal_year_end_not_comparative(eng):
    """A 10-K embeds prior-year comparatives under the same fy; take the latest period end."""
    facts = _facts("Assets", "USD", [
        {"end": "2022-12-31", "val": 800, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
        {"end": "2023-12-31", "val": 1000, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
    ])
    val, _ = eng._get_value_for_year(facts, "Assets", 2023)
    assert val == 1000


def test_resolver_point_in_time_excludes_later_restatement(eng):
    facts = _facts("Assets", "USD", [
        {"end": "2023-12-31", "val": 1000, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
        {"end": "2023-12-31", "val": 800, "fy": 2023, "fp": "FY", "form": "10-K/A", "filed": "2025-03-01"},
    ])
    # Default: most recently filed wins (the restatement).
    assert eng._get_value_for_year(facts, "Assets", 2023)[0] == 800
    # As-of before the restatement: only the original is visible -> no look-ahead.
    assert eng._get_value_for_year(facts, "Assets", 2023, as_of_date="2024-06-30")[0] == 1000


def test_resolver_frame_fallback(eng):
    """When no clean 10-K/FY tag exists, the calendar frame identifies the annual point."""
    facts = _facts("Assets", "USD", [
        {"end": "2023-12-31", "val": 1234, "fy": 2023, "fp": "FY", "form": "OTHER", "frame": "CY2023Q4I"},
    ])
    val, _ = eng._get_value_for_year(facts, "Assets", 2023)
    assert val == 1234


def test_resolver_profitloss_fallback_for_netincome(eng):
    facts = _facts("ProfitLoss", "USD", [
        {"start": "2023-01-01", "end": "2023-12-31", "val": 42, "fy": 2023,
         "fp": "FY", "form": "20-F", "filed": "2024-04-30"},
    ])
    val, prov = eng._get_value_for_year(facts, "NetIncome", 2023)
    assert val == 42
    assert "ProfitLoss" in prov


def test_resolver_prefers_usd_unit(eng):
    facts = {"facts": {"us-gaap": {"Assets": {"units": {
        "EUR": [{"end": "2023-12-31", "val": 111, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"}],
        "USD": [{"end": "2023-12-31", "val": 222, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"}],
    }}}}}
    val, _ = eng._get_value_for_year(facts, "Assets", 2023)
    assert val == 222


def test_extract_financials_tracks_missing_as_low_confidence(eng):
    """Sparse facts -> low coverage, None provenance, and Low-confidence scores (not silent zeros)."""
    facts = _facts("Assets", "USD", [
        {"end": "2023-12-31", "val": 1000, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-15"},
    ])
    fin = eng.extract_financials(facts, 2023)
    assert fin["Assets"] == 1000
    assert fin["_coverage"] < 0.2
    assert fin["_provenance"]["Sales"] is None
    # M-Score must still compute (no crash) but flag Low confidence.
    fin_prior = eng.extract_financials(facts, 2022)
    m = eng.calculate_m_score(fin, fin_prior)
    assert m["M_Score"] is not None
    assert m["Confidence"] == "Low"


def test_benford_insufficient_data_returns_none(eng):
    facts = _facts("Assets", "USD", [{"end": "2023-12-31", "val": 1000}])
    out = eng.calculate_benford_deviation(facts, 2023)
    assert out["MAD"] is None
    assert out["Status"] == "Insufficient Data"


def test_zscore_missing_assets_is_graceful(eng):
    out = eng.calculate_z_score({"Assets": 0})
    assert out["Z_Score"] is None
    assert out["Status"] == "Missing Data"
