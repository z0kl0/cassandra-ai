"""
Calibration regression tests — lock in the verdict-aggregation and extraction decisions made
while tuning against real blue-chips, so they can't silently drift.

  1. Verdict tiering (synthetic model dicts): which signals are standalone High Risk vs Watch
     vs advisory-only. These encode the deliberate choices that keep MSFT/AAPL/GOOGL clean and
     NVDA at Watch while real frauds stay High Risk.
  2. Extraction hardening (synthetic SEC facts): the PP&E finance-lease tag and the SG&A
     derivation that fixed Alphabet/Luckin.
  3. Live blue-chip smoke test (network, opt-in via `pytest --run-live`): the real "blue-chips
     stay calm" guarantee against current SEC data.
"""
import pytest

from forensics import ForensicEngine


@pytest.fixture(scope="module")
def eng():
    return ForensicEngine()


# ---- synthetic model-result factories (non-firing "clean" defaults) -------------------

def m(manip=False, conf="High"):
    return {"M_Score": -1.0 if manip else -3.0, "Is_Manipulator": manip, "Confidence": conf}


def z(status="Safe", conf="High"):
    val = {"Safe": 4.0, "Grey Zone": 2.4, "Distress": 1.0}[status]
    return {"Z_Score": val, "Status": status, "Confidence": conf}


def sloan(high=False, conf="High"):
    return {"Sloan_Ratio": 0.15 if high else 0.0,
            "Status": "High Risk (High Accruals)" if high else "Normal", "Confidence": conf}


def piotroski(status="Neutral"):
    return {"F_Score": {"Weak": 1, "Neutral": 6, "Strong": 8}[status], "Status": status}


def lev(status="Conservative"):
    em = {"Conservative": 2.0, "Extreme Leverage": 30.0}[status]
    return {"Equity_Multiplier": em, "Status": status, "Confidence": "High", "Trend": None}


def benford(bad=False):
    return {"MAD": 0.02 if bad else 0.008,
            "Status": "Nonconformity (High Risk)" if bad else "Conforms"}


def _verdict(eng, **overrides):
    base = {"m_score": m(), "z_score": z(), "sloan": sloan(),
            "piotroski": piotroski(), "leverage": lev(), "benford": benford()}
    base.update(overrides)
    return eng.calculate_verdict(**base)


# ---- 1. Verdict tiering ---------------------------------------------------------------

def test_all_clean_is_green(eng):
    assert _verdict(eng)["Emoji"] == "GREEN"


def test_m_manipulator_alone_is_red(eng):
    assert _verdict(eng, m_score=m(manip=True))["Emoji"] == "RED"


def test_z_distress_alone_is_red(eng):
    assert _verdict(eng, z_score=z("Distress"))["Emoji"] == "RED"


def test_extreme_leverage_alone_is_red(eng):
    assert _verdict(eng, leverage=lev("Extreme Leverage"))["Emoji"] == "RED"


def test_sloan_high_alone_is_only_watch(eng):
    """Earnings-quality accruals alone (NVDA pattern) must be Watch, not High Risk."""
    assert _verdict(eng, sloan=sloan(high=True))["Emoji"] == "YELLOW"


def test_benford_nonconformity_alone_does_not_escalate(eng):
    """Single-filer Benford is advisory only (MSFT/AAPL pattern) -> stays Clean."""
    v = _verdict(eng, benford=benford(bad=True))
    assert v["Emoji"] == "GREEN"
    assert any("Benford" in n for n in v["Notes"])


def test_grey_zone_alone_does_not_escalate(eng):
    """Altman grey zone is a note, not a flag (protects cash-rich tech from false Watch)."""
    assert _verdict(eng, z_score=z("Grey Zone"))["Emoji"] == "GREEN"


def test_two_moderate_signals_escalate_to_red(eng):
    assert _verdict(eng, sloan=sloan(high=True), piotroski=piotroski("Weak"))["Emoji"] == "RED"


def test_low_confidence_strong_signal_is_downgraded(eng):
    """A low-confidence manipulation signal drops to Watch, not High Risk."""
    assert _verdict(eng, m_score=m(manip=True, conf="Low"))["Emoji"] == "YELLOW"


# ---- 2. Extraction hardening (synthetic SEC facts) -----------------------------------

def _annual_instant(val, year):
    return {"end": f"{year}-12-31", "val": val, "fy": year, "fp": "FY",
            "form": "10-K", "filed": f"{year + 1}-02-15"}


def _annual_flow(val, year):
    return {"start": f"{year}-01-01", "end": f"{year}-12-31", "val": val, "fy": year,
            "fp": "FY", "form": "10-K", "filed": f"{year + 1}-02-15"}


def _facts(tag_points: dict):
    return {"facts": {"us-gaap": {t: {"units": {"USD": pts}} for t, pts in tag_points.items()}}}


def test_ppe_finance_lease_tag_resolves(eng):
    """Alphabet's FY2025 retag: PP&E under the finance-lease ROU tag must resolve."""
    facts = _facts({
        "Assets": [_annual_instant(1000, 2025)],
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization":
            [_annual_instant(400, 2025)],
    })
    fin = eng.extract_financials(facts, 2025)
    assert fin["PropertyPlantEquipment"] == 400


def test_sga_derivation_sums_gna_and_marketing(eng):
    facts = _facts({
        "Assets": [_annual_instant(1000, 2025)],
        "GeneralAndAdministrativeExpense": [_annual_flow(30, 2025)],
        "SellingAndMarketingExpense": [_annual_flow(20, 2025)],
    })
    fin = eng.extract_financials(facts, 2025)
    assert fin["SGA"] == 50
    assert "derived" in fin["_provenance"]["SGA"]


def test_sga_derivation_gna_only(eng):
    """When only G&A is reported (no S&M), SG&A falls back to G&A alone."""
    facts = _facts({
        "Assets": [_annual_instant(1000, 2025)],
        "GeneralAndAdministrativeExpense": [_annual_flow(30, 2025)],
    })
    assert eng.extract_financials(facts, 2025)["SGA"] == 30


def test_direct_sga_tag_beats_derivation(eng):
    """A reported combined SG&A tag takes priority over the G&A+S&M derivation."""
    facts = _facts({
        "Assets": [_annual_instant(1000, 2025)],
        "SellingGeneralAndAdministrativeExpense": [_annual_flow(99, 2025)],
        "GeneralAndAdministrativeExpense": [_annual_flow(30, 2025)],
    })
    fin = eng.extract_financials(facts, 2025)
    assert fin["SGA"] == 99
    assert "derived" not in (fin["_provenance"]["SGA"] or "")


# ---- 3. Live blue-chip smoke test (opt-in: pytest --run-live) ------------------------

@pytest.mark.live
@pytest.mark.parametrize("ticker", ["MSFT", "AAPL", "GOOGL", "JNJ"])
def test_bluechip_stays_calm_live(eng, ticker):
    from sec_client import SECClient
    facts = SECClient().get_company_facts(ticker)
    assert facts, f"no SEC facts for {ticker}"
    pts = facts["facts"]["us-gaap"]["Assets"]["units"]["USD"]
    yrs = sorted({dp["fy"] for dp in pts if dp.get("form") in eng.ANNUAL_FORMS
                  and dp.get("fp") == "FY" and dp.get("fy")}, reverse=True)
    cur = eng.extract_financials(facts, yrs[0])
    pri = eng.extract_financials(facts, yrs[1])
    mscore = eng.calculate_m_score(cur, pri)
    v = eng.calculate_verdict(
        m_score=mscore, z_score=eng.calculate_z_score(cur), sloan=eng.calculate_sloan_ratio(cur, pri),
        piotroski=eng.calculate_piotroski_f_score(cur, pri), leverage=eng.calculate_leverage(cur, pri),
        benford=eng.calculate_benford_deviation(facts, yrs[0]))
    assert v["Emoji"] != "RED", f"{ticker} unexpectedly RED: {v['Flags']}"
    assert mscore["Is_Manipulator"] is not True, f"{ticker} flagged manipulator: {mscore}"
    assert cur["_coverage"] >= 0.85, f"{ticker} low coverage {cur['_coverage']}"
