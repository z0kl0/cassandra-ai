"""
Calibration regression tests — lock in the verdict-aggregation and extraction decisions made
while tuning against real blue-chips, so they can't silently drift.

  1. Verdict tiering (synthetic model dicts): which signals are standalone High Risk vs Watch
     vs advisory-only. These encode the deliberate choices that keep MSFT/AAPL/GOOGL clean and
     NVDA at Watch while real frauds stay High Risk.
  2. Extraction hardening (synthetic SEC facts): the PP&E finance-lease tag and the SG&A
     derivation that fixed Alphabet's split SG&A reporting.
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


# ---- 4. Michael's scaling conviction (_bull_posture) ---------------------------------
# Lives in llm.py, which imports langchain_ollama; skip cleanly in the lean CI image where
# that isn't installed. The logic is pure (no Ollama call), so it runs anywhere with the deps.
try:
    from llm import ForensicMemoWriter
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False

pytestmark_llm = pytest.mark.skipif(not _HAS_LLM, reason="llm deps (langchain_ollama) not installed")


def _posture(eng, **overrides):
    base = {"m_score": m(), "z_score": z(), "sloan": sloan(),
            "piotroski": piotroski(), "leverage": lev(), "benford": benford()}
    base.update(overrides)
    verdict = eng.calculate_verdict(**base)
    return ForensicMemoWriter._bull_posture(verdict, base)  # (label, directive, temperature)


@pytestmark_llm
def test_posture_clean_is_confident(eng):
    label, _, temp = _posture(eng)
    assert label == "confident" and temp == 0.5


@pytestmark_llm
def test_posture_watch_is_confident(eng):
    # One moderate signal -> YELLOW Watch -> Michael still confident.
    assert _posture(eng, sloan=sloan(high=True))[0] == "confident"


@pytestmark_llm
def test_posture_red_single_flag_is_cautious(eng):
    # High-confidence distress alone -> RED with one flag -> cautious.
    assert _posture(eng, z_score=z("Distress"))[0] == "cautious"


@pytestmark_llm
def test_posture_manipulator_concedes(eng):
    label, _, temp = _posture(eng, m_score=m(manip=True))
    assert label == "concede" and temp == 0.2


@pytestmark_llm
def test_posture_two_flags_concede(eng):
    # Distress (strong) + high accruals (moderate) = two flags -> concede.
    assert _posture(eng, z_score=z("Distress"), sloan=sloan(high=True))[0] == "concede"


@pytestmark_llm
def test_posture_low_confidence_manipulator_does_not_concede(eng):
    # A low-confidence manipulation flag is only a Watch (verdict downgrades it), so Michael
    # must NOT be forced into outright concession on a weak signal.
    assert _posture(eng, m_score=m(manip=True, conf="Low"))[0] != "concede"


@pytestmark_llm
def test_posture_insufficient_data(eng):
    assert ForensicMemoWriter._bull_posture(eng.calculate_verdict(), {})[0] == "data-limited"


# ---- 5. Cassandra's scaling skepticism (_skeptic_posture) ----------------------------
# Symmetric to Michael: she must NOT manufacture a bear case when the numbers are clean.

def _sk_posture(eng, **overrides):
    base = {"m_score": m(), "z_score": z(), "sloan": sloan(),
            "piotroski": piotroski(), "leverage": lev(), "benford": benford()}
    base.update(overrides)
    verdict = eng.calculate_verdict(**base)
    return ForensicMemoWriter._skeptic_posture(verdict, base)  # (label, directive, temperature)


@pytestmark_llm
def test_skeptic_clean_stands_down(eng):
    # Clean GREEN, no flags -> Cassandra stands down (no manufactured bear case).
    label, directive, _ = _sk_posture(eng)
    assert label == "stand-down"
    assert "no material objection" in directive.lower()


@pytestmark_llm
def test_skeptic_watch_is_measured(eng):
    # One moderate signal -> YELLOW Watch -> measured, not full prosecution.
    assert _sk_posture(eng, sloan=sloan(high=True))[0] == "measured"


@pytestmark_llm
def test_skeptic_red_single_flag_prosecutes(eng):
    assert _sk_posture(eng, z_score=z("Distress"))[0] == "prosecute"


@pytestmark_llm
def test_skeptic_manipulator_prosecutes(eng):
    assert _sk_posture(eng, m_score=m(manip=True))[0] == "prosecute"


@pytestmark_llm
def test_skeptic_insufficient_data(eng):
    assert ForensicMemoWriter._skeptic_posture(eng.calculate_verdict(), {})[0] == "data-limited"


# ---- 6. Grounded model glossary + concise/deep answer cap ----------------------------
# Guards the interrogation Q&A: definitions must be ground truth (no invented formulas), and the
# answer length must adapt to whether the user asked for depth.

@pytestmark_llm
def test_glossary_has_correct_beneish_formula():
    from llm import MODEL_GLOSSARY
    # The real Beneish constants/indices — guards against regressing to a hallucinated formula.
    for token in ("-4.84", "DSRI", "TATA", "LVGI", "-2.22"):
        assert token in MODEL_GLOSSARY


@pytestmark_llm
def test_glossary_has_altman_zones():
    from llm import MODEL_GLOSSARY
    for token in ("1.81", "2.99", "Sloan", "Benford", "Piotroski"):
        assert token in MODEL_GLOSSARY


@pytestmark_llm
def test_answer_cap_scales_with_depth_request():
    plain = ForensicMemoWriter._answer_cap("what is the M-Score?")
    deep = ForensicMemoWriter._answer_cap("give me the full formula and explain it in detail")
    assert deep > plain and plain <= 250


@pytestmark_llm
def test_yes_after_offer_counts_as_depth():
    # The exact loop the user hit: "yes" to an offer must trigger the full answer, not another summary.
    offered = [("user", "what is the Altman Z?"),
               ("assistant", "It gauges distress.\nWant the full formula, or how it applied to this company?")]
    assert ForensicMemoWriter._wants_depth("yes", offered) is True
    assert ForensicMemoWriter._wants_depth("yes please", offered) is True
    # A bare "yes" with no preceding offer is NOT a depth request.
    assert ForensicMemoWriter._wants_depth("yes", [("assistant", "NVDA looks clean.")]) is False
    # An affirmation embedded in a real question still falls through to normal handling.
    assert ForensicMemoWriter._wants_depth("what is it?", offered) is False


@pytestmark_llm
def test_topic_model_locks_to_same_model_across_turns():
    # "give me the formula" with no model named must stay on Altman (the thread's model), not drift.
    hist = [("user", "explain the Altman Z-Score"),
            ("assistant", "The Altman Z-Score for NVDA is 6.568, in the Safe zone.")]
    assert ForensicMemoWriter._topic_model("yes give me the full formula", hist) == "Altman Z-Score"
    # A freshly named model in the current question wins over history.
    assert ForensicMemoWriter._topic_model("what about the M-Score?", hist) == "Beneish M-Score"
    assert ForensicMemoWriter._topic_model("hello", None) is None


# ---- 7. Proportionate host-confirm framing (no presumed "fraud" for healthy companies) ----
# The confirm directive must scale with the verdict tone: clean/watch must forbid the "fraud"/
# alarming framing; only high-risk earns serious language.

@pytestmark_llm
def test_confirm_tone_clean_forbids_fraud_language():
    d = ForensicMemoWriter._CONFIRM_TONE["clean"]
    assert "CLEAN" in d and "fraud" in d.lower()  # it explicitly forbids the word 'fraud'
    assert "do not" in d.lower()


@pytestmark_llm
def test_confirm_tone_watch_is_calm():
    d = ForensicMemoWriter._CONFIRM_TONE["watch"]
    assert "MINOR" in d and ("calm" in d.lower() or "measured" in d.lower())


@pytestmark_llm
def test_confirm_tone_high_risk_is_serious():
    d = ForensicMemoWriter._CONFIRM_TONE["high-risk"]
    assert "SERIOUS" in d and "red flag" in d.lower()


# ---- 8. Mic-safe name handling (per-utterance only; never invented or carried over) ----

@pytestmark_llm
@pytest.mark.parametrize("utterance,expected", [
    ("Hi, my name is Franz, analyze Apple", "Franz"),
    ("I'm Sarah, can you look at Ford?", "Sarah"),
    ("call me Mike", "Mike"),
    ("my name is franz", "Franz"),            # lowercase typed intro still works
    ("analyze Tesla", None),                  # no name -> none (was inventing "John")
    ("I'm looking at Tesla", None),           # stopword guard
    ("I'm interested in Nvidia", None),
    ("what can you do?", None),
])
def test_extract_name(utterance, expected):
    assert ForensicMemoWriter._extract_name(utterance) == expected


@pytestmark_llm
def test_name_directive_forbids_invention_when_absent():
    d = ForensicMemoWriter._name_directive("analyze Tesla")
    assert "do not address" in d.lower() and "invent" in d.lower()
    assert "Franz" not in ForensicMemoWriter._name_directive("analyze Tesla")


@pytestmark_llm
def test_name_directive_uses_current_name():
    assert "Franz" in ForensicMemoWriter._name_directive("hi I'm Franz, analyze Apple")
