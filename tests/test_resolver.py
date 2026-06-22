"""
Offline tests for the company-name -> ticker resolver. Uses an injected fixture map (no SEC
network) and works with either rapidfuzz or the difflib fallback (so it passes in lean CI too).
Typo/description correction is the LLM's job (llm.interpret) and isn't asserted here — the resolver's
job is exact/name/substring matching + validating the LLM's ticker guess.
"""
from ticker_resolver import TickerResolver

FIXTURE = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "3": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
    "4": {"cik_str": 1418121, "ticker": "APLE", "title": "Apple Hospitality REIT, Inc."},
}


def _r():
    return TickerResolver(ticker_map=FIXTURE)


def test_exact_ticker():
    top, ambiguous = _r().best_match("AAPL")
    assert top["ticker"] == "AAPL" and top["score"] == 100.0 and ambiguous is False


def test_exact_title_unambiguous():
    top, ambiguous = _r().best_match("Apple Inc.")
    assert top["ticker"] == "AAPL" and ambiguous is False


def test_name_substring_picks_main_company():
    # "apple" matches both Apple Inc. and Apple Hospitality; the tighter title wins.
    top, _ = _r().best_match("apple")
    assert top["ticker"] == "AAPL"
    tickers = {c["ticker"] for c in _r().resolve("apple", limit=5)}
    assert {"AAPL", "APLE"} <= tickers


def test_ticker_guess_validated_handles_common_name():
    # Google's legal title is "Alphabet Inc." — the validated LLM ticker_guess bridges that.
    top, ambiguous = _r().best_for("Google", "GOOGL")
    assert top["ticker"] == "GOOGL" and ambiguous is False


def test_invalid_ticker_guess_falls_back_to_name():
    # A bogus ticker must NOT be trusted; resolution falls back to the company name.
    top, _ = _r().best_for("Microsoft", "ZZZZ")
    assert top["ticker"] == "MSFT"


def test_unknown_company_is_none_or_flagged():
    # Robust across rapidfuzz/difflib: a nonsense query is either unresolved or flagged uncertain.
    top, ambiguous = _r().best_match("zzqq nonexistent holdings")
    assert top is None or ambiguous is True


def test_name_lookup_helper():
    assert _r().name("nvda") == "NVIDIA CORP"
