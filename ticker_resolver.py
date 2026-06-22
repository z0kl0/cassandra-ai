"""
Company-name -> ticker resolution over the SEC company map.

The SEC `company_tickers.json` (cached by sec_client) already lists every analyzable filer
(ticker + title + cik) — ~10k companies — so it doubles as the lookup DB AND the analyzable
universe. This resolves a user's free-text company name/ticker to candidates:
  exact ticker  ->  exact title  ->  substring  ->  fuzzy (rapidfuzz, difflib fallback).

For descriptive queries ("the AI chip company") the LLM proposes a company *name* (llm.interpret);
that name is then run through here so the final ticker is always SEC ground truth, never invented.
"""
import logging

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except Exception:  # CI / minimal installs fall back to stdlib difflib
    _HAS_RAPIDFUZZ = False
    import difflib


class TickerResolver:
    """Resolves company names/tickers against the SEC map. Pass `ticker_map` for offline tests."""

    def __init__(self, ticker_map: dict = None):
        if ticker_map is None:
            from sec_client import SECClient
            ticker_map = SECClient()._load_ticker_map() or {}
        self.entries = [
            {"ticker": str(v["ticker"]).upper(), "title": str(v["title"]), "cik": v.get("cik_str")}
            for v in ticker_map.values() if v.get("ticker") and v.get("title")
        ]
        self.by_ticker = {e["ticker"]: e for e in self.entries}
        self._titles = [e["title"] for e in self.entries]

    def name(self, ticker: str) -> str:
        e = self.by_ticker.get(str(ticker).upper())
        return e["title"] if e else str(ticker).upper()

    def resolve(self, query: str, limit: int = 5) -> list:
        """Ranked candidates: [{ticker, title, cik, score(0-100)}]. Higher = better."""
        q = (query or "").strip()
        if not q:
            return []
        scores = {}  # ticker -> (score, entry)

        def bump(entry, score):
            t = entry["ticker"]
            if t not in scores or score > scores[t][0]:
                scores[t] = (float(score), entry)

        # 1. Exact ticker.
        if q.upper() in self.by_ticker:
            bump(self.by_ticker[q.upper()], 100)

        ql = q.lower()
        # 2. Exact / substring title (prefer shorter titles = tighter match).
        for e in self.entries:
            tl = e["title"].lower()
            if tl == ql:
                bump(e, 100)
            elif ql in tl:
                bump(e, 90 + min(8, len(ql) / max(len(tl), 1) * 8))

        # 3. Fuzzy.
        if _HAS_RAPIDFUZZ:
            for _, score, idx in process.extract(q, self._titles, scorer=fuzz.WRatio, limit=limit * 4):
                bump(self.entries[idx], score)
        else:
            for title in difflib.get_close_matches(q, self._titles, n=limit * 4, cutoff=0.5):
                idx = self._titles.index(title)
                ratio = difflib.SequenceMatcher(None, ql, title.lower()).ratio() * 100
                bump(self.entries[idx], ratio)

        ranked = sorted(scores.values(), key=lambda x: -x[0])[:limit]
        return [{"ticker": e["ticker"], "title": e["title"], "cik": e["cik"], "score": round(s, 1)}
                for s, e in ranked]

    def best_match(self, query: str):
        """Returns (top_candidate | None, ambiguous: bool). Ambiguous → show a confirm step."""
        cands = self.resolve(query, limit=3)
        if not cands:
            return None, False
        top = cands[0]
        runner_up = cands[1]["score"] if len(cands) > 1 else 0
        ambiguous = top["score"] < 90 or (top["score"] < 100 and top["score"] - runner_up < 8)
        return top, ambiguous

    def best_for(self, company_query: str, ticker_guess: str = None):
        """Resolve an interpreted request. Prefers the LLM's ticker_guess ONLY if it's a real SEC
        ticker (validated — handles common-name != legal-title like Google->Alphabet/GOOGL); else
        fuzzy-matches the company name. Returns (candidate | None, ambiguous: bool)."""
        if ticker_guess:
            entry = self.by_ticker.get(str(ticker_guess).upper())
            if entry:
                return ({"ticker": entry["ticker"], "title": entry["title"],
                         "cik": entry["cik"], "score": 100.0}, False)
        return self.best_match(company_query or "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    r = TickerResolver()
    print(f"Indexed {len(r.entries)} SEC filers (rapidfuzz={_HAS_RAPIDFUZZ})\n")
    for q in ["AAPL", "apple", "microsft", "nvidia", "alphabet", "coca cola", "the iPhone maker"]:
        top, amb = r.best_match(q)
        tag = "  [ambiguous]" if amb else ""
        print(f"{q:18} -> {top['ticker'] if top else None} ({top['title'] if top else '-'}) "
              f"score={top['score'] if top else '-'}{tag}")
