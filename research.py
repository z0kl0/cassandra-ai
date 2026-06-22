"""
Bull-case research for the committee debate — Michael's *asymmetric* ammunition.

The forensic engine is backward-looking (reported ratios). A real bull argues the forward story the
statements can't capture. This module gathers that the way a buy-side analyst would:

  1. fetch_filing_outlook  — management's OWN forward-looking words (latest 10-K/10-Q MD&A),
                             summarized by the LLM into a short outlook brief.
  2. gather_news           — targeted DuckDuckGo queries around bull-thesis pillars (not "X news").
  3. curate_bull_facts     — an LLM "research analyst" pass that SIFTS the raw snippets down to a few
                             specific, plausible, on-thesis facts and discards vague/promotional noise.
  4. build_bull_brief      — combines both into one labeled block + sources.

INTEGRITY: everything here is forward-looking / external context, explicitly UNVERIFIED, and feeds
Michael's *argument only*. It never touches the deterministic scores or the verdict. All LLM calls go
through ForensicMemoWriter.complete (the single resident model, serialized — one task at a time).
"""
import re
import html
import logging
from urllib.parse import urlparse

from sec_client import SECClient

logger = logging.getLogger(__name__)

# Targeted, thesis-pillar queries (a great advisor gathers around a thesis, not generic headlines).
NEWS_QUERY_TEMPLATES = [
    "{c} latest quarterly revenue guidance growth",
    "{c} competitive moat market share advantage",
    "{c} demand outlook analyst estimates",
    "{c} new product launch catalyst",
    "{c} bear case risks concerns",  # so Michael can pre-empt the bear
]

FILING_OUTLOOK_SYSTEM = """You summarize management's FORWARD-LOOKING outlook from an MD&A excerpt of a
company's SEC filing. Extract concise bullets on growth drivers, demand, guidance, segment trends and
stated opportunities — the qualitative story behind the numbers. Use ONLY the provided text; do not
invent figures. Under 120 words, bullets only, no preamble."""

CURATION_SYSTEM = """You are a sharp buy-side research analyst building a BULL case. From the raw web
search results below, select the 3-5 MOST useful facts: specific, recent, plausible, and relevant to
the bull thesis (forward demand, guidance, moat, catalysts) — ideally ones that help rebut the listed
skeptic flags. DISCARD anything vague, promotional, dated, or unverifiable. Do not invent facts; only
use what the snippets support. Output 3-5 short bullets, each ending with its source domain in
parentheses. No preamble, no numbers you cannot support from the snippets."""


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _latest_filing_url(sec: SECClient, ticker: str):
    """(url, form) of the most recent 10-K/10-Q primary document, or None."""
    cik = sec._get_cik(ticker)
    if not cik:
        return None
    try:
        data = sec.session.get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    except Exception as e:
        logger.warning(f"submissions fetch failed for {ticker}: {e}")
        return None
    rec = data.get("filings", {}).get("recent", {})
    for form, accn, doc in zip(rec.get("form", []), rec.get("accessionNumber", []),
                               rec.get("primaryDocument", [])):
        if form in ("10-K", "10-Q") and doc:
            return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accn.replace('-', '')}/{doc}", form)
    return None


def _extract_mdna(raw_html: str, limit: int = 7000) -> str:
    """Strip tags and return a bounded MD&A excerpt focused on the actual discussion.

    Anchors on 'Results of Operations' / 'Overview' (the substance of MD&A) when present, to skip
    the forward-looking-statements / risk boilerplate that precedes it; falls back to the MD&A
    header (last occurrence = body, not the table of contents)."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw_html))
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"[ \t\r\n]+", " ", text)

    for anchor in (r"Results of Operations", r"\bOverview\b"):
        hits = [m.start() for m in re.finditer(anchor, text, re.I)]
        if hits:
            start = hits[-1]
            return text[start:start + limit].strip()

    hits = [m.start() for m in re.finditer(r"Management.{0,5}s Discussion and Analysis", text, re.I)]
    if not hits:
        return ""
    return text[hits[-1]:hits[-1] + limit].strip()


def fetch_filing_outlook(ticker: str, writer, sec: SECClient = None) -> dict:
    """LLM summary of the latest filing's MD&A outlook, or None. {summary, form, url}."""
    sec = sec or SECClient()
    found = _latest_filing_url(sec, ticker)
    if not found:
        return None
    url, form = found
    try:
        resp = sec.session.get(url)
        if resp.status_code != 200:
            return None
        excerpt = _extract_mdna(resp.text)
    except Exception as e:
        logger.warning(f"filing fetch failed for {ticker}: {e}")
        return None
    if len(excerpt) < 400:
        return None
    summary = writer.complete(FILING_OUTLOOK_SYSTEM, excerpt)
    return {"summary": summary, "form": form, "url": url}


def gather_news(company: str, max_per_query: int = 4) -> list:
    """Targeted DuckDuckGo searches → de-duplicated [{title, url, snippet}]."""
    from ddgs import DDGS  # lazy: only when research is actually run
    candidates, seen = [], set()
    try:
        with DDGS() as ddg:
            for tmpl in NEWS_QUERY_TEMPLATES:
                try:
                    for r in ddg.text(tmpl.format(c=company), max_results=max_per_query):
                        url = r.get("href", "")
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        candidates.append({"title": r.get("title", ""), "url": url,
                                           "snippet": r.get("body", "")})
                except Exception as e:
                    logger.warning(f"news query failed ({tmpl}): {e}")
    except Exception as e:
        logger.warning(f"DDGS unavailable: {e}")
    return candidates


def curate_bull_facts(company: str, candidates: list, skeptic_flags: list, writer) -> str:
    """LLM analyst pass: sift raw snippets to 3-5 vetted bull facts. Returns bullet text or ''."""
    if not candidates:
        return ""
    blob = "\n".join(f"- {c['title']}: {c['snippet']} (source: {_domain(c['url'])})"
                     for c in candidates[:25])
    flags = "; ".join(skeptic_flags) if skeptic_flags else "none"
    user = f"COMPANY: {company}\nSKEPTIC FLAGS TO HELP REBUT: {flags}\n\nRAW SEARCH RESULTS:\n{blob}"
    return writer.complete(CURATION_SYSTEM, user)


def build_bull_brief(ticker: str, company: str, skeptic_flags: list, writer,
                     sec: SECClient = None, include_news: bool = True) -> dict:
    """
    Assemble Michael's brief: management filing outlook + curated news facts. Returns
    {brief, sources} or None if nothing could be gathered. LLM/network heavy — caller should cache.
    """
    sec = sec or SECClient()
    parts, sources = [], []

    outlook = fetch_filing_outlook(ticker, writer, sec)
    if outlook:
        parts.append(f"MANAGEMENT OUTLOOK (from latest {outlook['form']} MD&A):\n{outlook['summary']}")
        sources.append(outlook["url"])

    if include_news:
        candidates = gather_news(company)
        curated = curate_bull_facts(company, candidates, skeptic_flags, writer)
        if curated:
            parts.append("CURATED MARKET/NEWS CONTEXT (external, UNVERIFIED — argument only):\n" + curated)
            sources += [c["url"] for c in candidates[:8]]

    if not parts:
        return None
    return {"brief": "\n\n".join(parts), "sources": sources}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    from llm import ForensicMemoWriter

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    sec = SECClient()
    company = (sec.get_company_facts(ticker) or {}).get("entityName", ticker)
    brief = build_bull_brief(ticker, company, ["Sloan accruals 0.109 > 0.10 (low earnings quality)"],
                             ForensicMemoWriter(), sec)
    if not brief:
        print(f"No bull brief assembled for {ticker}.")
    else:
        print(f"\n===== BULL BRIEF: {company} =====\n")
        print(brief["brief"])
        print("\nSOURCES:")
        for s in brief["sources"]:
            print(" -", s)
