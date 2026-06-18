import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sec_client import SECClient
from forensics import ForensicEngine
from curated_cases import CuratedCaseLoader
from llm import ForensicMemoWriter

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="CASSANDRA — Forensic Accountant", page_icon="🔍", layout="wide")

# Verdict color/emoji map (the engine returns a console-safe token; the UI renders the dot).
VERDICT_STYLE = {
    "RED": ("🔴", "#e63946"),
    "YELLOW": ("🟡", "#f4a261"),
    "GREEN": ("🟢", "#2a9d8f"),
    "WHITE": ("⚪", "#6c757d"),
}


# ----------------------------- cached singletons & data --------------------------------

@st.cache_resource
def get_sec():
    return SECClient()


@st.cache_resource
def get_engine():
    return ForensicEngine()


@st.cache_resource
def get_loader():
    return CuratedCaseLoader()


@st.cache_resource
def get_writer():
    return ForensicMemoWriter()


@st.cache_resource
def get_rag():
    from rag import FraudCorpusRAG
    return FraudCorpusRAG()


@st.cache_data(show_spinner=False)
def fetch_facts(ticker: str):
    return get_sec().get_company_facts(ticker)


@st.cache_data(show_spinner=False)
def available_years(source: str, identifier: str):
    """Annual fiscal years (descending) for which the company has data."""
    if source == "live":
        facts = fetch_facts(identifier)
        if not facts:
            return []
        engine = get_engine()
        pts = facts.get("facts", {}).get("us-gaap", {}).get("Assets", {}).get("units", {}).get("USD", [])
        years = {dp["fy"] for dp in pts
                 if dp.get("form") in engine.ANNUAL_FORMS and dp.get("fp") == "FY" and dp.get("fy")}
        return sorted(years, reverse=True)
    return sorted(get_loader().get_company_years(identifier).keys(), reverse=True)


@st.cache_data(show_spinner=False)
def populated_cases():
    """Curated case_ids that have real figures (excludes metadata-only rows), with labels."""
    df = get_loader().df
    have = df.dropna(subset=["Assets"])["case_id"].unique()
    labels = {cid: df[df["case_id"] == cid]["company"].iloc[0] for cid in have}
    return labels


@st.cache_data(show_spinner=False)
def analyze(source: str, identifier: str, cur_year: int, prior_year: int, rag_enabled: bool):
    """Runs the full deterministic pipeline. Returns only picklable dicts (cache-safe)."""
    engine = get_engine()
    benford = None
    if source == "live":
        facts = fetch_facts(identifier)
        company = facts.get("entityName", identifier)
        cur = engine.extract_financials(facts, cur_year)
        pri = engine.extract_financials(facts, prior_year)
        benford = engine.calculate_benford_deviation(facts, cur_year)
    else:
        yrs = get_loader().get_company_years(identifier)
        cur, pri = yrs[cur_year], yrs[prior_year]
        company = populated_cases().get(identifier, identifier)

    models = {
        "m_score": engine.calculate_m_score(cur, pri),
        "z_score": engine.calculate_z_score(cur),
        "sloan": engine.calculate_sloan_ratio(cur, pri),
        "piotroski": engine.calculate_piotroski_f_score(cur, pri),
        "leverage": engine.calculate_leverage(cur, pri),
        "benford": benford,
    }
    verdict = engine.calculate_verdict(**models)

    analogues = None
    if rag_enabled:
        try:
            q = "; ".join(verdict.get("Reasons", [])) or "financial statement fraud"
            res = get_rag().query_cases(q, n_results=2)
            analogues = res.get("documents", [[]])[0] or None
        except Exception as e:  # RAG is optional; never block the dashboard
            logging.warning(f"RAG unavailable: {e}")

    return {"company": company, "cur": cur, "pri": pri, "models": models,
            "verdict": verdict, "analogues": analogues, "cur_year": cur_year, "prior_year": prior_year}


# ----------------------------- presentation helpers ------------------------------------

def gauge(title, value, vmin, vmax, steps, threshold=None, fmt="{:.2f}"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": ".2f"},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [vmin, vmax]},
            "bar": {"color": "rgba(230,237,243,0.35)"},
            "steps": steps,
            "threshold": ({"line": {"color": "#e6edf3", "width": 3}, "thickness": 0.8,
                           "value": threshold} if threshold is not None else None),
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf3"})
    return fig


def render_gauges(models):
    m, z, s = models["m_score"], models["z_score"], models["sloan"]
    p, lev, b = models["piotroski"], models["leverage"], models["benford"]
    row = st.columns(3)
    G, A, R = "#2a9d8f", "#f4a261", "#e63946"  # green / amber / red

    with row[0]:
        if m and m.get("M_Score") is not None:
            st.plotly_chart(gauge("Beneish M-Score", m["M_Score"], -6, 2,
                                  [{"range": [-6, -2.22], "color": G}, {"range": [-2.22, 2], "color": R}],
                                  threshold=-2.22), use_container_width=True)
        else:
            st.metric("Beneish M-Score", "N/A")
    with row[1]:
        if z and z.get("Z_Score") is not None:
            st.plotly_chart(gauge("Altman Z-Score", z["Z_Score"], 0, 6,
                                  [{"range": [0, 1.81], "color": R}, {"range": [1.81, 2.99], "color": A},
                                   {"range": [2.99, 6], "color": G}]), use_container_width=True)
        else:
            st.metric("Altman Z-Score", "N/A")
    with row[2]:
        if s and s.get("Sloan_Ratio") is not None:
            st.plotly_chart(gauge("Sloan Accruals", s["Sloan_Ratio"], -0.2, 0.3,
                                  [{"range": [-0.2, 0.10], "color": G}, {"range": [0.10, 0.3], "color": R}],
                                  threshold=0.10), use_container_width=True)
        else:
            st.metric("Sloan Accruals", "N/A")

    row2 = st.columns(3)
    with row2[0]:
        if p and p.get("F_Score") is not None:
            st.plotly_chart(gauge("Piotroski F-Score", p["F_Score"], 0, 9,
                                  [{"range": [0, 2], "color": R}, {"range": [2, 7], "color": A},
                                   {"range": [7, 9], "color": G}]), use_container_width=True)
        else:
            st.metric("Piotroski F-Score", "N/A")
    with row2[1]:
        if lev and lev.get("Equity_Multiplier") is not None:
            st.plotly_chart(gauge("Leverage (Equity Mult.)", lev["Equity_Multiplier"], 0, 35,
                                  [{"range": [0, 4], "color": G}, {"range": [4, 10], "color": A},
                                   {"range": [10, 20], "color": "#e76f51"}, {"range": [20, 35], "color": R}],
                                  threshold=20), use_container_width=True)
        else:
            st.metric("Leverage", lev.get("Status", "N/A") if lev else "N/A")
    with row2[2]:
        if b and b.get("MAD") is not None:
            st.plotly_chart(gauge("Benford MAD", b["MAD"], 0, 0.03,
                                  [{"range": [0, 0.012], "color": G}, {"range": [0.012, 0.015], "color": A},
                                   {"range": [0.015, 0.03], "color": R}], threshold=0.015,
                                  fmt="{:.4f}"), use_container_width=True)
        else:
            st.info("Benford's Law needs full XBRL filings — available for live tickers only.")


def render_verdict_banner(verdict):
    emoji, color = VERDICT_STYLE.get(verdict.get("Emoji", "WHITE"), ("⚪", "#6c757d"))
    st.markdown(
        f"""<div style="padding:1rem 1.3rem;border-radius:10px;background:{color}1f;
        border-left:7px solid {color};margin-bottom:0.5rem;">
        <span style="font-size:2rem;">{emoji}</span>
        <span style="font-size:1.7rem;font-weight:700;color:{color};"> {verdict.get('Verdict')}</span>
        <span style="opacity:0.75;font-size:1rem;"> &nbsp;|&nbsp; verdict confidence: {verdict.get('Confidence')}</span>
        </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------- app -------------------------------------------

st.title("🔍 CASSANDRA")
st.caption("AI Forensic Accountant — deterministic red-flag models + a cited, local-LLM memo.")
st.warning("Educational analysis, not investment advice. The LLM never invents a number — all "
           "scores are computed deterministically from SEC filings.", icon="⚠️")

with st.sidebar:
    st.header("Analyze a company")
    source = st.radio("Source", ["Live ticker (SEC XBRL)", "Curated demo case"], index=1)
    source_key = "live" if source.startswith("Live") else "curated"

    if source_key == "live":
        identifier = st.text_input("Ticker", value="MSFT").strip().upper()
    else:
        labels = populated_cases()
        ids = sorted(labels)
        identifier = st.selectbox("Demo case", ids, format_func=lambda c: f"{labels[c]} ({c})")

    rag_enabled = st.checkbox("Include historical analogues (RAG)", value=False,
                              help="Retrieves similar historical fraud cases. First use loads the embedding model.")
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

if analyze_clicked:
    st.session_state["active"] = (source_key, identifier)

active = st.session_state.get("active")
if not active:
    st.info("Pick a source in the sidebar and press **Analyze**. Try the `SUNBEAM` vs "
            "`SUNBEAM_RESTATED` demo cases for a clean before/after, or a live ticker like `NVDA`.")
    st.stop()

src, ident = active
years = available_years(src, ident)
if len(years) < 2:
    st.error(f"Need at least two annual reporting years for **{ident}**; found {years or 'none'}. "
             "For live tickers, check the symbol is a US/ADR SEC filer with XBRL data.")
    st.stop()

# Fiscal-year selection (default latest two; prior auto-follows current).
cur_year = st.sidebar.selectbox("Current fiscal year", years, index=0)
lower = [y for y in years if y < cur_year]
prior_year = lower[0] if lower else cur_year - 1

with st.spinner("Running forensic models..."):
    res = analyze(src, ident, cur_year, prior_year, rag_enabled)

st.subheader(f"{res['company']}  ·  FY{res['prior_year']} → FY{res['cur_year']}")
render_verdict_banner(res["verdict"])

cov = res["cur"].get("_coverage")
flags = res["verdict"].get("Flags", [])
c1, c2, c3 = st.columns(3)
c1.metric("Data coverage", f"{cov:.0%}" if cov is not None else "—")
c2.metric("Flags triggered", len(flags))
c3.metric("Source", "Live SEC XBRL" if src == "live" else "Curated CSV")

render_gauges(res["models"])

tab_memo, tab_ev, tab_raw, tab_an = st.tabs(
    ["📝 Forensic Risk Memo", "🔬 Evidence & Data", "🧾 Raw output", "📚 Analogues"])

with tab_memo:
    if flags:
        st.markdown("**Flags:** " + "; ".join(flags))
    memo_key = f"memo::{res['company']}::{res['cur_year']}::{res['verdict'].get('Verdict')}"
    if st.button("Generate memo (local LLM)", type="primary"):
        writer = get_writer()
        try:
            with st.spinner("Loading model / generating (CPU inference can take 1–3 min)..."):
                text = st.write_stream(writer.stream_memo(
                    res["company"], res["verdict"], res["models"], res["cur"], res["analogues"]))
            st.session_state[memo_key] = text
        except Exception as e:
            st.error(f"Could not reach the local LLM (Ollama). Start it and `ollama pull llama3`.\n\n{e}")
    elif st.session_state.get(memo_key):
        st.markdown(st.session_state[memo_key])
    else:
        st.caption("The deterministic verdict above stands on its own. Generate the memo for a "
                   "plain-English, cited explanation.")

with tab_ev:
    st.code(ForensicMemoWriter.build_evidence(
        res["company"], res["verdict"], res["models"], res["cur"], res["analogues"]), language="text")
    prov = res["cur"].get("_provenance", {})
    rows = [{"Concept": k, "Value": res["cur"].get(k), "Source": v}
            for k, v in prov.items() if v is not None]
    if rows:
        st.markdown("**Resolved figures (provenance):**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_raw:
    st.json({"verdict": res["verdict"], "models": res["models"]})

with tab_an:
    if res["analogues"]:
        for a in res["analogues"]:
            st.markdown(f"- {a}")
    elif rag_enabled:
        st.caption("No comparable cases retrieved.")
    else:
        st.caption("Enable 'Include historical analogues (RAG)' in the sidebar and re-analyze.")
