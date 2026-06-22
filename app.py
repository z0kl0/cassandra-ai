import logging
import os
import json
import time
import base64
import hashlib
import importlib.util

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import voice
import demo_store
import narration_cache
from sec_client import SECClient
from forensics import ForensicEngine
from curated_cases import CuratedCaseLoader
from llm import ForensicMemoWriter

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="CASSANDRA — Forensic Accountant", page_icon="🔍", layout="wide")

# Show the mic in the input bar whenever speech-to-text is installed (no env var needed). Recording is
# native Streamlit; only transcription needs faster-whisper, so this lightweight check gates the mic.
MIC_AVAILABLE = importlib.util.find_spec("faster_whisper") is not None

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


@st.cache_resource
def get_resolver():
    from ticker_resolver import TickerResolver
    return TickerResolver()


@st.cache_data(show_spinner=False)
def fetch_facts(ticker: str):
    return get_sec().get_company_facts(ticker)


@st.cache_data(show_spinner=False)
def get_bull_brief(ticker: str, company: str, flags: tuple):
    """Assemble + cache Michael's forward-looking research (filings + curated news) per ticker."""
    import research
    return research.build_bull_brief(ticker, company, list(flags), get_writer(), get_sec())


@st.cache_data(show_spinner=False)
def available_years(source: str, identifier: str):
    """Annual fiscal years (descending) with usable data, in the filing's reporting currency."""
    if source == "live":
        facts = fetch_facts(identifier)
        if not facts:
            return []
        engine = get_engine()
        cur = engine._detect_currency(facts)
        pts = facts.get("facts", {}).get("us-gaap", {}).get("Assets", {}).get("units", {}).get(cur, [])
        years = {dp["fy"] for dp in pts
                 if dp.get("form") in engine.ANNUAL_FORMS and dp.get("fp") == "FY" and dp.get("fy")}
        return sorted(years, reverse=True)
    return sorted(get_loader().get_company_years(identifier).keys(), reverse=True)


@st.cache_data(show_spinner=False)
def analyzable(ticker: str):
    """Pre-flight for a live ticker. Returns (ok, currency, reason). reason in
    {ifrs, no_usgaap, few_years, not_found} when not ok."""
    facts = fetch_facts(ticker)
    if not facts:
        return False, None, "not_found"
    engine = get_engine()
    cur = engine._detect_currency(facts)
    years = available_years("live", ticker)
    if len(years) >= 2:
        return True, cur, None
    fx = facts.get("facts", {})
    if not fx.get("us-gaap", {}).get("Assets"):
        return False, cur, ("ifrs" if fx.get("ifrs-full") else "no_usgaap")
    return False, cur, "few_years"


# Verdict tier -> the tone word the host confirm uses to frame the result proportionately.
_TONE_BY_EMOJI = {"GREEN": "clean", "YELLOW": "watch", "RED": "high-risk"}


def verdict_tone(source: str, identifier: str):
    """One-word tone (clean/watch/high-risk) for the host confirm, from the deterministic verdict on
    the latest two years. Reuses the cached `analyze`; returns None if it can't be computed cheaply."""
    try:
        years = available_years(source, identifier)
        if len(years) < 2:
            return None
        res = analyze(source, identifier, years[0], years[1], False)
        return _TONE_BY_EMOJI.get(res["verdict"].get("Emoji"))
    except Exception:
        return None


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


# --- "clear" command: reset the UI between demo users ----------------------------------
# Wipes the conversation, the active analysis, and any generated memo/debate/research artifacts.
# Cached SEC/LLM singletons and saved demo files on disk are left intact.
_CLEAR_KEYS = ("active", "active_ctx", "host_pending", "host_history", "last_audio_hash",
               "pending_voice_text")
_CLEAR_PREFIXES = ("memo::", "debate::", "voice_edit_")
_CLEAR_WORDS = {"clear", "reset", "/clear", "clear all", "clear chat", "start over"}


def clear_session():
    """Remove all chat bubbles, the active analysis, and generated artifacts from session state."""
    for k in list(st.session_state.keys()):
        if k in _CLEAR_KEYS or k.startswith(_CLEAR_PREFIXES):
            del st.session_state[k]


def render_narration(debate_key, transcript):
    """Synthesize (cached) per-turn clips and render an in-order, click-to-play browser audio queue.
    One <audio> element chained via onended → correct order, no overlap; plays in the viewer's browser
    (so it works over Tailscale / into Zoom)."""
    if not transcript:
        return
    try:
        with st.spinner("Preparing narration…"):
            clips = narration_cache.get_or_build(debate_key, transcript)
    except Exception as e:
        st.warning(f"Narration unavailable: {e}")
        return
    srcs = ["data:audio/wav;base64," + base64.b64encode(c).decode() for c in clips]
    components.html(f"""
      <button id="play" style="font:600 14px sans-serif;padding:6px 14px;border-radius:8px;
        border:1px solid #888;background:#f4a261;cursor:pointer;">▶ Play debate aloud</button>
      <span id="status" style="font:13px sans-serif;margin-left:10px;color:#888;"></span>
      <script>
        const srcs = {json.dumps(srcs)};
        const GAP_MS = 700;   // pause between speakers
        const audio = new Audio();
        let i = 0;
        const btn = document.getElementById('play');
        const status = document.getElementById('status');
        function playNext() {{
          if (i >= srcs.length) {{ btn.textContent='▶ Play debate aloud'; status.textContent=''; i=0; return; }}
          status.textContent = 'Speaking ' + (i + 1) + ' / ' + srcs.length;
          audio.src = srcs[i++];
          audio.play();
        }}
        audio.onended = () => setTimeout(playNext, GAP_MS);
        btn.onclick = () => {{ audio.pause(); i = 0; btn.textContent = '⏸ Playing…'; playNext(); }};
      </script>
    """, height=60)


def _animate_transcript(transcript):
    """Replay a saved debate with a quick word-by-word reveal (so a cached run still feels live)."""
    for speaker, text in transcript:
        ph = st.chat_message(speaker, avatar=DEBATE_AVATARS.get(speaker, "🗣️")).empty()
        shown = ""
        for word in text.split(" "):
            shown += word + " "
            ph.markdown(shown)
            time.sleep(0.015)   # tune for reveal speed
        ph.markdown(text)


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
st.caption("AI Forensic Accountant — a local LLM writes the memo, debates the bull case, and "
           "answers your questions, all grounded in deterministic red-flag models.")
st.warning("Educational analysis, not investment advice. The LLM never invents a number — all "
           "scores are computed deterministically from SEC filings.", icon="⚠️")

# ----------------------------- Conversational host (talk to Cassandra) ------------------
st.markdown("#### 💬 Ask Cassandra")
st.caption("Say a company by name — you don't need the ticker. e.g. *\"analyze Apple\"*, "
           "*\"is Nvidia cooking the books?\"*, or *\"what can you do?\"*  ·  Type **clear** to reset.")
host_hist = st.session_state.setdefault("host_history", [])
for _role, _text in host_hist[-8:]:
    st.chat_message(_role, avatar="🔍" if _role == "assistant" else None).markdown(_text)

# A submitted turn streams into this container (which sits ABOVE the input bar) so it appears above the
# chat box right away, instead of rendering below it and jumping up after processing.
live_area = st.container()

# Optional "read aloud" of Cassandra's latest reply (gated; voice INPUT is on the chat bar below).
if os.getenv("ENABLE_VOICE", "false").lower() == "true" and host_hist and host_hist[-1][0] == "assistant":
    if st.button("🔊 Read last answer aloud"):
        try:
            with st.spinner("Synthesizing speech..."):
                st.audio(voice.synthesize(host_hist[-1][1]))
        except Exception as e:
            st.error(f"Voice synthesis unavailable: {e}")

# Disambiguation buttons from a prior ambiguous resolve.
_pending = st.session_state.get("host_pending")
if _pending:
    st.caption("Did you mean:")
    for _c in _pending:
        if st.button(f"{_c['ticker']} — {_c['title']}", key="pick_" + _c["ticker"]):
            st.session_state["active"] = ("live", _c["ticker"])
            host_hist.append(("assistant", f"Pulling up **{_c['title']} ({_c['ticker']})** below."))
            st.session_state.pop("host_pending", None)
            st.rerun()

# Input bar: mic to the LEFT of the text box (small gap), with the box on the right.
# The mic uses a rotating key so we can reset it to a clean state after each recording (otherwise the
# widget re-renders the consumed clip and shows Streamlit's "error / reload" boundary).
_audio = None
_mic_key = f"mic_{st.session_state.get('mic_nonce', 0)}"
if MIC_AVAILABLE:
    # Strip the audio widget down to just the record/stop button — hide the waveform & timecode.
    st.markdown(
        """<style>
        [data-testid="stAudioInputWaveSurfer"],
        [data-testid="stAudioInputWaveformTimeCode"] { display: none !important; }
        </style>""",
        unsafe_allow_html=True,
    )
    _c_mic, _c_in = st.columns([1, 9], gap="small", vertical_alignment="bottom")
    with _c_mic:
        _audio = st.audio_input("Speak", label_visibility="collapsed", key=_mic_key)
    with _c_in:
        _prompt = st.chat_input("Ask Cassandra…")
else:
    _prompt = st.chat_input("Ask Cassandra…")

# A new recording: transcribe it into an EDITABLE draft (do NOT auto-send), then rotate the mic key
# and rerun so the widget returns to a fresh mic (no leftover clip / error boundary / manual reload).
if _audio is not None and not _prompt:
    _bytes = _audio.getvalue()
    _h = hashlib.md5(_bytes).hexdigest()
    if st.session_state.get("last_audio_hash") != _h:   # transcribe each recording once
        st.session_state["last_audio_hash"] = _h
        try:
            with st.spinner("Transcribing…"):
                _text = voice.transcribe(_bytes)
            if _text:
                st.session_state["pending_voice_text"] = _text   # prefill the editable draft
        except Exception as e:
            st.error(f"Voice transcription unavailable: {e}")
        st.session_state["mic_nonce"] = st.session_state.get("mic_nonce", 0) + 1
        st.rerun()

# Editable review of the transcript: the user can correct it before sending it to Cassandra.
# (st.chat_input can't be pre-filled, so the draft lives in a text_input with a Send button.)
if st.session_state.get("pending_voice_text") and not _prompt:
    st.caption("🎤 Heard you — edit if needed, then press Send:")
    _rc1, _rc2, _rc3 = st.columns([8, 1, 1], gap="small", vertical_alignment="bottom")
    with _rc1:
        _edited = st.text_input(
            "Voice message", value=st.session_state["pending_voice_text"],
            key=f"voice_edit_{st.session_state.get('mic_nonce', 0)}", label_visibility="collapsed")
    with _rc2:
        if st.button("Send", type="primary", use_container_width=True):
            _prompt = (_edited or "").strip()
            st.session_state.pop("pending_voice_text", None)
    with _rc3:
        if st.button("✕", help="Discard", use_container_width=True):
            st.session_state.pop("pending_voice_text", None)
            st.rerun()

if _prompt and _prompt.strip().lower() in _CLEAR_WORDS:
    clear_session()
    st.toast("Cleared — fresh start. Name a company to begin.", icon="🧹")
    st.rerun()

if _prompt:
    host_hist.append(("user", _prompt))
    live_area.chat_message("user").markdown(_prompt)
    intent = get_writer().interpret(_prompt, has_active=bool(st.session_state.get("active")))
    kind = intent.get("intent")
    if kind == "analyze":
        cand, ambiguous = get_resolver().best_for(intent.get("company_query"), intent.get("ticker_guess"))
        if cand and not ambiguous:
            ok, currency, reason = analyzable(cand["ticker"])
            if ok:
                st.session_state["active"] = ("live", cand["ticker"])
                cur_note = "" if currency in (None, "USD") else f" (figures are in {currency})"
                tone = verdict_tone("live", cand["ticker"])  # proportionate framing in the confirm
                try:
                    with live_area.chat_message("assistant", avatar="🔍"):
                        _reply = st.write_stream(get_writer().stream_host_confirm(
                            _prompt, cand["title"] + cur_note, cand["ticker"], host_hist, tone))
                except Exception:  # Ollama down -> still proceed with a canned confirmation
                    _reply = f"On it — pulling up **{cand['title']} ({cand['ticker']})**{cur_note} below."
                host_hist.append(("assistant", _reply))
                st.rerun()
            else:  # resolved, but unanalyzable (IFRS-only / no us-gaap / too few years)
                try:
                    with live_area.chat_message("assistant", avatar="🔍"):
                        _reply = st.write_stream(get_writer().stream_host_decline(
                            _prompt, cand["title"], reason, host_hist))
                except Exception:
                    _reply = (f"Sorry — I can't analyze **{cand['title']} ({cand['ticker']})** right "
                              "now. Try a US-listed company like Apple, Microsoft, or Nvidia.")
                host_hist.append(("assistant", _reply))
                st.rerun()
        elif cand:
            st.session_state["host_pending"] = get_resolver().resolve(
                intent.get("company_query") or _prompt, limit=3)
            host_hist.append(("assistant", "I found a few matches — which did you mean?"))
            st.rerun()
        else:
            host_hist.append(("assistant", "I couldn't find that company on the SEC list. "
                              "Try another name, or its ticker."))
            st.rerun()
    elif kind == "followup":
        # Answer in-place, grounded in the analysis currently on screen (stashed last render).
        # Works even with nothing loaded: the model glossary still answers concept questions.
        ctx = st.session_state.get("active_ctx") or {}
        try:
            with live_area.chat_message("assistant", avatar="🔍"):
                _reply = st.write_stream(get_writer().stream_answer(
                    _prompt, ctx.get("evidence", ""), host_hist, ctx.get("analogues")))
        except Exception:
            _reply = ("I couldn't reach the local LLM (Ollama). Start it and `ollama pull llama3`, "
                      "then ask again.")
            live_area.chat_message("assistant", avatar="🔍").markdown(_reply)
        host_hist.append(("assistant", _reply))
        st.rerun()
    else:  # help / unknown — warm host reply
        with live_area.chat_message("assistant", avatar="🔍"):
            _reply = st.write_stream(get_writer().stream_host_reply(_prompt, host_hist))
        host_hist.append(("assistant", _reply))

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
    st.info("Ask Cassandra above (e.g. *\"analyze Apple\"*), or use the sidebar. Try the `SUNBEAM` "
            "vs `SUNBEAM_RESTATED` demo cases for a clean before/after, or a live ticker like `NVDA`.")
    st.stop()

src, ident = active
years = available_years(src, ident)
if len(years) < 2:
    if src == "live":
        _ok, _cur, _reason = analyzable(ident)
        _why = {"ifrs": "it reports under international (IFRS) standards, not US-GAAP",
                "no_usgaap": "it doesn't file US-GAAP statements with the SEC",
                "few_years": "it doesn't have two years of comparable annual filings yet",
                "not_found": "I couldn't find its SEC filings"}.get(_reason, "of insufficient data")
        st.error(f"I can't analyze **{ident}** — {_why}. Try a US-listed company that files US-GAAP "
                 "(e.g. Apple, Microsoft, Nvidia).")
    else:
        st.error(f"Need at least two annual reporting years for **{ident}**; found {years or 'none'}.")
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
currency = res["cur"].get("_currency", "USD")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Data coverage", f"{cov:.0%}" if cov is not None else "—")
c2.metric("Flags triggered", len(flags))
c3.metric("Currency", currency)
c4.metric("Source", "Live SEC XBRL" if src == "live" else "Curated CSV")
if currency != "USD":
    st.caption(f"📐 Figures are in {currency}. The forensic scores are ratios, so the verdict is "
               "currency-independent — no conversion needed.")

# Deterministic evidence block — the single ground-truth input shared by memo / debate / chat.
evidence = ForensicMemoWriter.build_evidence(
    res["company"], res["verdict"], res["models"], res["cur"], res["analogues"])
# Stash for the unified top chat: on the next rerun (which runs before this block) the "Ask Cassandra"
# bar answers follow-ups grounded in exactly what's on screen.
st.session_state["active_ctx"] = {
    "company": res["company"], "evidence": evidence, "analogues": res["analogues"]}

# ----------------------------- LLM hero (the centerpiece) ------------------------------
DEBATE_AVATARS = {"Cassandra (Skeptic)": "🔻", "Michael (Bull)": "🐂", "CIO (Ruling)": "⚖️"}
tab_memo, tab_debate = st.tabs(["📝 Forensic Memo", "🎭 Committee Debate"])

with tab_memo:
    if flags:
        st.markdown("**Flags:** " + "; ".join(flags))
    memo_key = f"memo::{res['company']}::{res['cur_year']}::{res['verdict'].get('Verdict')}"
    if st.button("Generate memo", type="primary"):
        try:
            with st.spinner("Generating on local LLM (CPU inference can take 1–3 min)..."):
                text = st.write_stream(get_writer().stream_memo(
                    res["company"], res["verdict"], res["models"], res["cur"], res["analogues"]))
            st.session_state[memo_key] = text
        except Exception as e:
            st.error(f"Could not reach the local LLM (Ollama). Start it and `ollama pull llama3`.\n\n{e}")
    elif st.session_state.get(memo_key):
        st.markdown(st.session_state[memo_key])
    else:
        st.caption("A cited, plain-English risk memo grounded in the deterministic scores.")

with tab_debate:
    st.caption("Michael (Bull) vs. Cassandra (Skeptic), ruled by the CIO — multi-round, sequential "
               "on CPU (several minutes). The CIO respects the deterministic verdict.")

    is_live = src == "live"
    use_research = is_live  # bull research always on for live tickers (curated cases can't web-research)

    LENGTH_LABELS = {
        "Soundbite (~40w, 1 round)": "soundbite",
        "Brief (~75w, 1 round)": "brief",
        "Balanced (~100w, 2 rounds)": "balanced",
        "In-depth (~180w, 2 rounds)": "in-depth",
    }
    length_label = st.selectbox("Debate length", list(LENGTH_LABELS), index=2,
                                help="Shorter = more digestible on stage and faster on CPU.")
    length = LENGTH_LABELS[length_label]

    debate_key = (f"debate::{res['company']}::{res['cur_year']}::"
                  f"{res['verdict'].get('Verdict')}::{use_research}::{length}")
    brief_key = debate_key + "::brief"

    def _show_brief(brief):
        with st.expander("🐂 Michael's research & sources", expanded=False):
            st.markdown(brief["brief"])
            st.markdown("**Sources:**")
            for s in brief["sources"]:
                st.caption(s)

    # Hydrate a saved debate from disk so the Replay button knows one exists — but do NOT show it yet.
    if debate_key not in st.session_state:
        saved = demo_store.load_debate(debate_key)
        if saved:
            st.session_state[debate_key] = saved.get("transcript")
            if saved.get("brief"):
                st.session_state[brief_key] = saved["brief"]

    has_saved = bool(st.session_state.get(debate_key))
    _bc1, _bc2 = st.columns(2)
    run = _bc1.button("Run committee debate", type="primary", use_container_width=True)
    replay = _bc2.button("▶ Replay saved", disabled=not has_saved, use_container_width=True,
                         help="Replays the last saved debate with a quick reveal."
                              if has_saved else "No saved debate yet — run one first.")

    if run:
        bull_context = None
        if use_research:
            with st.spinner("🐂 Michael researching (filings + web)..."):
                try:
                    brief = get_bull_brief(ident, res["company"], tuple(res["verdict"].get("Flags", [])))
                except Exception as e:
                    brief = None
                    st.warning(f"Bull research unavailable ({e}); debating on the numbers only.")
            if brief:
                bull_context = brief["brief"]
                st.session_state[brief_key] = brief
                _show_brief(brief)
        else:
            brief = None
        try:
            transcript, spk, buf, ph = [], None, "", None
            with st.spinner("Committee debating (CPU; several minutes)..."):
                for speaker, chunk in get_writer().stream_debate(
                        res["company"], res["verdict"], res["models"], res["cur"], res["analogues"],
                        length=length, bull_context=bull_context):
                    if speaker != spk:
                        if spk is not None:
                            transcript.append((spk, buf))
                        spk, buf = speaker, ""
                        ph = st.chat_message(speaker, avatar=DEBATE_AVATARS.get(speaker, "🗣️")).empty()
                    buf += chunk
                    ph.markdown(buf)
                if spk is not None:
                    transcript.append((spk, buf))
            st.session_state[debate_key] = transcript
            st.session_state["debate_view"] = debate_key
            demo_store.save_debate(debate_key, transcript, brief)  # persist for instant replay
            narration_cache.clear(debate_key)  # fresh transcript -> drop stale audio clips
            if voice.debate_voices_ready():
                render_narration(debate_key, transcript)
        except Exception as e:
            st.error(f"Could not reach the local LLM (Ollama). Start it and `ollama pull llama3`.\n\n{e}")
    elif replay and has_saved:
        st.session_state["debate_view"] = debate_key
        if st.session_state.get(brief_key):
            _show_brief(st.session_state[brief_key])
        _animate_transcript(st.session_state[debate_key])   # quick reveal, not all-at-once
        if voice.debate_voices_ready():
            render_narration(debate_key, st.session_state[debate_key])
    elif st.session_state.get("debate_view") == debate_key and has_saved:
        # Keep the debate on screen across unrelated reruns — render instantly (no re-animation).
        if st.session_state.get(brief_key):
            _show_brief(st.session_state[brief_key])
        for speaker, text in st.session_state[debate_key]:
            st.chat_message(speaker, avatar=DEBATE_AVATARS.get(speaker, "🗣️")).markdown(text)
        if voice.debate_voices_ready():
            render_narration(debate_key, st.session_state[debate_key])
    else:
        st.caption("**Run committee debate** to generate fresh, or **▶ Replay saved** for the last one.")

# Q&A now lives in the unified "💬 Ask Cassandra" chat at the top (grounded in this analysis).

# ----------------------------- Quantitative detail (collapsible) -----------------------
with st.expander("📊 Quantitative detail — scores, gauges, evidence & data sources", expanded=False):
    render_gauges(res["models"])

    st.markdown("**Deterministic evidence (exact input handed to the LLM):**")
    st.code(evidence, language="text")

    prov = res["cur"].get("_provenance", {})
    rows = [{"Concept": k, "Value": res["cur"].get(k), "Source": v}
            for k, v in prov.items() if v is not None]
    if rows:
        st.markdown("**Resolved figures (provenance):**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if res["analogues"]:
        st.markdown("**Historical analogues (RAG):**")
        for a in res["analogues"]:
            st.markdown(f"- {a}")

    st.markdown("**Raw model output:**")
    st.json({"verdict": res["verdict"], "models": res["models"]}, expanded=False)
