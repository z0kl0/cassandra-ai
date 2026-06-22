# Capstone Project Proposal — "CASSANDRA"
### An AI Forensic Accountant for Detecting Financial-Statement Fraud & Distress

**Course:** CAP 942 — Capstone Project: AI Application Development
**Project type:** Advanced — multi-step LangChain pipeline + RAG + (optional) multi-agent
**Format:** Streamlit web app powered by a local open-source LLM (Ollama)

---

## 1. Problem Statement

Major accounting frauds and corporate failures — **Enron, WorldCom, Sunbeam, Lehman Brothers** —
were detectable in companies' own public financial statements *years* before they collapsed.
Forensic accountants catch these cases using a well-established toolkit of quantitative red-flag
models, but applying that toolkit is slow, manual, and locked inside specialist expertise. Most
investors, analysts, and students never run it.

**CASSANDRA** turns that forensic toolkit into an automated, explainable AI application. A user
enters a stock ticker; the system pulls the company's structured financials directly from the SEC,
computes the standard forensic red-flag models in deterministic code, and then uses a local LLM to
write a plain-English **Forensic Risk Memo** — citing the exact figures behind each flag and
delivering a clear verdict: 🟢 Clean / 🟡 Watch / 🔴 High Risk. 

As an **optional stretch feature**, CASSANDRA includes an **"Interrogation Mode"** — fully local, push-to-talk voice using `faster-whisper` (speech-to-text) and **Piper** (local, open-source text-to-speech). Users can verbally question the AI about specific anomalies, creating a "JARVIS for short-sellers" experience. The text interface is the always-works primary path; voice is layered on after the core works and gated behind an `ENABLE_VOICE` flag. Crucially, **the LLM never invents a number.** All scoring is deterministic and reproducible; the LLM is used only for explanation, historical analogy, and investment-committee-style agentic debate.

## 2. Why This Project Matters

- **Real, high-value problem.** Undetected accounting fraud destroys billions in investor capital.
  Tools that surface red flags early have direct, obvious value to any investment firm, auditor, or
  analyst.
- **Demonstrates the right use of LLMs.** It draws a clean line between *deterministic quantitative
  analysis* (code) and *language generation* (LLM) — the architecture pattern that separates
  serious AI applications from "ask the chatbot to guess."
- **Unforgettable, defensible demo.** Running it on a known fraud — **Sunbeam**, where the same
  company flips 🔴 on its *as-reported* books and 🟢 on its *restated* books — and watching the red
  flags fire, then on a healthy blue-chip (MSFT/AAPL/GOOGL) where they stay calm, is dramatic *and*
  backed by a real precision/recall evaluation, not anecdote. Historical frauds (Enron, WorldCom,
  Sunbeam, Lehman) are demoed from a small curated dataset; healthy controls run live from SEC XBRL.
- **Cinematic stretch.** With voice enabled, you can physically *listen* to the Forensic Skeptic
  and the Bull agent debate the stock before the CIO agent rules — an unusual level of
  interactivity for financial analysis.
- **Genuinely novel for a capstone.** It is forensic-accounting infrastructure, not another
  summarizer or chatbot.

## 3. Tools & Frameworks Chosen

| Layer | Tool | Role |
|---|---|---|
| **LLM & Agents** | Ollama (single resident Llama 3 8B) + LangChain | One local model; the Cassandra / Bull / CIO personas are different system prompts on the *same* model (no second model — avoids CPU model-swap thrash) |
| **Voice / Audio** *(optional stretch)* | `faster-whisper` (local STT) + **Piper** (local TTS) | Push-to-talk "Interrogation Mode". Piper is fully local/open-source; Edge-TTS is only an online fallback |
| **Data source** | SEC EDGAR XBRL APIs (`companyfacts`, `companyconcept`, `submissions`) | Free, official, structured financial data for every US public company |
| **Forensic engine** | Python, pandas, numpy | Deterministic computation of Beneish M-Score, Altman Z-Score, Sloan accruals, Benford's Law |
| **Vector DB** | ChromaDB | Caches fundamentals and stores a historical fraud-case corpus for RAG analogues |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Embeds fraud-case corpus for fast, local CPU retrieval |
| **Frontend** | Streamlit (+ plotly; `st.audio_input` for optional push-to-talk) | Ticker input, red-flag gauges, forensic memo, debate view, optional audio capture |

**No paid APIs. No model training. Runs entirely on the local machine (i9 / 32 GB RAM, CPU
inference) — the only network call is to the free SEC EDGAR API.**

### The forensic models (computed in code, not by the LLM)
- **Beneish M-Score** — 8-ratio earnings-manipulation model; `M > −2.22` flags a likely manipulator.
- **Altman Z-Score** — financial-distress / bankruptcy model; `Z < 1.81` signals distress.
- **Sloan Accruals Ratio** — `(Net Income − Operating Cash Flow) / Average Total Assets`; high
  accruals signal low earnings quality.
- **Benford's Law** — tests whether the leading-digit distribution of reported figures conforms to
  `P(d) = log10(1 + 1/d)`; large deviations flag possible fabrication.

## 4. Expected Output & User Interaction

**User input:** a stock ticker (e.g., `MSFT`, `AAPL`, `GOOGL`) or a curated case (e.g., `SUNBEAM`)
selected in the Streamlit UI — or, with the optional voice layer enabled, spoken via push-to-talk
(e.g., *"Cassandra, analyze Microsoft."*).

**Processing:** ticker → CIK lookup → SEC XBRL fundamentals → deterministic forensic engine →
LangChain + Ollama memo generation (with RAG over historical fraud cases).

**Output:**
1. A **red-flag dashboard** — gauges/scores for M-Score, Z-Score, accruals, and Benford
   conformity, color-coded.
2. An **Interrogation Mode** — chat with the AI to drill into the Forensic Risk Memo (text always;
   optional spoken Q&A with the AI reading findings aloud when voice is enabled).
3. A **"Bring in the Bull" toggle** — summons "Michael" (the Bull persona) to defend management's
   narrative. The debate runs ≤2 rounds (token-capped and streamed to keep CPU latency acceptable)
   before the "CIO" persona weighs the deterministic math against the narrative and delivers a
   final verdict.

**Workflow:**
```
Ticker → CIK lookup → SEC companyfacts XBRL API
  → Forensic Engine (M-Score · Z-Score · Accruals · Benford)  [deterministic]
    → Flags + scores (cached in ChromaDB)
      → LangChain → Ollama LLM (RAG analogues → cited memo → optional debate)
        → Streamlit: gauges + Forensic Risk Memo + verdict
```

## 5. Evaluation Plan

Assemble a small labeled set of ~10 known fraud/failure cases and ~10 clean firms, run the engine
across all of them, and report **precision and recall** with an honest discussion of base rates and
limitations. This proves the tool works rather than asserting it.

## 6. Scope & Risks

- **In scope:** US public companies with XBRL data; the four forensic models; LLM memo + dashboard;
  optional agent debate; a curated backtest.
- **Primary risk:** XBRL tag inconsistency across filers/years — mitigated with a tag-mapping layer
  with fallbacks and a curated set of demo tickers.
- **Data-availability note:** SEC XBRL (`companyfacts`) only goes back to ~2009, so pre-XBRL frauds
  (Enron, WorldCom, Sunbeam, Lehman) are served from a small curated CSV transcribed from primary
  filings; healthy controls (MSFT/AAPL/GOOGL/…) run live from XBRL. *Luckin Coffee and Wirecard were
  evaluated and dropped as quantitative cases:* Luckin's fraud year was never filed as an audited
  annual report (only restated figures exist), and Wirecard is a foreign (Frankfurt/IFRS) filer not
  in EDGAR whose fraud was fabricated **cash** — which makes a company look *healthier*, a documented
  blind spot of the Beneish/Altman models (as is pre-revenue **Nikola**). These limitations are
  reported honestly in the evaluation rather than papered over.
- **Compliance:** prominent "educational analysis, not investment advice" disclaimer; SEC API used
  with a proper `User-Agent` and within rate limits.

---

*Deliverables per course requirements: this proposal, a working Streamlit application, a data-sources
note (SEC EDGAR), a workflow diagram, final documentation, and a 5–10 minute presentation/demo.
Code and docs published to GitHub as `FirstName_LastName_CapstoneAI`.*
