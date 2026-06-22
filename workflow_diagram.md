# CASSANDRA — Technical Workflow Diagram

How a request flows through the system: **user input → resolution → deterministic scoring → grounded
LLM language → output.** The single most important property is the **EVIDENCE BLOCK gate** (yellow):
all numbers and the verdict are produced by deterministic code; the LLM only ever sees — and may only
cite — that block, so it can explain and debate but **never invents a number, formula, or ticker.**

> Renders on GitHub as-is. To export an image for slides/Word: paste `docs/workflow.mmd` into
> https://mermaid.live (Actions → PNG/SVG), or open it in VS Code with a Mermaid extension and export.
> A pre-rendered copy lives at `docs/workflow.png` when available.

```mermaid
flowchart TD
    %% ---- standalone nodes ----
    U(["👤 User"])
    BAR["💬 Ask Cassandra<br/>chat bar (text)"]
    STT["🎤 faster-whisper<br/>STT"]
    DRAFT["Editable transcript<br/>(user corrects, then sends)"]
    INTENT{{"Host LLM:<br/>parse intent"}}
    HOST["Warm host reply"]
    RES["TickerResolver<br/>name → SEC ticker<br/>(fuzzy + LLM, validated)"]
    PRE{"Analyzable?<br/>≥2 yrs · US-GAAP"}
    DECL["Cassandra declines<br/>(IFRS / no data)"]
    FETCH["Fetch financials"]
    EVID["📋 EVIDENCE BLOCK<br/>deterministic ground truth<br/>+ model glossary<br/>the only thing the LLM may cite"]
    CHROMA[("🔎 ChromaDB<br/>fraud-case corpus<br/>MiniLM embeddings")]
    WEB[("🌐 DuckDuckGo + filing MD&A<br/>bull brief · UNVERIFIED")]

    subgraph SRC["📥 Data sources"]
      direction LR
      SEC[("SEC EDGAR XBRL<br/>companyfacts · cached")]
      CSV[("Curated CSV<br/>Enron · WorldCom · Sunbeam · Lehman")]
    end

    subgraph DET["🔢 Deterministic forensic engine — NO LLM (ground truth)"]
      EXT["extract_financials<br/>currency-aware · provenance"]
      MOD["6 models:<br/>Beneish M · Altman Z · Sloan<br/>Piotroski · Leverage · Benford"]
      VER["calculate_verdict<br/>🟢 / 🟡 / 🔴 + flags"]
      EXT --> MOD --> VER
    end

    subgraph GEN["🗣️ Local LLM · Ollama / Llama 3 — language only, grounded in the evidence"]
      MEMO["Cited risk memo"]
      DEB["Committee debate<br/>Cassandra · Michael · CIO"]
      QA["Interrogation Q&A"]
    end

    subgraph OUT["🖥️ Streamlit UI — output"]
      BAN["Verdict banner + metrics"]
      OM["Memo"]
      OD["Debate transcript<br/>+ 🔊 Piper voices (browser)"]
      OC["Chat answers"]
    end

    %% ---- flow ----
    U -->|types| BAR
    U -->|speaks| STT
    STT --> DRAFT --> BAR
    BAR --> INTENT
    INTENT -->|help / greeting| HOST
    HOST --> OC
    INTENT -->|follow-up| QA
    INTENT -->|analyze a company| RES
    RES --> PRE
    PRE -->|no| DECL
    DECL --> OC
    PRE -->|yes| FETCH
    FETCH --> SEC & CSV
    SEC & CSV --> EXT
    VER --> EVID
    VER --> BAN
    CHROMA -. analogues .-> EVID
    EVID --> MEMO & DEB & QA
    WEB -. asymmetric ammo .-> DEB
    MEMO --> OM
    DEB --> OD
    QA --> OC

    %% ---- styling: green = deterministic, orange = LLM, grey = data, yellow = evidence gate ----
    classDef det fill:#d8f3dc,stroke:#2a9d8f,color:#111
    classDef llm fill:#ffe8d6,stroke:#e76f51,color:#111
    classDef data fill:#e9ecef,stroke:#6c757d,color:#111
    classDef gate fill:#fff3bf,stroke:#f4a261,color:#111,stroke-width:3px
    class EXT,MOD,VER det
    class INTENT,RES,HOST,DECL,MEMO,DEB,QA,STT llm
    class SEC,CSV,CHROMA,WEB data
    class EVID gate
```

**Legend** — 🟩 green = deterministic code (numbers/verdict) · 🟧 orange = LLM (language only) ·
⬜ grey = data sources / external · 🟨 yellow = the evidence gate the LLM cannot cross.

---

## Walkthrough (maps to the required deliverable points)

**1. User input flow.** The user talks to the **host persona** in one chat bar — typed, or spoken via
the **mic** (`faster-whisper` STT → an *editable transcript* they confirm before sending). A host LLM
call parses the message into an intent: **analyze a company / follow-up question / help**.

**2. Resolution (never trust a free-text ticker).** For "analyze", `TickerResolver` maps the company
*name* to a real SEC ticker (fuzzy match + the LLM's guess, both validated against the official SEC
company list). An `analyzable` pre-flight checks for ≥2 annual years of US-GAAP data; if not (e.g. an
IFRS-only filer), Cassandra declines gracefully.

**3. Data sources.** Financials come from **live SEC EDGAR XBRL** (`companyfacts`, throttled + locally
cached) or, for pre-XBRL classics, a **curated CSV** (Enron, WorldCom, Sunbeam ± restated, Lehman).

**4. Deterministic scoring (the core — no LLM).** `extract_financials` (currency-aware, with
provenance/coverage) feeds **six models** — Beneish M-Score, Altman Z, Sloan accruals, Piotroski F,
Leverage, Benford — and `calculate_verdict` aggregates them into a **🟢/🟡/🔴 verdict + flags**. This
is plain, auditable Python; identical inputs always give identical scores.

**5. The evidence gate + retrieval (vector DB).** The verdict and figures are rendered into an
**evidence block** (plus a fixed *model glossary* of correct formulas). Optionally, **ChromaDB** RAG
(SentenceTransformers `all-MiniLM-L6-v2` over a fraud-case corpus) adds historical analogues. This
block is the **only** thing the LLM is allowed to cite.

**6. LLM interaction (language only).** A single resident **Ollama / Llama 3** model, grounded in the
evidence block, produces: a **cited memo**, a 3-persona **committee debate** (Cassandra the skeptic,
Michael the bull, a CIO who respects the verdict), and **interrogation Q&A**. For the debate only,
`research.py` adds an *asymmetric* bull brief (DuckDuckGo news + the latest filing's MD&A), explicitly
labeled **UNVERIFIED** — it fuels the argument but never the scores.

**7. Output handling (UI + API).** Everything renders in **Streamlit**: a verdict banner + metric row,
the memo, the debate transcript (optionally **read aloud** in three Piper voices played in the
browser), and chat answers. External API touchpoints are the free **SEC EDGAR** API and **DuckDuckGo**
search; the LLM, vector DB, and TTS are all **local**.

## Tools & frameworks shown
UI **Streamlit** · LLM **Ollama / Llama 3** via **LangChain** · vector DB **ChromaDB** +
**Sentence-Transformers** · data **SEC EDGAR XBRL** (`requests`) + curated CSV (**pandas/numpy**) ·
resolver **rapidfuzz** · research **ddgs** · voice **faster-whisper** (STT) + **Piper** (TTS).
