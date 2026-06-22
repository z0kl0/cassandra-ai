# CASSANDRA: Infrastructure & Deployment Layout

This document describes the **local-first** infrastructure for CASSANDRA. The entire application
runs as a single-machine monolith on the local i9 workstation — there is no cloud/distributed tier;
local is the design, not a fallback.

> **Why local-first:** the i9-10900K does CPU-only inference (no usable GPU), so an 8B model runs
> at ~5–10 tok/s. Every remote service adds latency and a network failure point to a live demo that
> is graded on *"runs end-to-end without errors."* A local monolith removes those failure modes and
> the resource contention. The only network call in the core app is to the free SEC EDGAR API.
> (When presenting from another computer, the local app is simply viewed over the LAN/Tailscale —
> still the same single local process, no hosted services.)
>
> **Storage Policy:** To preserve space on the system (C:) drive, all LLM weights are stored in
> `D:\LLM\.ollama\models`. This is controlled via the `OLLAMA_MODELS` environment variable.

---

## 1. Architecture (single machine)

### Local Machine — the whole app
* **Specs:** Intel Core i9-10900K (10c/20t, CPU inference), 32 GB RAM, 1 TB NVMe.
* **Role:** AI inference, deterministic forensic math, data access, and the UI — all in one process tree.
* **Services / components:**
  * **Ollama daemon** — a **single resident model** (`llama3` 8B, Q4_K_M). The Cassandra / Bull /
    CIO personas are different *system prompts* on the same model. Set `OLLAMA_KEEP_ALIVE=-1` so the
    model stays warm; tune `num_thread` to the physical cores. *No second model* — XBRL is already
    model stays warm (persistent in RAM); alternatively, use the default 5m timeout to run "on-demand."
    Tune `num_thread` to the physical cores (10 for i9-10900K). 
    *No second model* — XBRL is already
    structured JSON, so data "extraction" is deterministic parsing, not an LLM job.
  * **Forensic engine** (`forensics.py`) — pandas/numpy implementation of Beneish M-Score, Altman
    Z-Score, Sloan accruals, and Benford's Law. Deterministic, unit-tested, auditable.
  * **SEC client** (`sec_client.py`) — ticker→CIK lookup + `companyfacts` fetch, **on-demand with a
    local file cache** (no continuous polling cron — that risks tripping SEC fair-access limits).
    Sends the required descriptive `User-Agent`.
  * **Embedded ChromaDB** — local, **file-based** (`./data/chroma`). Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) for fast, local CPU text embeddings. Stores the historical fraud-case corpus for RAG analogues. No server, no network hop.
  * **Streamlit UI** — gauges (plotly), the cited Forensic Risk Memo, and the debate view.
  * **Optional voice layer** (`voice.py`, behind `ENABLE_VOICE`) — `faster-whisper` (local STT) +
    **Piper** (local TTS), push-to-talk. See note D below.

### Data flow
```
Ticker → CIK lookup → SEC companyfacts API (cached locally)
  → Forensic Engine (M-Score · Z · Accruals · Benford)        [deterministic]
    → flags + scores
      → LangChain + Ollama (RAG over local Chroma fraud corpus → cited memo → optional debate)
        → Streamlit: gauges + memo + verdict  (+ optional Piper audio)
```

### Suggested module layout
```
app.py              # Streamlit entrypoint
sec_client.py       # ticker→CIK, companyfacts fetch + local cache
forensics.py        # M-Score / Z-Score / accruals / Benford  (pure, testable)
llm.py              # LangChain memo chain + personas/debate (single model)
rag.py              # Chroma fraud-case corpus (build + query)
voice.py            # optional, gated by ENABLE_VOICE
data/
  chroma/           # embedded vector store
  sec_cache/        # cached companyfacts JSON
  fraud_cases.csv   # curated pre-XBRL cases (Enron, WorldCom, Sunbeam, Lehman)
tests/
  test_forensics.py # math validated against textbook values
```

---

## 2. Professional polish (right-sized to the local app)

These make the project read as a serious engineering portfolio piece without adding demo risk.

### A. Environment & secrets
* All config via `python-dotenv` — no hardcoded endpoints. Example `.env` (local-first):
  ```env
  SEC_USER_AGENT=Your Name your.email@example.com
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODELS=D:\LLM\.ollama\models
  OLLAMA_MODEL=llama3
  OLLAMA_KEEP_ALIVE=-1
  CHROMA_PATH=./data/chroma
  SEC_CACHE_PATH=./data/sec_cache
  LOG_LEVEL=INFO
  ENABLE_VOICE=false
  ```
  (The vector store is embedded and file-based — there is no separate DB host to configure.)
  ### Change OLLAMA_KEEP_ALIVE=-1 to 5 or 10 min when not in demo

### B. Source control & CI (GitHub)
* **GitHub Projects** Kanban (To Do / In Progress / Review / Done) per feature
  (e.g., "Implement Z-Score", "Wire Chroma RAG", "Voice layer").
* **GitHub Actions:** on push to `main`, run `flake8` (lint) and **`pytest` on `forensics.py`** —
  proving the forensic math is correct is the single most valuable CI signal here.
* **Pre-commit hooks** to keep `main` clean.

### C. Logging
* Use the stdlib `logging` module (not `print`), level from `LOG_LEVEL`. Example:
  `[2026-06-15 14:32:01] [INFO] [sec_client] Fetched CIK 0000789019 (MSFT), 16 yrs of facts`.

---

## 3. Local-first execution plan

**Week 1 — Foundation.** Run `setup.bat`; install Ollama and `ollama pull llama3`; build
`sec_client.py` (ticker→CIK→companyfacts + cache); implement the Beneish M-Score as a vertical
slice with a unit test.

**Week 2 — Forensic engine + CI.** Complete `forensics.py` (M-Score, Altman Z, Sloan accruals,
Benford) with `pytest` against textbook values; add the XBRL tag-mapping/fallback layer; wire up
GitHub Actions.

**Week 3 — LLM + UI (core demo works).** `llm.py` memo chain (single model, JSON-structured
output) + `rag.py` Chroma fraud-case corpus; Streamlit dashboard with plotly gauges and the cited
memo.

**Week 4 — Debate.** Multi-persona Bull/Skeptic/CIO debate on one model (≤2 rounds, token-capped,
streamed).

**Week 5 — Evaluate + document.** Backtest table (precision/recall on curated classics —
Sunbeam/Enron/WorldCom/Lehman — plus live blue-chip controls); "educational, not investment
advice" disclaimer; README, screenshots, workflow diagram.

**Week 6 — Stretch.** Voice layer (Piper + faster-whisper, push-to-talk, behind `ENABLE_VOICE`).
