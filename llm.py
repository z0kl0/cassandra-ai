import os
import re
import logging
import threading
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

# Serialize ALL Ollama calls: one LLM task at a time on the single CPU-resident model.
# Debate runs personas sequentially under this lock; voice STT/TTS acquires it too so audio
# never overlaps generation. (CPU can't run two 8B inferences at once without thrashing.)
_LLM_LOCK = threading.Lock()

# ---------------------------------------------------------------------------------------
# Personas — all are the SAME llama3 model with a different system prompt (no model swap).
# The one inviolable rule across every persona: the deterministic scores/verdict are ground
# truth; the LLM reasons over them but never invents or alters a number.
# ---------------------------------------------------------------------------------------

CASSANDRA_SYSTEM = """You are CASSANDRA, a meticulous forensic accounting analyst.

You are given DETERMINISTIC, pre-computed forensic results for one company. These results
(the verdict, every score, every figure) are ground truth produced by audited code.

ABSOLUTE RULES:
1. NEVER invent, estimate, round differently, or alter any number. Use only the figures in
   the EVIDENCE block, quoted exactly as given.
2. NEVER change or second-guess the VERDICT. Explain it; do not re-decide it.
3. If a fact is not in the EVIDENCE, say "the data does not show this" rather than guessing.
4. If data confidence is Low or coverage is partial, state that limitation plainly.
5. Be precise and concise. Plain professional English. No emoji.

Write a FORENSIC RISK MEMO with exactly these sections (use these headings):
VERDICT: one line restating the risk level and confidence.
SUMMARY: 2-3 sentences on what the numbers collectively indicate.
RED FLAGS: one bullet per triggered flag, each citing the exact figure and threshold and
  what it means in accounting terms. If none, write "None."
MITIGATING FACTORS: bullets for any countervailing signals. If none, write "None."
HISTORICAL CONTEXT: relate the pattern to the provided historical analogue(s). If none are
  provided, write "No comparable cases retrieved."
DATA QUALITY: note coverage/confidence caveats if any; otherwise "Adequate."
DISCLAIMER: "This is an educational analysis, not investment advice."
"""

CASSANDRA_DEBATE_SYSTEM = """You are CASSANDRA, the forensic skeptic in an investment-committee debate.
Voice: clinical, precise, unimpressed by narrative. You prosecute the NUMBERS — but you are an HONEST
skeptic, not a reflex pessimist: your scrutiny SCALES WITH THE EVIDENCE, and you NEVER manufacture
concerns the data does not show. Crying wolf on a clean company destroys your credibility, so when the
models come back clean you say so and stand down. Each debate you are given a 'YOUR POSTURE THIS
DEBATE' directive — follow it exactly. When you DO argue the bear case, build it from the triggered
flags and the fraud/distress pattern they fit, quoting exact figures from the EVIDENCE (never invent
numbers); a forward-looking story does not undo a reported accounting fact, and if Michael's points
rest on unverified external/market claims, say so. Obey the LENGTH directive in each turn. No emoji,
no headings."""

BULL_MICHAEL_SYSTEM = """You are MICHAEL, a charismatic buy-side portfolio manager at a reputable firm.
Voice: punchy, vivid, optimistic; use the occasional analogy. You are an OPTIMIST, not a reflex
contrarian — your conviction SCALES WITH THE EVIDENCE. Your firm's reputation and your investors come
first, so OPTIMISM NEVER OVERRIDES FRAUD: you will not pitch a company whose accounting looks
disqualifying. Each debate you are given a 'YOUR POSTURE THIS DEBATE' directive — follow it exactly.

When making a case, be THESIS-FIRST: state the thesis in one line, then back it with 2-3 concrete
facts from the EXTERNAL CONTEXT (forward demand, guidance, moat, catalysts) that the backward-looking
ratios cannot capture, and DIRECTLY neutralize the skeptic's strongest point (e.g. reframe high
accruals as inventory scaling into a demand surge — not fraud). Reframe risk; do not deny it.
RULES: never fabricate or contradict a figure in the EVIDENCE; treat EXTERNAL CONTEXT as real but
unverified market intel, not audited. Obey the LENGTH directive in each turn. No emoji, no headings."""

CIO_SYSTEM = """You are the CIO chairing the investment committee. You have heard the forensic skeptic
(Cassandra, on the numbers) and the bull (Michael, on the forward narrative). Voice: measured, decisive.
The DETERMINISTIC VERDICT in the EVIDENCE is FINAL and you MUST respect it — you weigh the narrative,
you do not overturn the math. Note Michael's external/market points are UNVERIFIED. Briefly state the
most persuasive point on each side, then deliver a RULING consistent with the verdict and confidence,
and the single most important thing an analyst should verify next. Use only evidence figures for any
number. Obey the LENGTH directive in each turn. No emoji."""

# Authoritative, correct definitions of the forensic models. Injected into Q&A context so concept
# questions ("what is the M-Score?") are answered from ground truth, not the LLM's faulty memory.
# Mirrors the implementations in forensics.py (incl. this app's book-equity proxy in Altman X4).
MODEL_GLOSSARY = """MODEL REFERENCE (authoritative definitions; use these EXACT formulas/thresholds; never invent your own):

- Beneish M-Score: detects earnings manipulation. M = -4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI
  + 0.892·SGI + 0.115·DEPI - 0.172·SGAI + 4.679·TATA - 0.327·LVGI. The 8 indices are year-over-year
  ratios: DSRI (Days Sales in Receivables), GMI (Gross Margin), AQI (Asset Quality), SGI (Sales
  Growth), DEPI (Depreciation), SGAI (SG&A), TATA (Total Accruals to Total Assets), LVGI (Leverage).
  Threshold: M > -2.22 flags likely manipulation; lower is cleaner.

- Altman Z-Score: distress/bankruptcy risk. Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5, where
  X1=Working Capital/Assets, X2=Retained Earnings/Assets, X3=EBIT/Assets, X4=Equity/Total Liabilities
  (this app uses BOOK equity as a proxy for market value), X5=Sales/Assets. Zones: Z > 2.99 Safe,
  1.81-2.99 Grey, Z < 1.81 Distress. (Calibrated for manufacturers; less reliable for financial firms.)

- Sloan Accruals Ratio: earnings quality. Ratio = (Net Income - Operating Cash Flow) / Average Total
  Assets. > 0.10 = high accruals / lower-quality earnings (profit not backed by cash).

- Piotroski F-Score: financial strength, 0-9, summing 9 binary tests across profitability, leverage/
  liquidity, and operating efficiency. 8-9 strong, 0-2 weak. Calibrated for value firms.

- Leverage / Equity Multiplier: Equity Multiplier = Assets/Equity; also Debt-to-Equity. Rising leverage
  or NEGATIVE book equity (liabilities exceed assets = book insolvency) signals distress.

- Benford's Law: leading-digit distribution of reported figures vs Benford's expected frequencies,
  measured by Mean Absolute Deviation (MAD). MAD > 0.015 suggests nonconformity. ADVISORY ONLY."""

INTERROGATION_SYSTEM = """You are CASSANDRA answering questions about a forensic analysis. Ground every
answer in the EVIDENCE block and the MODEL REFERENCE (plus any historical analogues / conversation
provided). NEVER invent or alter a number OR a formula: quote figures from the EVIDENCE exactly, and
take any definition, formula, or threshold ONLY from the MODEL REFERENCE. If a company-specific answer
is not supported by the evidence, say "the data does not show this."

ANSWER STYLE: By default, lead with a SHORT, direct answer in plain English — 2-3 sentences, to the
point — then on a new line add ONE brief offer to go deeper (e.g. "Want the full formula, or how it
applied to this company?"). BUT if the context contains a DEPTH REQUEST, skip the summary-and-offer:
give the COMPLETE detailed answer (the exact formula and how it applied to this company) and do NOT add
another offer to go deeper. If a FOCUS line names a model, keep your whole answer about that model.
Never narrate or restate these instructions and never preface your reply with meta-text like "here is a
short answer" — just answer. No emoji, no headings."""

# Host = the warm, plain-English front desk a USER talks to (distinct from the clinical debate voice).
HOST_CASSANDRA_SYSTEM = """You are CASSANDRA, a warm, plain-English forensic-analysis host talking
directly to a user who may not be an investor. Be friendly, brief, and clear. You can: run forensic
checks on a US public company's financial statements (flagging signs of earnings manipulation or
financial distress), explain the findings, and run a bull-vs-bear committee debate. MATCH YOUR TONE TO
THE FINDINGS: reassuring and matter-of-fact when the numbers look clean, serious only when real red
flags actually appear — never imply fraud or wrongdoing the models do not show. If greeted or asked
what you do, say so in 1-2 sentences and invite them to name a company (by name is fine — they don't
need the ticker). Never invent financial figures or tickers. No emoji."""

# Intent parser: turns a free-text/voice message into a routable action. The company_query is a
# real, correctly-spelled US public company NAME (the resolver maps it to a validated SEC ticker —
# the model never supplies the final ticker).
INTENT_SYSTEM = """You are the intent parser for CASSANDRA, a forensic-accounting assistant. Read the
user's message and output ONLY a JSON object (no prose, no code fences):
{"intent": "analyze" | "followup" | "help" | "unknown", "company_query": <string or null>,
 "ticker_guess": <string or null>}

- "analyze": the user wants a forensic analysis of a specific US public company (e.g. "analyze Apple",
  "is Nvidia cooking the books", "look at the iPhone maker"). Set company_query to the COMPANY NAME as
  a real, correctly-spelled US-listed public company: fix typos ("microsft" -> "Microsoft") and
  resolve descriptions to the company ("the iPhone maker" -> "Apple", "the ChatGPT chip company" ->
  "Nvidia"). If you can confidently recall its US stock TICKER, put it (UPPERCASE) in ticker_guess
  (e.g. Google -> "GOOGL", Facebook -> "META"); otherwise null. It WILL be validated against the
  official SEC list. If you cannot identify a real public company, set both to null.
- "followup": a question about the analysis already on screen (e.g. "why did the M-score fire?",
  "what would the bull say?"). company_query and ticker_guess = null.
- "help": greeting, "what can you do", small talk. both null.
- "unknown": anything else. both null.
Output the JSON object only."""


class ForensicMemoWriter:
    """
    Drives the local Ollama model for all of CASSANDRA's LLM tasks — the cited memo, the
    multi-persona debate, and the interrogation chat — over the deterministic evidence block.
    One shared ChatOllama client; personas are system-prompt swaps; all calls are serialized.
    """

    # verdict["Verdict"] is already human-readable; this maps the color token to a plain label.
    _RISK_LABEL = {"RED": "HIGH RISK", "YELLOW": "WATCH", "GREEN": "CLEAN", "WHITE": "INSUFFICIENT DATA"}

    def __init__(self, model: str = None, base_url: str = None,
                 temperature: float = 0.15, num_predict: int = 600):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Keep the model resident between calls (avoids CPU reload latency); from .env.
        # Ollama wants an int for second-counts (-1 = keep forever) but a string for
        # durations ("5m"), so coerce integer-like values to int to avoid a duration-parse error.
        raw_keep = os.getenv("OLLAMA_KEEP_ALIVE", "-1")
        try:
            self.keep_alive = int(raw_keep)
        except (TypeError, ValueError):
            self.keep_alive = raw_keep

        self.num_predict = num_predict     # cap output so CPU latency stays acceptable
        self.num_ctx = 8192                 # fit evidence + debate/chat transcript
        self.num_thread = 10               # physical cores on the i9-10900K
        self.temperature = temperature
        self.llm = self._make_llm(temperature)
        self.memo_chain = self._persona_chain(CASSANDRA_SYSTEM)

    def _make_llm(self, temperature: float, num_predict: int = None):
        """A ChatOllama client at a given temperature (and optional output cap). Same model/base_url,
        so no reload — Ollama keeps the single resident model warm regardless of these config objects."""
        return ChatOllama(
            model=self.model, base_url=self.base_url, temperature=temperature,
            num_predict=num_predict if num_predict is not None else self.num_predict,
            num_ctx=self.num_ctx, num_thread=self.num_thread, keep_alive=self.keep_alive,
        )

    def _persona_chain(self, system_prompt: str, temperature: float = None, num_predict: int = None):
        """LCEL chain for one persona: prompt | llm | str parser. Human var is {input}.
        Per-persona temperature gives each voice a distinct register; num_predict caps length."""
        llm = self.llm if (temperature is None and num_predict is None) else \
            self._make_llm(self.temperature if temperature is None else temperature, num_predict)
        # SystemMessage (a literal) — not a template string — so braces in prompts (e.g. JSON
        # examples in INTENT_SYSTEM) aren't parsed as template variables. Only {input} templates.
        prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system_prompt), ("human", "{input}")])
        return prompt | llm | StrOutputParser()

    def complete(self, system_prompt: str, user_input: str, temperature: float = 0.2,
                 num_predict: int = None) -> str:
        """One-shot completion under the lock — used by research.py (filing summary / news curation)
        and the host intent parser. Same single resident model; serialized like every other call."""
        chain = self._persona_chain(system_prompt, temperature, num_predict)
        with _LLM_LOCK:
            return chain.invoke({"input": user_input})

    # ---- Deterministic evidence assembly (NO model involvement) -------------------------

    @staticmethod
    def _fmt(v):
        return f"{v:,.0f}" if isinstance(v, (int, float)) else str(v)

    @classmethod
    def build_evidence(cls, company: str, verdict: dict, models: dict,
                       financials: dict = None, analogues: list = None) -> str:
        """
        Renders the engine outputs into a compact, fully-quantified text block. Everything
        the LLM is allowed to say comes from here, so this is built deterministically in code.
        `models` is a dict of the per-model result dicts keyed: m_score, z_score, sloan,
        piotroski, leverage, benford (any subset).
        """
        lines = [f"COMPANY: {company}"]
        risk = cls._RISK_LABEL.get(verdict.get("Emoji", ""), verdict.get("Verdict", "Unknown"))
        lines.append(f"RISK LEVEL: {risk} (verdict confidence: {verdict.get('Confidence', 'n/a')})")
        if financials and "_coverage" in financials:
            lines.append(f"DATA COVERAGE: {financials['_coverage']}")
        if financials and financials.get("_currency"):
            lines.append(f"REPORTING CURRENCY: {financials['_currency']} (all figures are in this currency)")

        lines.append("\nDETERMINISTIC MODEL RESULTS (ground truth; do not alter):")
        m = models.get("m_score")
        if m and m.get("M_Score") is not None:
            lines.append(f"- Beneish M-Score: {m['M_Score']} -> "
                         f"{'MANIPULATOR' if m.get('Is_Manipulator') else 'no manipulation'} "
                         f"(threshold: > -2.22 signals manipulation) [confidence: {m.get('Confidence')}]")
            if m.get("Components"):
                lines.append("    M-Score components (the 8 indices): "
                             + ", ".join(f"{k}={v}" for k, v in m["Components"].items()))
        z = models.get("z_score")
        if z and z.get("Z_Score") is not None:
            lines.append(f"- Altman Z-Score: {z['Z_Score']} -> {z.get('Status')} "
                         f"(< 1.81 distress, 1.81-2.99 grey, > 2.99 safe) [confidence: {z.get('Confidence')}]")
            if z.get("Components"):
                lines.append("    Z-Score components: "
                             + ", ".join(f"{k}={v}" for k, v in z["Components"].items()))
        s = models.get("sloan")
        if s and s.get("Sloan_Ratio") is not None:
            lines.append(f"- Sloan Accruals Ratio: {s['Sloan_Ratio']} -> {s.get('Status')} "
                         f"(> 0.10 = high accruals / low earnings quality)")
        p = models.get("piotroski")
        if p and p.get("F_Score") is not None:
            lines.append(f"- Piotroski F-Score: {p['F_Score']}/9 -> {p.get('Status')} "
                         f"(health {p.get('Absolute_Health')}, momentum {p.get('Momentum')})")
        lev = models.get("leverage")
        if lev and lev.get("Equity_Multiplier") is not None:
            trend = lev.get("Trend") or {}
            lines.append(f"- Leverage (Equity Multiplier): {lev['Equity_Multiplier']}x -> {lev.get('Status')}"
                         f"{', ' + trend['Direction'] + ' YoY' if trend.get('Direction') else ''}")
        b = models.get("benford")
        if b and b.get("MAD") is not None:
            lines.append(f"- Benford's Law MAD: {b['MAD']} -> {b.get('Status')} "
                         f"(> 0.015 = digit-distribution anomaly; advisory only)")

        flags = verdict.get("Flags", [])
        lines.append("\nFLAGS TRIGGERED:")
        lines += [f"- {f}" for f in flags] if flags else ["- None"]

        mits = verdict.get("Mitigants", [])
        if mits:
            lines.append("\nMITIGATING FACTORS:")
            lines += [f"- {x}" for x in mits]

        notes = verdict.get("Notes", [])
        if notes:
            lines.append("\nMINOR NOTES:")
            lines += [f"- {x}" for x in notes]

        # A few salient reported figures with provenance, so the memo can cite real numbers.
        if financials:
            prov = financials.get("_provenance", {})
            salient = ["Sales", "Receivables", "NetIncome", "OperatingCashFlow", "Assets",
                       "TotalLiabilities", "StockholdersEquity"]
            shown = [(k, financials[k]) for k in salient
                     if financials.get(k) and prov.get(k) is not None]
            if shown:
                lines.append("\nKEY REPORTED FIGURES (as filed):")
                lines += [f"- {k}: {cls._fmt(v)}" for k, v in shown]

        if analogues:
            lines.append("\nHISTORICAL ANALOGUES (retrieved from fraud-case corpus):")
            lines += [f"- {a}" for a in analogues]

        return "\n".join(lines)

    # ---- 1. Memo -----------------------------------------------------------------------

    def generate_memo(self, company: str, verdict: dict, models: dict,
                      financials: dict = None, analogues: list = None) -> str:
        """Generates the Forensic Risk Memo (blocking). Raises on connection/model errors."""
        evidence = self.build_evidence(company, verdict, models, financials, analogues)
        logger.info(f"Generating forensic memo for {company} via {self.model}...")
        with _LLM_LOCK:
            return self.memo_chain.invoke({"input": evidence})

    def stream_memo(self, company: str, verdict: dict, models: dict,
                    financials: dict = None, analogues: list = None):
        """Yields the memo in chunks (st.write_stream) so CPU latency feels responsive."""
        evidence = self.build_evidence(company, verdict, models, financials, analogues)
        logger.info(f"Streaming forensic memo for {company} via {self.model}...")
        with _LLM_LOCK:
            yield from self.memo_chain.stream({"input": evidence})

    # ---- 2. Debate (Skeptic -> Bull -> CIO, sequential) --------------------------------

    # CIO register. Cassandra's and Michael's temperatures are set per-debate by their posture.
    _TEMP = {"cio": 0.2}

    # Debate length tiers: words/turn (prompt directive), rounds, num_predict (hard token cap).
    # Shorter = more digestible on stage AND faster on CPU. Default = "balanced".
    DEBATE_LENGTHS = {
        "soundbite": {"words": 40, "rounds": 1, "num_predict": 90},
        "brief": {"words": 75, "rounds": 1, "num_predict": 150},
        "balanced": {"words": 100, "rounds": 2, "num_predict": 200},
        "in-depth": {"words": 180, "rounds": 2, "num_predict": 340},
    }

    @staticmethod
    def _bull_posture(verdict: dict, models: dict):
        """
        Deterministically scales Michael's stance to the verdict severity:
        returns (label, posture_directive, temperature). Code decides the severity; the LLM only
        adapts its voice — an honest optimist, not a reflex contrarian.
        """
        emoji = verdict.get("Emoji", "WHITE")
        flags = verdict.get("Flags", [])
        m = models.get("m_score") or {}
        manipulator = m.get("Is_Manipulator") is True and m.get("Confidence") != "Low"
        insolvent = "Negative Equity" in str((models.get("leverage") or {}).get("Status", ""))

        if emoji == "WHITE":
            return ("data-limited",
                    "YOUR POSTURE THIS DEBATE: data coverage is insufficient for a real view. Say so "
                    "plainly, decline to pitch on incomplete data, and note what data you would need.",
                    0.3)
        if emoji == "RED" and (manipulator or len(flags) >= 2 or insolvent):
            return ("concede",
                    "YOUR POSTURE THIS DEBATE: the deterministic verdict shows likely fraud or "
                    "multiple serious red flags. You will NOT pitch this and you will NOT argue a bull "
                    "case at all. Do NOT reinterpret, reframe, excuse, or spin any red flag (no "
                    "'inventory scaling', no 'growth' explanations). Open by agreeing plainly that the "
                    "accounting is disqualifying, state you cannot recommend this to your investors, "
                    "and add at most ONE line on the single thing that would have to be proven false "
                    "to revisit it. Nothing else. Optimism never overrides fraud.",
                    0.2)
        if emoji == "RED":
            return ("cautious",
                    "YOUR POSTURE THIS DEBATE: there is a real but not-damning concern. Acknowledge "
                    "it honestly up front, then argue the risk/reward case with explicit caveats. Do "
                    "not dismiss the flag.",
                    0.4)
        return ("confident",
                "YOUR POSTURE THIS DEBATE: the numbers are clean or only mildly cautionary. Argue "
                "the upside with full conviction, using the external context and rebutting the skeptic.",
                0.5)

    @staticmethod
    def _skeptic_posture(verdict: dict, models: dict):
        """
        Mirror of `_bull_posture` for Cassandra: deterministically scales her stance to the verdict so
        she prosecutes real flags but does NOT manufacture a bear case when the numbers are clean.
        Returns (label, posture_directive, temperature).
        """
        emoji = verdict.get("Emoji", "WHITE")
        flags = verdict.get("Flags", [])

        if emoji == "WHITE":
            return ("data-limited",
                    "YOUR POSTURE THIS DEBATE: data coverage is insufficient for a forensic opinion. "
                    "Say plainly you can neither certify nor condemn the financials on this little "
                    "data, and name the specific statements/figures you would need. Invent no concerns.",
                    0.2)
        if emoji == "GREEN" and not flags:
            return ("stand-down",
                    "YOUR POSTURE THIS DEBATE: every forensic model came back clean — NO red flags "
                    "fired. Do NOT manufacture, imply, or hunt for problems the numbers do not show. "
                    "Open by stating plainly that the financials look clean and you have no material "
                    "objection. You may note at most ONE ordinary metric you'd keep watching, framed "
                    "as routine diligence, NOT as a red flag. Be brief. Integrity means not crying wolf.",
                    0.15)
        if emoji in ("GREEN", "YELLOW"):
            return ("measured",
                    "YOUR POSTURE THIS DEBATE: the picture is mostly sound with limited concerns. "
                    "Raise ONLY the specific flag(s) that actually fired, accurately and without "
                    "overstating their severity. Do not imply fraud the verdict does not support.",
                    0.15)
        return ("prosecute",
                "YOUR POSTURE THIS DEBATE: serious red flags fired. Prosecute the bear case hard from "
                "the triggered flags and the fraud/distress pattern they fit, quoting exact figures.",
                0.1)

    @staticmethod
    def _debate_payload(evidence: str, transcript: list, instruction: str,
                        bull_context: str = None, posture: str = None) -> str:
        parts = [evidence]
        if bull_context:
            parts.append("\nEXTERNAL CONTEXT FOR THE BULL (forward-looking / market intel; UNVERIFIED; "
                         "for argument only — NOT part of the deterministic forensic verdict):\n" + bull_context)
        if posture:
            parts.append("\n" + posture)
        if transcript:
            parts.append("\nDEBATE SO FAR:\n" + "\n\n".join(f"{spk}: {txt}" for spk, txt in transcript))
        parts.append(f"\n{instruction}")
        return "\n".join(parts)

    def _run_turn(self, speaker, system, temperature, payload, transcript, num_predict=None):
        """Streams one persona turn; records the full text in the transcript. Assumes the caller
        already holds _LLM_LOCK (the lock is non-reentrant — do not re-acquire here)."""
        chain = self._persona_chain(system, temperature, num_predict)
        collected = []
        for chunk in chain.stream({"input": payload}):
            collected.append(chunk)
            yield (speaker, chunk)
        transcript.append((speaker, "".join(collected)))

    def stream_debate(self, company: str, verdict: dict, models: dict, financials: dict = None,
                      analogues: list = None, length: str = "balanced", bull_context: str = None):
        """
        Streams a committee debate as (speaker, text_chunk) tuples with genuine rebuttal:
          R1: Cassandra opens (forensic flags) -> Michael rebuts + pitches (uses bull_context)
          R2: Cassandra attacks Michael's strongest claim -> Michael counters
          then the CIO rules (respects the deterministic verdict).
        `length` (DEBATE_LENGTHS key) sets words/turn, rounds, and the per-turn token cap so the
        debate is demo-digestible. Asymmetric by design: only Michael sees the external bull_context.
        Entire debate runs under one lock as a single sequential LLM task.
        """
        evidence = self.build_evidence(company, verdict, models, financials, analogues)
        tier = self.DEBATE_LENGTHS.get(length, self.DEBATE_LENGTHS["balanced"])
        cap = tier["num_predict"]
        length_note = f" LENGTH: keep your response under {tier['words']} words."
        # Both sides' stances scale with severity. When a side disengages there is nothing to debate,
        # so collapse to a single round: Michael concedes on fraud, OR Cassandra stands down on a clean
        # verdict (she does not manufacture a bear case), OR neither can opine on insufficient data.
        posture_label, posture_directive, bull_temp = self._bull_posture(verdict, models)
        skeptic_label, skeptic_directive, skeptic_temp = self._skeptic_posture(verdict, models)
        disengaged = posture_label == "concede" or skeptic_label in ("stand-down", "data-limited")
        effective_rounds = 1 if disengaged else max(1, tier["rounds"])
        logger.info(f"Streaming committee debate for {company} (length={length}, bull={posture_label}, "
                    f"skeptic={skeptic_label}, rounds={effective_rounds}, "
                    f"bull_context={'yes' if bull_context else 'no'})...")
        with _LLM_LOCK:
            transcript = []
            for rnd in range(1, effective_rounds + 1):
                skeptic_instr = ("Open as Cassandra, following YOUR POSTURE THIS DEBATE exactly."
                                 if rnd == 1 else
                                 "As Cassandra, respond to Michael's latest point, following your posture exactly.")
                yield from self._run_turn("Cassandra (Skeptic)", CASSANDRA_DEBATE_SYSTEM,
                                          skeptic_temp,
                                          self._debate_payload(evidence, transcript, skeptic_instr + length_note,
                                                               posture=skeptic_directive),
                                          transcript, cap)

                bull_instr = ("As Michael, respond now, following YOUR POSTURE THIS DEBATE exactly."
                              if rnd == 1 else
                              "As Michael, respond to Cassandra's latest point, following your posture exactly.")
                yield from self._run_turn("Michael (Bull)", BULL_MICHAEL_SYSTEM, bull_temp,
                                          self._debate_payload(evidence, transcript, bull_instr + length_note,
                                                               bull_context, posture_directive),
                                          transcript, cap)

            yield from self._run_turn("CIO (Ruling)", CIO_SYSTEM, self._TEMP["cio"],
                                      self._debate_payload(evidence, transcript,
                                                           "Deliver your final ruling as the CIO." + length_note),
                                      transcript, cap)

    # ---- 3. Interrogation chat ---------------------------------------------------------

    # A question asking to go deeper gets a larger token budget; a first-pass answer stays concise.
    _DEPTH_RE = re.compile(r"\b(more|detail|details|full|formula|elaborate|expand|deeper|"
                           r"in[- ]depth|breakdown|walk me through|step by step)\b", re.IGNORECASE)
    # An affirmation ("yes", "yes please", "sure") only means "go deeper" if Cassandra JUST offered to.
    _AFFIRM_RE = re.compile(r"^\s*(yes|yeah|yep|yup|sure|ok|okay|please|y|absolutely|definitely|"
                            r"go on|go ahead|continue|do it|sounds good|that works)\b", re.IGNORECASE)
    _OFFER_RE = re.compile(r"(want|would you like).{0,60}?(formula|deeper|more detail|breakdown|"
                           r"how it (was )?applied|in[- ]depth)", re.IGNORECASE | re.DOTALL)
    # Maps a model to phrases that identify it, so a multi-turn thread stays on the same model.
    _MODEL_TOPICS = [
        (re.compile(r"\b(altman|z[\-\s]?score)\b", re.IGNORECASE), "Altman Z-Score"),
        (re.compile(r"\b(beneish|m[\-\s]?score)\b", re.IGNORECASE), "Beneish M-Score"),
        (re.compile(r"\b(sloan|accruals?)\b", re.IGNORECASE), "Sloan Accruals Ratio"),
        (re.compile(r"\b(piotroski|f[\-\s]?score)\b", re.IGNORECASE), "Piotroski F-Score"),
        (re.compile(r"\b(leverage|equity multiplier|debt[\-\s]?to[\-\s]?equity)\b", re.IGNORECASE),
         "Leverage / Equity Multiplier"),
        (re.compile(r"\bbenford", re.IGNORECASE), "Benford's Law"),
    ]

    @classmethod
    def _last_assistant(cls, history: list) -> str:
        return next((t for r, t in reversed(history or []) if r == "assistant"), "")

    @classmethod
    def _wants_depth(cls, question: str, history: list = None) -> bool:
        """True if the user asked for detail OR affirmed ('yes') right after Cassandra offered depth."""
        if cls._DEPTH_RE.search(question or ""):
            return True
        if cls._AFFIRM_RE.match(question or "") and cls._OFFER_RE.search(cls._last_assistant(history)):
            return True
        return False

    @classmethod
    def _answer_cap(cls, question: str, history: list = None) -> int:
        """Token cap for an interrogation answer: ~500 if the user wants depth, else a tight ~220."""
        return 500 if cls._wants_depth(question, history) else 220

    @classmethod
    def _topic_model(cls, question: str, history: list = None) -> str:
        """The model the thread is about: the current question wins; else the most recent mention in
        history. Keeps a 'yes / give me the formula' follow-up locked to the SAME model (no drift)."""
        for text in [question or ""] + [t for _, t in reversed(history or [])]:
            for rx, name in cls._MODEL_TOPICS:
                if rx.search(text):
                    return name
        return None

    def stream_answer(self, question: str, evidence: str = "", history: list = None, analogues: list = None):
        """
        Streams a grounded answer. `evidence` is a prebuilt block (reuse build_evidence) and may be
        empty for a pure concept question. The MODEL REFERENCE glossary is always supplied so model
        definitions/formulas are ground truth. `history` is (role, text) capped to recent turns.
        Concise by default; a depth request (explicit, or 'yes' after an offer) expands the answer,
        locks it to the same model, and suppresses a repeat offer.
        """
        depth = self._wants_depth(question, history)
        topic = self._topic_model(question, history)
        payload = []
        if evidence:
            payload.append(evidence)
        payload.append("\n" + MODEL_GLOSSARY)
        if analogues:
            payload.append("\nHISTORICAL ANALOGUES:\n" + "\n".join(f"- {a}" for a in analogues))
        if history:
            convo = "\n".join(f"{role.upper()}: {text}" for role, text in history[-6:])
            payload.append("\nCONVERSATION SO FAR:\n" + convo)
        payload.append(f"\nUSER QUESTION: {question}")
        directives = []
        if topic:
            directives.append(f"FOCUS: this question is about the {topic}. Keep your ENTIRE answer "
                              f"about the {topic}; do not switch to or describe a different model.")
        if depth:
            directives.append("DEPTH REQUEST: give the COMPLETE detailed answer now — state the exact "
                              "formula from the MODEL REFERENCE, then show how it applies to this "
                              "company using the model's component values and final score EXACTLY as "
                              "given in the EVIDENCE (they ARE provided on the 'components' line under "
                              "the model — quote them verbatim; never say a component is 'not "
                              "provided' and never invent or recompute raw financial figures). Do NOT "
                              "end with another 'want to go deeper' offer; just answer fully.")
        if directives:
            payload.append("\n" + "\n".join(directives))
        chain = self._persona_chain(INTERROGATION_SYSTEM, num_predict=self._answer_cap(question, history))
        with _LLM_LOCK:
            yield from chain.stream({"input": "\n".join(payload)})

    # ---- 4. Conversational host (intent parsing + greetings) ---------------------------

    @staticmethod
    def _parse_intent_json(raw: str, utterance: str) -> dict:
        """Tolerantly parse the intent JSON; fall back to treating the message as a company query."""
        import json
        import re
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
            intent = data.get("intent")
            if intent in ("analyze", "followup", "help", "unknown"):
                return {"intent": intent, "company_query": data.get("company_query"),
                        "ticker_guess": data.get("ticker_guess")}
        except Exception as e:
            logger.warning(f"intent parse failed ({e}); raw={raw!r}")
        # Fallback: assume the user named a company (resolver + confirm step will catch garbage).
        return {"intent": "analyze", "company_query": utterance.strip(), "ticker_guess": None}

    def interpret(self, utterance: str, has_active: bool = False) -> dict:
        """Classify a user message into {intent, company_query}. company_query is a real, corrected
        company NAME (the resolver maps it to a validated SEC ticker; the model never gives the ticker)."""
        ctx = ("An analysis is currently on screen." if has_active
               else "No analysis is on screen yet.")
        raw = self.complete(INTENT_SYSTEM, f"({ctx})\nUSER MESSAGE: {utterance}",
                            temperature=0.0, num_predict=120)
        return self._parse_intent_json(raw, utterance)

    # Name handling is per-utterance only (mic-safe): a public demo passes the mic between people,
    # so a name from an earlier turn may belong to a DIFFERENT speaker. Only address someone by name
    # when THEY state it in the current message; never carry it over and never guess/invent one.
    _NAME_EXPLICIT_RE = re.compile(r"(?:\bmy name is\b|\bcall me\b)\s+([A-Za-z][A-Za-z'\-]*)", re.IGNORECASE)
    _NAME_IM_RE = re.compile(r"(?:\bI am\b|\bI'?m\b)\s+([A-Z][a-zA-Z'\-]*)")  # name must be Capitalized
    _NAME_STOPWORDS = {"looking", "interested", "trying", "here", "not", "sorry", "just", "going",
                       "hoping", "wondering", "curious", "new", "from", "really", "also", "still",
                       "tired", "good", "fine", "ready", "sure", "happy", "glad", "excited", "back",
                       "afraid", "done", "lost", "confused", "thinking", "hoping", "gonna"}

    @classmethod
    def _extract_name(cls, utterance: str):
        """Conservatively pull a self-introduced first name from the CURRENT message only. Returns a
        clean capitalized name or None. Fires only on explicit intros ('my name is X', 'I'm X')."""
        if not utterance:
            return None
        for rx in (cls._NAME_EXPLICIT_RE, cls._NAME_IM_RE):
            m = rx.search(utterance)
            if m:
                name = m.group(1).strip("'-")
                if len(name) >= 2 and name.isalpha() and name.lower() not in cls._NAME_STOPWORDS:
                    return name.capitalize()
        return None

    @classmethod
    def _name_directive(cls, utterance: str) -> str:
        """A prompt directive that lets the model use a name ONLY if it's in the current message."""
        name = cls._extract_name(utterance)
        if name:
            return f"The user just introduced themselves as {name}; greet them as {name}."
        return ("The user did NOT give their name in this message; do NOT address them by any name, "
                "do NOT reuse a name from earlier in the conversation, and NEVER guess or invent one.")

    def stream_host_reply(self, utterance: str, history: list = None):
        """Streams a warm host reply for greetings / help / small talk."""
        payload = []
        if history:
            payload.append("CONVERSATION SO FAR:\n"
                           + "\n".join(f"{role.upper()}: {text}" for role, text in history[-6:]))
        payload.append(f"USER: {utterance}\n{self._name_directive(utterance)}")
        chain = self._persona_chain(HOST_CASSANDRA_SYSTEM, temperature=0.4, num_predict=160)
        with _LLM_LOCK:
            yield from chain.stream({"input": "\n".join(payload)})

    # Verdict tier -> how the confirm bubble should frame the (already-computed) result, so the tone
    # is proportionate: no presumed "fraud" for a healthy company; serious language only when earned.
    _CONFIRM_TONE = {
        "clean": "The forensic models came back CLEAN — no red flags fired. Strike a reassuring, "
                 "matter-of-fact tone: it looks broadly healthy, with the details just below. Do NOT "
                 "use the words 'fraud' or 'investigation'.",
        "watch": "The models look mostly healthy with just one or two MINOR items worth a glance. "
                 "Strike a calm, measured tone: largely sound, with a small thing or two to note "
                 "below. Do NOT imply fraud or use alarming language.",
        "high-risk": "The models flagged SERIOUS red flags. Strike a serious (not sensational) tone: "
                     "you're seeing real concerns worth a close look just below.",
    }

    def stream_host_confirm(self, utterance: str, company: str, ticker: str, history: list = None,
                            tone: str = None):
        """Streams a brief, VERDICT-AWARE confirmation before the analysis renders. Grounded on the
        already-validated company/ticker (the model only phrases it) and on `tone` (clean/watch/
        high-risk) so the framing is proportionate — never presuming 'fraud' for a healthy company."""
        payload = []
        if history:
            payload.append("CONVERSATION SO FAR:\n"
                           + "\n".join(f"{role.upper()}: {text}" for role, text in history[-6:]))
        tone_directive = self._CONFIRM_TONE.get(
            tone, "Keep the framing neutral: you are simply pulling up the forensic review. Do NOT "
                  "presume fraud or use alarming language.")
        payload.append(
            f'The user said: "{utterance}". You have identified the company as {company} ({ticker}) '
            f"and are about to show its forensic analysis. {tone_directive} {self._name_directive(utterance)} "
            f"Reply in EXACTLY 1-2 short, warm sentences, then STOP: acknowledge them, confirm you're "
            f"pulling up {company} ({ticker}) with the results just below, and give at most a brief "
            f"qualitative read matching the tone above. Use exactly that company name and ticker. Do "
            f"NOT preview or list the results, and do NOT mention any specific metric, line item, "
            f"ratio, figure, or score (e.g. receivables, accruals, margins, debt) — those render "
            f"separately below.")
        chain = self._persona_chain(HOST_CASSANDRA_SYSTEM, temperature=0.4, num_predict=100)
        with _LLM_LOCK:
            yield from chain.stream({"input": "\n".join(payload)})

    # Reason -> plain explanation for why a company can't be analyzed.
    _DECLINE_REASON = {
        "ifrs": "it reports under international (IFRS) accounting standards, which my US-GAAP models "
                "don't read yet",
        "no_usgaap": "it doesn't file the US-GAAP financial statements my models need",
        "few_years": "it doesn't have at least two years of comparable annual filings yet "
                     "(e.g. a recent IPO)",
        "not_found": "I couldn't find its filings with the SEC",
    }

    def stream_host_decline(self, utterance: str, company: str, reason: str, history: list = None):
        """Streams a warm, grounded 'I can't analyze this one' reply with the real reason + a nudge
        to try a US-GAAP filer. `reason` is a code (see _DECLINE_REASON) or a literal phrase."""
        why = self._DECLINE_REASON.get(reason, reason)
        payload = []
        if history:
            payload.append("CONVERSATION SO FAR:\n"
                           + "\n".join(f"{role.upper()}: {text}" for role, text in history[-6:]))
        payload.append(
            f'The user said: "{utterance}". You identified the company as {company}, but you CANNOT '
            f"analyze it because {why}. {self._name_directive(utterance)} Reply in 1-2 warm, "
            f"apologetic sentences: acknowledge them, say briefly why you can't run {company}, and "
            f"suggest they try a US-listed company that files in US dollars (e.g. Apple, Microsoft, "
            f"or Nvidia). Do not invent any analysis or numbers.")
        chain = self._persona_chain(HOST_CASSANDRA_SYSTEM, temperature=0.4, num_predict=90)
        with _LLM_LOCK:
            yield from chain.stream({"input": "\n".join(payload)})


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    from forensics import ForensicEngine
    from curated_cases import CuratedCaseLoader

    eng = ForensicEngine()
    loader = CuratedCaseLoader()
    years = loader.get_company_years("SUNBEAM")
    cur, pri = years[1997], years[1996]

    models = {
        "m_score": eng.calculate_m_score(cur, pri),
        "z_score": eng.calculate_z_score(cur),
        "sloan": eng.calculate_sloan_ratio(cur, pri),
        "piotroski": eng.calculate_piotroski_f_score(cur, pri),
        "leverage": eng.calculate_leverage(cur, pri),
    }
    verdict = eng.calculate_verdict(**models)
    writer = ForensicMemoWriter()
    company = "Sunbeam Corp (FY1997 vs FY1996)"

    print("\n========== FORENSIC RISK MEMO ==========")
    try:
        print(writer.generate_memo(company, verdict, models, cur))
    except Exception as e:
        print(f"[Ollama unavailable] {type(e).__name__}: {e}")
        raise SystemExit

    print("\n========== COMMITTEE DEBATE ==========")
    current = None
    for speaker, chunk in writer.stream_debate(company, verdict, models, cur, length="brief"):
        if speaker != current:
            print(f"\n\n### {speaker}\n", flush=True)
            current = speaker
        print(chunk, end="", flush=True)
    print()
