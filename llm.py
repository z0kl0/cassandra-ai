import os
import logging
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

# CASSANDRA's "Forensic Skeptic" persona. The single most important constraint of the
# whole project lives here: the LLM EXPLAINS the deterministic results, it never produces
# or alters a number. The verdict and every figure are computed in forensics.py and passed
# in as ground truth; the model's only job is plain-English narrative + historical analogy.
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


class ForensicMemoWriter:
    """
    Turns the forensic engine's deterministic output into a cited Forensic Risk Memo via a
    local Ollama model, using a LangChain (LCEL) chain: prompt | llm | parser.
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

        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=temperature,       # low: factual consistency over creativity
            num_predict=num_predict,       # cap output so CPU latency stays acceptable
            keep_alive=self.keep_alive,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", CASSANDRA_SYSTEM), ("human", "{evidence}")]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

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

        lines.append("\nDETERMINISTIC MODEL RESULTS (ground truth; do not alter):")
        m = models.get("m_score")
        if m and m.get("M_Score") is not None:
            lines.append(f"- Beneish M-Score: {m['M_Score']} -> "
                         f"{'MANIPULATOR' if m.get('Is_Manipulator') else 'no manipulation'} "
                         f"(threshold: > -2.22 signals manipulation) [confidence: {m.get('Confidence')}]")
        z = models.get("z_score")
        if z and z.get("Z_Score") is not None:
            lines.append(f"- Altman Z-Score: {z['Z_Score']} -> {z.get('Status')} "
                         f"(< 1.81 distress, 1.81-2.99 grey, > 2.99 safe) [confidence: {z.get('Confidence')}]")
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
                         f"(> 0.015 = digit-distribution anomaly)")

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

    # ---- LLM call ----------------------------------------------------------------------

    def generate_memo(self, company: str, verdict: dict, models: dict,
                      financials: dict = None, analogues: list = None) -> str:
        """Generates the Forensic Risk Memo. Raises on connection/model errors (caller handles)."""
        evidence = self.build_evidence(company, verdict, models, financials, analogues)
        logger.info(f"Generating forensic memo for {company} via {self.model}...")
        return self.chain.invoke({"evidence": evidence})

    def stream_memo(self, company: str, verdict: dict, models: dict,
                    financials: dict = None, analogues: list = None):
        """
        Yields the memo in text chunks as the model generates them. Lets the UI render
        token-by-token (st.write_stream) so the ~1-3 min CPU latency feels responsive.
        """
        evidence = self.build_evidence(company, verdict, models, financials, analogues)
        logger.info(f"Streaming forensic memo for {company} via {self.model}...")
        yield from self.chain.stream({"evidence": evidence})


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

    # Optional RAG analogues (degrade gracefully if Chroma/embeddings unavailable).
    analogues = None
    try:
        from rag import FraudCorpusRAG
        rag = FraudCorpusRAG()
        q = "; ".join(verdict.get("Reasons", [])) or "earnings manipulation"
        res = rag.query_cases(q, n_results=2)
        analogues = res.get("documents", [[]])[0] or None
    except Exception as e:
        logger.info(f"RAG analogues skipped: {e}")

    writer = ForensicMemoWriter()
    evidence = writer.build_evidence("Sunbeam Corp (FY1997 vs FY1996)", verdict, models, cur, analogues)
    print("\n========== DETERMINISTIC EVIDENCE (input to LLM) ==========")
    print(evidence)

    print("\n========== FORENSIC RISK MEMO (LLM-generated) ==========")
    try:
        print(writer.generate_memo("Sunbeam Corp (FY1997 vs FY1996)", verdict, models, cur, analogues))
    except Exception as e:
        print(f"[Ollama unavailable - memo not generated] {type(e).__name__}: {e}")
        print("Start Ollama and `ollama pull llama3`, then re-run.")
