import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class CuratedCaseLoader:
    """
    Loads curated, point-in-time financial cases from a CSV and emits dicts that are
    drop-in compatible with ForensicEngine's scorers (same shape as
    ForensicEngine.extract_financials output).

    The CSV exists because the SEC XBRL companyfacts API cannot serve every case:
      * Pre-XBRL classics (Enron, WorldCom, Sunbeam) predate mandatory XBRL (~2009).
      * Foreign filers not in EDGAR XBRL (Wirecard — a Frankfurt issuer).
      * Frauds whose fabricated figures were never filed as audited XBRL at all
        (Luckin's 2019 numbers lived in interim press releases until restated).

    Each row is one company-fiscal-year. The figures must be transcribed from primary
    sources (10-K/annual report, SEC AAER/litigation release) and cited in `source_url`;
    a blank financial cell is treated as missing data (not zero) and lowers the case's
    reported coverage, exactly like a missing XBRL tag — so the engine never presents
    fabricated precision.
    """

    # The financial concepts the engine consumes, matching ForensicEngine.tag_map keys.
    # GrossMargin and WorkingCapital are derived (like extract_financials does), so they
    # are NOT columns in the CSV.
    CONCEPTS = [
        "Sales", "CostOfGoodsSold", "Receivables", "CurrentAssets", "PropertyPlantEquipment",
        "Securities", "Assets", "Depreciation", "SGA", "CurrentLiabilities", "LongTermDebt",
        "NetIncome", "OperatingCashFlow", "TotalLiabilities", "RetainedEarnings", "EBIT",
        "StockholdersEquity", "SharesOutstanding", "IncomeTaxExpense", "InterestExpense",
    ]
    META_COLUMNS = [
        "case_id", "company", "ticker", "fiscal_year", "period_end", "category",
        "fraud_type", "data_source", "source_url", "notes",
    ]

    def __init__(self, csv_path: str = None):
        self.csv_path = Path(csv_path or "./data/fraud_cases.csv")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Curated case file not found: {self.csv_path}")
        # Keep blanks as NaN so we can distinguish "missing" from a real 0.
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} curated case-years from {self.csv_path}")

    def list_cases(self) -> pd.DataFrame:
        """Returns the metadata (one row per case-year) for browsing the corpus."""
        cols = [c for c in self.META_COLUMNS if c in self.df.columns]
        return self.df[cols].copy()

    def _row_to_financials(self, row: pd.Series) -> dict:
        """Converts one CSV row into an engine-compatible financials dict."""
        source = str(row.get("source_url", "") or "curated")
        financials, provenance, found = {}, {}, 0
        for concept in self.CONCEPTS:
            val = row.get(concept)
            if pd.isna(val):
                financials[concept] = 0.0
                provenance[concept] = None
            else:
                financials[concept] = float(val)
                provenance[concept] = f"curated:{source}"
                found += 1

        # Derived metrics — mirror ForensicEngine.extract_financials exactly.
        financials["GrossMargin"] = financials["Sales"] - financials["CostOfGoodsSold"]
        financials["WorkingCapital"] = financials["CurrentAssets"] - financials["CurrentLiabilities"]

        financials["_provenance"] = provenance
        financials["_coverage"] = round(found / len(self.CONCEPTS), 3)
        return financials

    def get_company_years(self, case_id: str) -> dict:
        """
        Returns {fiscal_year: financials_dict} for a curated case, keyed by integer year.
        Feed two consecutive years to ForensicEngine (current, prior) just like live data.
        """
        rows = self.df[self.df["case_id"] == case_id]
        if rows.empty:
            raise KeyError(f"No curated case with case_id={case_id!r}")
        return {int(r["fiscal_year"]): self._row_to_financials(r) for _, r in rows.iterrows()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from forensics import ForensicEngine

    loader = CuratedCaseLoader()
    eng = ForensicEngine()

    print("\n--- Curated corpus ---")
    print(loader.list_cases().to_string(index=False))

    # Validate the loader -> engine path against the synthetic textbook case, whose
    # expected scores are known (M-Score -0.084, Z-Score 2.635).
    years = loader.get_company_years("TEXTBOOK_DEMO")
    cur, pri = years[2023], years[2022]
    m = eng.calculate_m_score(cur, pri)
    z = eng.calculate_z_score(cur)
    lev = eng.calculate_leverage(cur, pri)
    print("\n--- TEXTBOOK_DEMO through engine (expect M=-0.084, Z=2.635) ---")
    print(f"Coverage: {cur['_coverage']} | M-Score: {m['M_Score']} | Z-Score: {z['Z_Score']} "
          f"| Equity Multiplier: {lev['Equity_Multiplier']}x")

    # Corpus-wide backtest view: run every populated case (>=2 years with data) through
    # the full model suite. Each fraud should be caught by at least one model.
    print("\n--- Curated corpus: full model suite + aggregated verdict ---")
    header = f"{'case':26} {'M-Score':>8} {'Manip':>6} {'Z':>9} {'Sloan':>8} {'EqMult':>8} {'VERDICT':>11} {'Cov':>5}"
    print(header)
    print("-" * len(header))
    for case_id in sorted(loader.df["case_id"].unique()):
        yrs = loader.get_company_years(case_id)
        if len(yrs) < 2:
            continue
        keys = sorted(yrs)
        cur, pri = yrs[keys[-1]], yrs[keys[-2]]
        if not cur.get("Assets"):  # metadata-only row, nothing to score
            continue
        m = eng.calculate_m_score(cur, pri)
        z = eng.calculate_z_score(cur)
        s = eng.calculate_sloan_ratio(cur, pri)
        lev = eng.calculate_leverage(cur, pri)
        p = eng.calculate_piotroski_f_score(cur, pri)
        v = eng.calculate_verdict(m_score=m, z_score=z, sloan=s, piotroski=p, leverage=lev)
        label = f"{case_id} {keys[-2]}>{keys[-1]}"
        verdict_col = f"{v['Emoji']} {v['Verdict']}"
        print(f"{label:26} {str(m['M_Score']):>8} {str(m['Is_Manipulator']):>6} "
              f"{str(z['Z_Score']):>9} {str(s['Sloan_Ratio']):>8} "
              f"{str(lev['Equity_Multiplier']):>8} {verdict_col:>11} {str(cur['_coverage']):>5}")
