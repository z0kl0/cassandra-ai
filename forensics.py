import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ForensicEngine:
    """
    Deterministic forensic accounting engine.
    Computes Beneish M-Score, Altman Z-Score, and other red flags from SEC XBRL data.
    """
    # Balance-sheet (instant / point-in-time) concepts vs. income/cash-flow (flow / period)
    # concepts. The distinction drives how we identify the correct annual XBRL data point:
    # flows use the "CY{year}" frame, instants use "CY{year}Q4I".
    INSTANT_CONCEPTS = {
        "Assets", "CurrentAssets", "PropertyPlantEquipment", "Securities", "Receivables",
        "CurrentLiabilities", "LongTermDebt", "TotalLiabilities", "RetainedEarnings",
        "StockholdersEquity", "SharesOutstanding",
    }
    SHARE_CONCEPTS = {"SharesOutstanding"}

    # Annual report forms whose fp=="FY" point carries the full-year figure. Includes
    # foreign private issuers (20-F) and Canadian filers (40-F), not just domestic 10-K,
    # so the engine works on SEC-listed ADRs such as Luckin Coffee. Amendments (/A) are
    # included so restatements are picked up (most-recently-filed wins); use `as_of_date`
    # to get the original as-filed figure for point-in-time backtests.
    ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
    # Interim filings that must never be mistaken for an annual value (domestic 10-Q,
    # foreign 6-K). Flow spans are also length-checked, but excluding these guards
    # against picking a mid-year balance-sheet (instant) figure.
    INTERIM_FORMS = {"10-Q", "6-K"}

    def __init__(self):
        # SEC XBRL tags are notoriously inconsistent (the project's #1 risk). We map
        # standard accounting concepts to a priority-ordered list of candidate tags;
        # the resolver tries each in order and resolves per-year, so a concept still
        # resolves when the underlying tag changes between years (e.g. a filer moving
        # from Revenues to RevenueFromContractWithCustomerExcludingAssessedTax).
        self.tag_map = {
            "Assets": ["Assets"],
            "CurrentAssets": ["AssetsCurrent"],
            "PropertyPlantEquipment": [
                "PropertyPlantAndEquipmentNet",
                # Filers (e.g. Alphabet from FY2025) fold finance-lease ROU assets into the PP&E
                # line under this tag; without it PP&E reads as 0 and blows up AQI/DEPI.
                "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
            ],
            "Securities": ["MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent"],
            "Sales": [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "Revenues", "SalesRevenueNet", "RevenuesNetOfInterestExpense",
            ],
            "CostOfGoodsSold": [
                "CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold", "CostOfServices",
            ],
            "Receivables": [
                "AccountsReceivableNetCurrent", "AccountsAndNotesReceivableNet", "ReceivablesNetCurrent",
            ],
            "CurrentLiabilities": ["LiabilitiesCurrent"],
            "LongTermDebt": [
                "LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations",
            ],
            "Depreciation": ["DepreciationDepletionAndAmortization", "Depreciation"],
            "SGA": ["SellingGeneralAndAdministrativeExpense"],
            "NetIncome": ["NetIncomeLoss", "ProfitLoss"],
            "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
            "TotalLiabilities": ["Liabilities"],
            "RetainedEarnings": ["RetainedEarningsAccumulatedDeficit", "RetainedEarnings"],
            "EBIT": ["OperatingIncomeLoss"],
            "IncomeTaxExpense": ["IncomeTaxExpenseBenefit"],
            "InterestExpense": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
            "StockholdersEquity": [
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ],
            "SharesOutstanding": [
                "CommonStockSharesOutstanding",
                "WeightedAverageNumberOfSharesOutstandingBasic",
            ],
        }

        # Derivations: compute a concept as a SUM of component tag-groups when its direct
        # tag_map lookup fails. Each component is a candidate-tag list; the first group is the
        # required base, the rest are added when present. Used only as a fallback, so direct
        # tags always win. (See _derive_value.)
        self.derivations = {
            # Many filers (Alphabet, Luckin, ...) report G&A and Selling&Marketing separately
            # instead of a combined SG&A line.
            "SGA": [
                ["GeneralAndAdministrativeExpense"],
                ["SellingAndMarketingExpense", "MarketingExpense", "SellingExpense"],
            ],
            # EBIT add-back when OperatingIncomeLoss is absent: NetIncome + income tax + interest.
            "EBIT": [
                ["NetIncomeLoss", "ProfitLoss"],
                ["IncomeTaxExpenseBenefit"],
                ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
            ],
        }

    def _is_annual_flow(self, dp: dict) -> bool:
        """True if a flow data point spans roughly a full fiscal year (~365 days)."""
        if "start" not in dp:
            return False
        days = (pd.to_datetime(dp["end"]) - pd.to_datetime(dp["start"])).days
        return days >= 300

    def _select_annual_point(self, data_points: list, year: int, instant: bool, as_of_date: str = None):
        """
        Picks the single best annual (10-K / full-year) data point for fiscal `year`,
        returning (value, period_end, filed) or None.

        Selection priority:
          1. The annual report (form in ANNUAL_FORMS — 10-K / 20-F / 40-F, fp=FY) point
             for that fiscal year. An annual report embeds several years of comparatives
             all tagged with the filing's `fy`, so we take the one with the latest `end`
             (the fiscal year-end, not a prior-year comparative). This is correct even
             for non-December fiscal years (e.g. MSFT's June 30 year-end), where calendar
             `frame`s do NOT align.
          2. Fallback to the calendar `frame` (CY{year} for flows, CY{year}Q4I for
             instants) for filers/points without clean annual-report tagging.
          3. Backstop: any non-interim point whose `end` falls in the year.
        Among ties the most recently `filed` wins (favors the as-reported annual report
        and makes restatement duplicates deterministic). Interim 10-Q/6-K points are
        never accepted as annual values.

        `as_of_date` ("YYYY-MM-DD") enables point-in-time backtests: only data points
        filed on or before that date are considered, so the engine sees a company as it
        was reported at decision time and never uses a later restatement (no look-ahead
        bias). Filing dates are ISO strings, so a lexicographic comparison is correct.
        Combined with the most-recently-filed tiebreak, this yields the value as it
        stood on `as_of_date`.
        """
        if as_of_date:
            data_points = [dp for dp in data_points if dp.get("filed", "") <= as_of_date]

        def value_of(dp):
            return float(dp["val"]), dp.get("end", ""), dp.get("filed", "")

        # 1. Annual report for this fiscal year -> fiscal year-end (max `end`).
        annual = [
            dp for dp in data_points
            if dp.get("form") in self.ANNUAL_FORMS and dp.get("fp") == "FY" and dp.get("fy") == year
            and (instant or self._is_annual_flow(dp))
        ]
        if annual:
            return value_of(max(annual, key=lambda dp: (dp.get("end", ""), dp.get("filed", ""))))

        # 2. Calendar frame fallback (cleanest for December year-end filers).
        target_frame = f"CY{year}Q4I" if instant else f"CY{year}"
        framed = [dp for dp in data_points if dp.get("frame") == target_frame]
        if framed:
            return value_of(max(framed, key=lambda dp: dp.get("filed", "")))

        # 3. Backstop for filers without clean frames/forms. Never accept interim data.
        backstop = []
        for dp in data_points:
            if dp.get("form") in self.INTERIM_FORMS:
                continue
            if str(year) not in dp.get("end", ""):
                continue
            if instant or self._is_annual_flow(dp):
                backstop.append(dp)
        if backstop:
            return value_of(max(backstop, key=lambda dp: (dp.get("end", ""), dp.get("filed", ""))))
        return None

    def _resolve_tags(self, facts: dict, tag_list: list, year: int, instant: bool,
                      preferred_unit: str = "USD", as_of_date: str = None):
        """
        Resolves the first tag in `tag_list` that has an annual value for `year`, returning
        (value, provenance_string) or (None, None). Core lookup shared by direct concept
        resolution and derivation components.
        """
        all_facts = facts.get("facts", {})
        for tag in tag_list:
            tag_data = all_facts.get("us-gaap", {}).get(tag) or all_facts.get("dei", {}).get(tag)
            if not tag_data:
                continue
            units_dict = tag_data.get("units", {})
            if not units_dict:
                continue
            unit = preferred_unit if preferred_unit in units_dict else next(iter(units_dict))
            result = self._select_annual_point(units_dict[unit], year, instant, as_of_date)
            if result is not None:
                value, period_end, filed = result
                return value, f"{tag} [{unit}] end={period_end} filed={filed}"
        return None, None

    def _get_value_for_year(self, facts: dict, concept: str, year: int, as_of_date: str = None):
        """
        Resolves a single concept to its annual value for `year` via its `tag_map` candidates,
        preferring the expected unit (USD for money, shares for share counts).
        `as_of_date` is forwarded to enable point-in-time resolution.
        """
        instant = concept in self.INSTANT_CONCEPTS
        preferred_unit = "shares" if concept in self.SHARE_CONCEPTS else "USD"
        return self._resolve_tags(facts, self.tag_map[concept], year, instant, preferred_unit, as_of_date)

    def _derive_value(self, facts: dict, concept: str, year: int, as_of_date: str = None):
        """
        Computes a concept as a SUM of component tag-groups when its direct tags are absent
        (e.g. SG&A = G&A + Selling&Marketing for split-reporters; EBIT = NetIncome + tax +
        interest). Each component is a candidate-tag list resolved like a normal concept. The
        FIRST component group is required (the base); the rest are added when present. Returns
        (value, provenance) or (None, None). All components treated as USD flows.
        """
        spec = self.derivations.get(concept)
        if not spec:
            return None, None
        base_val, base_prov = self._resolve_tags(facts, spec[0], year, instant=False, as_of_date=as_of_date)
        if base_val is None:
            return None, None
        total, parts = base_val, [base_prov.split(" [")[0]]
        for comp in spec[1:]:
            val, prov = self._resolve_tags(facts, comp, year, instant=False, as_of_date=as_of_date)
            if val is not None:
                total += val
                parts.append(prov.split(" [")[0])
        return total, "derived: " + " + ".join(parts)

    def extract_financials(self, facts: dict, year: int, as_of_date: str = None) -> dict:
        """
        Extracts all standardized financial figures for a given year.

        Missing concepts are kept at 0.0 so the math never raises, but they are
        tracked: `_provenance` records the resolving tag per concept (None if
        missing) and `_coverage` is the fraction of concepts actually found. This
        supports the memo's "cite the exact figure" requirement and lets the
        scorers downgrade confidence when key inputs are absent.

        For point-in-time backtests, pass `as_of_date` ("YYYY-MM-DD"): only filings
        made on or before that date are used, so the engine reflects what was known
        at decision time and never sees a later restatement. Run both the current and
        prior year with the same `as_of_date` before feeding the scorers.
        """
        financials = {}
        provenance = {}
        found = 0
        for concept in self.tag_map:
            value, source = self._get_value_for_year(facts, concept, year, as_of_date)
            if value is None:
                financials[concept] = 0.0
                provenance[concept] = None
            else:
                financials[concept] = value
                provenance[concept] = source
                found += 1

        # Derivation fallbacks: for any concept still unresolved, compute it from a sum of
        # component tags (e.g. SG&A = G&A + Selling&Marketing; EBIT = NetIncome + tax + interest).
        # Direct tags always take priority -- this only fills genuine gaps.
        for concept in self.derivations:
            if provenance.get(concept) is None:
                value, source = self._derive_value(facts, concept, year, as_of_date)
                if value is not None:
                    financials[concept] = value
                    provenance[concept] = source
                    found += 1

        # Derived base metrics
        financials["GrossMargin"] = financials["Sales"] - financials["CostOfGoodsSold"]
        financials["WorkingCapital"] = financials.get("CurrentAssets", 0.0) - financials.get("CurrentLiabilities", 0.0)

        financials["_provenance"] = provenance
        financials["_coverage"] = round(found / len(self.tag_map), 3)
        return financials

    @staticmethod
    def _confidence(dicts: list, required: list) -> str:
        """
        Returns "Low" if any required concept is missing from a year's data
        (per its `_provenance` map), else "High". When `_provenance` is absent
        (e.g. hand-built test dicts), assumes the data is complete.
        """
        for d in dicts:
            prov = d.get("_provenance")
            if prov is None:
                continue
            for concept in required:
                if prov.get(concept) is None:
                    return "Low"
        return "High"

    def calculate_m_score(self, current: dict, prior: dict) -> dict:
        """
        Calculates the Beneish M-Score for Earnings Manipulation.
        M-Score > -2.22 indicates a high probability of manipulation.
        """
        try:
            # Safe ratio: Beneish indices are constructed so that 1.0 means "no change".
            # When an input is missing (denominator 0), fall back to that neutral value so
            # a single absent tag (e.g. a filer that doesn't report combined SG&A) degrades
            # one component instead of nulling the whole score. Coverage is flagged via
            # Confidence below.
            def r(num, den, default=1.0):
                return num / den if den else default

            # 1. Days Sales in Receivables Index (DSRI)
            dsri = r(r(current["Receivables"], current["Sales"]), r(prior["Receivables"], prior["Sales"]))

            # 2. Gross Margin Index (GMI)
            gmi = r(r(prior["GrossMargin"], prior["Sales"]), r(current["GrossMargin"], current["Sales"]))

            # 3. Asset Quality Index (AQI)
            aqi_current = 1 - r(current["CurrentAssets"] + current["PropertyPlantEquipment"] + current["Securities"], current["Assets"], 0.0)
            aqi_prior = 1 - r(prior["CurrentAssets"] + prior["PropertyPlantEquipment"] + prior["Securities"], prior["Assets"], 0.0)
            aqi = r(aqi_current, aqi_prior)

            # 4. Sales Growth Index (SGI)
            sgi = r(current["Sales"], prior["Sales"])

            # 5. Depreciation Index (DEPI)
            dep_rate_prior = r(prior["Depreciation"], prior["PropertyPlantEquipment"] + prior["Depreciation"])
            dep_rate_current = r(current["Depreciation"], current["PropertyPlantEquipment"] + current["Depreciation"])
            depi = r(dep_rate_prior, dep_rate_current)

            # 6. Sales, General and Administrative Expenses Index (SGAI)
            sgai = r(r(current["SGA"], current["Sales"]), r(prior["SGA"], prior["Sales"]))

            # 7. Leverage Index (LVGI)
            lvg_current = r(current["CurrentLiabilities"] + current["LongTermDebt"], current["Assets"])
            lvg_prior = r(prior["CurrentLiabilities"] + prior["LongTermDebt"], prior["Assets"])
            lvgi = r(lvg_current, lvg_prior)

            # 8. Total Accruals to Total Assets (TATA)
            tata = r(current["NetIncome"] - current["OperatingCashFlow"], current["Assets"], 0.0)

            # M-Score Formula
            m_score = (
                -4.84 
                + 0.920 * dsri 
                + 0.528 * gmi 
                + 0.404 * aqi 
                + 0.892 * sgi 
                + 0.115 * depi 
                - 0.172 * sgai 
                + 4.679 * tata 
                - 0.327 * lvgi
            )
            
            is_manipulator = m_score > -2.22
            
            confidence = self._confidence(
                [current, prior],
                ["Sales", "Receivables", "Assets", "NetIncome", "OperatingCashFlow"],
            )

            return {
                "M_Score": round(m_score, 3),
                "Is_Manipulator": is_manipulator,
                "Confidence": confidence,
                "Components": {
                    "DSRI": round(dsri, 3),
                    "GMI": round(gmi, 3),
                    "AQI": round(aqi, 3),
                    "SGI": round(sgi, 3),
                    "DEPI": round(depi, 3),
                    "SGAI": round(sgai, 3),
                    "LVGI": round(lvgi, 3),
                    "TATA": round(tata, 3)
                }
            }
        except ZeroDivisionError:
            logger.error("Zero division error during M-Score calculation due to missing data.")
            return {"M_Score": None, "Is_Manipulator": None, "Components": {}}
            
    def calculate_z_score(self, current: dict) -> dict:
        """
        Calculates the Altman Z-Score for financial distress.
        Z-Score < 1.81 indicates high risk of distress/bankruptcy.
        Note: Uses Book Value of Equity as a proxy for Market Value.
        """
        try:
            assets = current.get("Assets", 0)
            if not assets:
                return {"Z_Score": None, "Status": "Missing Data", "Components": {}}

            # X1 = Working Capital / Total Assets
            x1 = current.get("WorkingCapital", 0) / assets
            # X2 = Retained Earnings / Total Assets
            x2 = current.get("RetainedEarnings", 0) / assets
            # X3 = EBIT / Total Assets
            x3 = current.get("EBIT", 0) / assets
            
            liabilities = current.get("TotalLiabilities", 0)
            equity = current.get("StockholdersEquity", 0) 
            # X4 = Market Value of Equity / Total Liabilities (Using Book Value as proxy)
            x4 = equity / liabilities if liabilities else 1.0
            
            # X5 = Sales / Total Assets
            x5 = current.get("Sales", 0) / assets

            z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
            
            if z_score < 1.81:
                status = "Distress"
            elif z_score < 2.99:
                status = "Grey Zone"
            else:
                status = "Safe"
                
            confidence = self._confidence(
                [current], ["Assets", "RetainedEarnings", "EBIT", "StockholdersEquity", "Sales"]
            )

            return {
                "Z_Score": round(z_score, 3),
                "Status": status,
                "Confidence": confidence,
                "Components": {"X1": round(x1, 3), "X2": round(x2, 3), "X3": round(x3, 3), "X4": round(x4, 3), "X5": round(x5, 3)}
            }
        except ZeroDivisionError:
            logger.error("Zero division error during Z-Score calculation.")
            return {"Z_Score": None, "Status": "Error", "Components": {}}

    def calculate_leverage(self, current: dict, prior: dict = None) -> dict:
        """
        Computes balance-sheet leverage as a distress red flag, covering the blind spot
        where Beneish (earnings manipulation) and Altman Z (which Altman explicitly
        excluded financial institutions from) miss over-leveraged firms.

        Reports:
          * Equity_Multiplier = Total Assets / Stockholders' Equity (a.k.a. financial
            leverage). This is the headline ratio that flagged Lehman (30.7x in 2007).
          * Debt_to_Equity = Total Liabilities / Stockholders' Equity.
          * Trend = year-over-year change in the equity multiplier when `prior` is given;
            *rising* leverage is the real warning sign, not a high static level alone.

        Absolute thresholds are deliberately conservative and sector-aware in wording:
        non-financial firms typically sit at 1.5-3x, banks/broker-dealers structurally
        run ~10-15x, so >20x is dangerous even for a financial. Negative equity is treated
        as the most severe signal (book insolvency).
        """
        try:
            equity = current.get("StockholdersEquity", 0)
            assets = current.get("Assets", 0)
            if not assets:
                return {"Equity_Multiplier": None, "Status": "Missing Data", "Components": {}}

            if equity <= 0:
                # Negative/zero book equity = liabilities exceed assets: book insolvency.
                return {
                    "Equity_Multiplier": None,
                    "Debt_to_Equity": None,
                    "Status": "Negative Equity (Insolvent)",
                    "Trend": None,
                    "Confidence": self._confidence([current], ["Assets", "StockholdersEquity"]),
                    "Components": {"Assets": assets, "Equity": equity},
                }

            equity_mult = assets / equity
            debt_to_equity = current.get("TotalLiabilities", 0) / equity
            ltd_to_equity = current.get("LongTermDebt", 0) / equity

            if equity_mult > 20:
                status = "Extreme Leverage"
            elif equity_mult > 10:
                status = "High Leverage (typical only of financials)"
            elif equity_mult > 4:
                status = "Moderate"
            else:
                status = "Conservative"

            # Trend: rising leverage is the actionable signal.
            trend = None
            if prior:
                prior_equity = prior.get("StockholdersEquity", 0)
                prior_assets = prior.get("Assets", 0)
                if prior_equity > 0 and prior_assets:
                    prior_mult = prior_assets / prior_equity
                    delta = equity_mult - prior_mult
                    direction = "rising" if delta > 0.5 else "falling" if delta < -0.5 else "stable"
                    trend = {
                        "Prior_Equity_Multiplier": round(prior_mult, 2),
                        "Change": round(delta, 2),
                        "Direction": direction,
                    }

            return {
                "Equity_Multiplier": round(equity_mult, 2),
                "Debt_to_Equity": round(debt_to_equity, 2),
                "LongTermDebt_to_Equity": round(ltd_to_equity, 2),
                "Status": status,
                "Trend": trend,
                "Confidence": self._confidence([current], ["Assets", "StockholdersEquity"]),
            }
        except ZeroDivisionError:
            logger.error("Zero division error during leverage calculation.")
            return {"Equity_Multiplier": None, "Status": "Error", "Components": {}}

    def calculate_sloan_ratio(self, current: dict, prior: dict) -> dict:
        """
        Calculates the Sloan Accruals Ratio.
        Ratio = (Net Income - Operating Cash Flow) / Average Total Assets
        Consistently high ratios (> 0.10) can indicate poor earnings quality.
        """
        try:
            avg_assets = (current.get("Assets", 0) + prior.get("Assets", 0)) / 2
            if not avg_assets:
                return {"Sloan_Ratio": None, "Status": "Missing Data"}
                
            ni = current.get("NetIncome", 0)
            ocf = current.get("OperatingCashFlow", 0)
            accruals = ni - ocf
            
            ratio = accruals / avg_assets
            
            if ratio > 0.10:
                status = "High Risk (High Accruals)"
            elif ratio < -0.10:
                status = "Low Risk (Negative Accruals)"
            else:
                status = "Normal"
                
            confidence = self._confidence([current, prior], ["Assets", "NetIncome", "OperatingCashFlow"])

            return {
                "Sloan_Ratio": round(ratio, 4),
                "Status": status,
                "Confidence": confidence,
                "Components": {"Accruals": accruals, "Average_Assets": avg_assets}
            }
        except Exception as e:
            logger.error(f"Error calculating Sloan Ratio: {e}")
            return {"Sloan_Ratio": None, "Status": "Error"}

    def calculate_benford_deviation(self, facts: dict, year: int) -> dict:
        """
        Applies Benford's Law to the first digits of all non-zero numerical facts reported for a given year.
        Calculates Mean Absolute Deviation (MAD). MAD > 0.015 typically indicates nonconformity.
        """
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        first_digits = []
        
        # Benford's Law applies to magnitudes of naturally-occurring monetary amounts. Restrict
        # to USD values only -- including per-share ratios (USD/shares), share counts, durations
        # or "pure" ratios pollutes the distribution and produces false nonconformity on clean
        # filers. Match annual figures (period end in the year) and skip values < 1.
        for tag, data in us_gaap.items():
            pts = data.get("units", {}).get("USD")
            if not pts:
                continue
            for dp in pts:
                if str(year) in dp.get("end", ""):
                    val = abs(dp.get("val", 0))
                    if val >= 1:  # need a meaningful leading digit
                        first_char = str(val)[0]
                        if first_char.isdigit() and first_char != '0':
                            first_digits.append(int(first_char))
                            
        total_count = len(first_digits)
        if total_count < 50:
            return {"MAD": None, "Status": "Insufficient Data", "Sample_Size": total_count}
            
        actual_freq = {i: first_digits.count(i) / total_count for i in range(1, 10)}
        expected_freq = {i: np.log10(1 + 1/i) for i in range(1, 10)}
        
        mad = sum(abs(actual_freq[i] - expected_freq[i]) for i in range(1, 10)) / 9
        
        if mad > 0.015:
            status = "Nonconformity (High Risk)"
        elif mad > 0.012:
            status = "Marginal"
        else:
            status = "Conforms"
            
        return {
            "MAD": round(mad, 4),
            "Status": status,
            "Sample_Size": total_count,
            "Distribution": {str(i): round(actual_freq[i], 3) for i in range(1, 10)}
        }

    def calculate_piotroski_f_score(self, current: dict, prior: dict) -> dict:
        """
        Calculates the Piotroski F-Score (0-9), a measure of fundamental financial
        strength across profitability, leverage/liquidity, and operating efficiency.
        Each of nine binary tests scores 1 point. A high score (>=7) signals a
        fundamentally strong firm; a low score (<=2) signals weakness. It acts as a
        counterweight to the fraud-oriented red-flag models.
        """
        def ratio(num, den):
            return num / den if den else None

        signals = {}

        # --- Profitability (4 points) ---
        roa_cur = ratio(current.get("NetIncome", 0), current.get("Assets", 0))
        roa_pri = ratio(prior.get("NetIncome", 0), prior.get("Assets", 0))
        signals["positive_roa"] = bool(roa_cur is not None and roa_cur > 0)
        signals["positive_ocf"] = bool(current.get("OperatingCashFlow", 0) > 0)
        signals["improving_roa"] = bool(roa_cur is not None and roa_pri is not None and roa_cur > roa_pri)
        # Accruals: operating cash flow should exceed net income (earnings backed by cash).
        signals["accruals_quality"] = bool(
            current.get("OperatingCashFlow", 0) > current.get("NetIncome", 0)
        )

        # --- Leverage, Liquidity & Source of Funds (3 points) ---
        ltd_cur = ratio(current.get("LongTermDebt", 0), current.get("Assets", 0))
        ltd_pri = ratio(prior.get("LongTermDebt", 0), prior.get("Assets", 0))
        signals["lower_leverage"] = bool(ltd_cur is not None and ltd_pri is not None and ltd_cur < ltd_pri)

        cr_cur = ratio(current.get("CurrentAssets", 0), current.get("CurrentLiabilities", 0))
        cr_pri = ratio(prior.get("CurrentAssets", 0), prior.get("CurrentLiabilities", 0))
        signals["higher_current_ratio"] = bool(cr_cur is not None and cr_pri is not None and cr_cur > cr_pri)

        # No new shares issued (dilution). Skipped (None) when share data is unavailable
        # rather than awarding/denying the point arbitrarily.
        shares_cur = current.get("SharesOutstanding", 0)
        shares_pri = prior.get("SharesOutstanding", 0)
        if shares_cur and shares_pri:
            signals["no_dilution"] = bool(shares_cur <= shares_pri)
        else:
            signals["no_dilution"] = None

        # --- Operating Efficiency (2 points) ---
        gm_cur = ratio(current.get("GrossMargin", 0), current.get("Sales", 0))
        gm_pri = ratio(prior.get("GrossMargin", 0), prior.get("Sales", 0))
        signals["higher_gross_margin"] = bool(gm_cur is not None and gm_pri is not None and gm_cur > gm_pri)

        turn_cur = ratio(current.get("Sales", 0), current.get("Assets", 0))
        turn_pri = ratio(prior.get("Sales", 0), prior.get("Assets", 0))
        signals["higher_asset_turnover"] = bool(turn_cur is not None and turn_pri is not None and turn_cur > turn_pri)

        f_score = sum(1 for v in signals.values() if v is True)
        tested = sum(1 for v in signals.values() if v is not None)

        if f_score >= 7:
            status = "Strong"
        elif f_score <= 2:
            status = "Weak"
        else:
            status = "Neutral"

        # Diagnostic split of the canonical nine signals into "level" (absolute health
        # this year) vs. "momentum" (year-over-year improvement). This is our breakdown,
        # not part of Piotroski's original grouping, but it explains why a mature,
        # already-efficient mega-cap can post a mid-range F-Score: high absolute health
        # with little room left to improve. It does NOT alter the canonical F_Score.
        health_keys = ["positive_roa", "positive_ocf", "accruals_quality"]
        momentum_keys = [
            "improving_roa", "lower_leverage", "higher_current_ratio",
            "no_dilution", "higher_gross_margin", "higher_asset_turnover",
        ]
        health = sum(1 for k in health_keys if signals[k] is True)
        momentum = sum(1 for k in momentum_keys if signals[k] is True)

        # Piotroski's original three-bucket structure (the canonical grouping of the
        # nine signals): Profitability (4), Leverage/Liquidity & Source of Funds (3),
        # and Operating Efficiency (2).
        categories_keys = {
            "Profitability": ["positive_roa", "positive_ocf", "improving_roa", "accruals_quality"],
            "Leverage_Liquidity": ["lower_leverage", "higher_current_ratio", "no_dilution"],
            "Operating_Efficiency": ["higher_gross_margin", "higher_asset_turnover"],
        }
        categories = {
            name: f"{sum(1 for k in keys if signals[k] is True)}/{len(keys)}"
            for name, keys in categories_keys.items()
        }

        # Standing disclaimer: the F-Score is calibrated for value (high book-to-market)
        # firms and rewards improvement, so this guards against misreading a mid-range
        # score on a strong, stable large-cap as a bearish signal.
        note = (
            "Piotroski F-Score is calibrated for value (high book-to-market) firms and rewards "
            "year-over-year improvement; mature, already-efficient large-caps often score "
            "mid-range despite strong fundamentals. Read it with the Absolute_Health vs. "
            "Momentum split, not as a standalone verdict."
        )
        if status == "Neutral" and health == len(health_keys):
            interpretation = (
                "Sound but mature: strong absolute profitability with limited year-over-year "
                "improvement. A mid-range score here is typical for large, stable firms and is "
                "not inherently bearish."
            )
        elif status == "Strong":
            interpretation = "Fundamentally strong and improving across most signals."
        elif status == "Weak":
            interpretation = "Fundamentally weak; multiple deteriorating signals."
        else:
            interpretation = "Mixed fundamentals; review the individual signals before concluding."

        return {
            "F_Score": f_score,
            "Tests_Applied": tested,  # < 9 when a test was skipped for missing data
            "Status": status,
            "Categories": categories,  # Piotroski's original profitability/leverage/efficiency buckets
            "Absolute_Health": f"{health}/{len(health_keys)}",
            "Momentum": f"{momentum}/{len(momentum_keys)}",
            "Interpretation": interpretation,
            "Note": note,
            "Confidence": self._confidence(
                [current, prior], ["NetIncome", "Assets", "OperatingCashFlow", "Sales"]
            ),
            "Signals": signals,
        }

    def calculate_verdict(self, m_score=None, z_score=None, sloan=None,
                          piotroski=None, leverage=None, benford=None) -> dict:
        """
        Deterministically aggregates the individual model outputs into a single
        risk verdict: "Clean" / "Watch" / "High Risk" (with emoji), plus the exact
        reasons behind it. This is the classification the precision/recall backtest
        scores; it is intentionally rule-based (NOT LLM-decided) so it is reproducible,
        auditable, and defensible. The LLM later *explains* this verdict in prose; it
        never changes it.

        Signal tiers (a Low-confidence model output is downgraded one tier so a
        structurally-invalid reading -- e.g. Altman Z on a bank -- can't drive a verdict):
          * strong  -> High Risk on its own (high-confidence fraud/distress/insolvency)
          * moderate-> Watch (one) or High Risk (two or more)
          * minor   -> reported as a note only; never changes the verdict
        """
        def conf(d):
            return d.get("Confidence", "High") if d else "High"

        strong, moderate, minor, mitigants = [], [], [], []

        # --- Beneish M-Score (earnings manipulation) ---
        if m_score and m_score.get("M_Score") is not None:
            if m_score.get("Is_Manipulator"):
                reason = f"Beneish M-Score {m_score['M_Score']} > -2.22 (earnings manipulation likely)"
                (strong if conf(m_score) != "Low" else moderate).append(
                    reason + ("" if conf(m_score) != "Low" else " [low confidence]"))
            else:
                mitigants.append(f"Beneish M-Score {m_score['M_Score']} <= -2.22 (no manipulation signal)")

        # --- Sloan accruals (earnings quality) ---
        # High accruals are an earnings-QUALITY concern, not proof of fraud, and legitimately
        # spike for hypergrowth firms (working-capital build-up, e.g. NVDA). So it is a moderate
        # signal: a Watch on its own, escalating to High Risk only alongside a manipulation/
        # distress flag. The dedicated fraud/distress models stay standalone-RED triggers.
        if sloan and sloan.get("Sloan_Ratio") is not None:
            if str(sloan.get("Status", "")).startswith("High Risk"):
                reason = f"Sloan accruals {sloan['Sloan_Ratio']} > 0.10 (low earnings quality)"
                moderate.append(reason + ("" if conf(sloan) != "Low" else " [low confidence]"))

        # --- Altman Z-Score (distress) ---
        if z_score and z_score.get("Z_Score") is not None:
            status = z_score.get("Status")
            if status == "Distress":
                reason = f"Altman Z {z_score['Z_Score']} < 1.81 (financial distress)"
                (strong if conf(z_score) != "Low" else moderate).append(
                    reason + ("" if conf(z_score) != "Low" else " [low confidence]"))
            elif status == "Grey Zone":
                minor.append(f"Altman Z {z_score['Z_Score']} in grey zone (1.81-2.99, not 'safe')")
            elif status == "Safe":
                mitigants.append(f"Altman Z {z_score['Z_Score']} >= 2.99 (safe zone)")

        # --- Leverage (over-leverage / insolvency) ---
        if leverage:
            lstatus = leverage.get("Status", "")
            if lstatus.startswith("Negative Equity"):
                strong.append("Negative book equity (liabilities exceed assets; insolvent)")
            elif lstatus == "Extreme Leverage":
                reason = f"Equity multiplier {leverage.get('Equity_Multiplier')}x (extreme leverage)"
                (strong if conf(leverage) != "Low" else moderate).append(reason)
            elif lstatus.startswith("High Leverage"):
                moderate.append(f"Equity multiplier {leverage.get('Equity_Multiplier')}x (high leverage)")
            trend = leverage.get("Trend")
            if trend and trend.get("Direction") == "rising" and lstatus not in ("Extreme Leverage",) \
                    and not lstatus.startswith("Negative Equity"):
                moderate.append(f"Leverage rising YoY ({trend.get('Prior_Equity_Multiplier')}x -> "
                                f"{leverage.get('Equity_Multiplier')}x)")

        # --- Piotroski F-Score (fundamental strength) ---
        if piotroski and piotroski.get("F_Score") is not None:
            if piotroski.get("Status") == "Weak":
                moderate.append(f"Piotroski F-Score {piotroski['F_Score']}/9 (weak fundamentals)")
            elif piotroski.get("Status") == "Strong":
                mitigants.append(f"Piotroski F-Score {piotroski['F_Score']}/9 (strong fundamentals)")

        # --- Benford's Law (digit-distribution anomaly) ---
        # Single-filer, single-year Benford on ~1000 statement line items is too noisy to drive a
        # verdict -- it false-positives on clean blue-chips (e.g. MSFT, AAPL). It is therefore
        # advisory only: surfaced as a NOTE (and on its gauge) but never escalates the verdict.
        # Benford is designed for large transaction-level datasets, not a single filer's facts.
        if benford and benford.get("MAD") is not None:
            bstatus = benford.get("Status", "")
            if bstatus.startswith("Nonconformity"):
                minor.append(f"Benford MAD {benford['MAD']} > 0.015 (digit anomaly; advisory only -- "
                             f"single-filer Benford is unreliable, not used in the verdict)")
            elif bstatus == "Marginal":
                minor.append(f"Benford MAD {benford['MAD']} marginally nonconforming (advisory)")

        # --- Aggregate to a single verdict ---
        if not (strong or moderate or minor or mitigants):
            return {"Verdict": "Insufficient Data", "Emoji": "WHITE",
                    "Flags": [], "Mitigants": [], "Notes": [], "Confidence": "Low",
                    "Reasons": ["No model produced a usable result."]}

        if strong or len(moderate) >= 2:
            verdict, emoji = "High Risk", "RED"
        elif len(moderate) == 1:
            verdict, emoji = "Watch", "YELLOW"
        else:
            verdict, emoji = "Clean", "GREEN"

        # Verdict confidence is Low when it rests only on low-confidence/moderate signals.
        verdict_conf = "High" if strong else ("Low" if (moderate and not strong) and emoji == "RED"
                                              else "High" if emoji == "GREEN" else "Medium")

        return {
            "Verdict": verdict,
            "Emoji": emoji,  # RED / YELLOW / GREEN / WHITE -- UI maps to the dot
            "Flags": strong + moderate,
            "Mitigants": mitigants,
            "Notes": minor,
            "Confidence": verdict_conf,
            "Reasons": strong + moderate,  # exactly what the LLM memo verbalizes
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Smoke Test with textbook dummy data
    engine = ForensicEngine()
    
    # Fictitious Prior Year
    prior_year = {
        "Sales": 1000, "CostOfGoodsSold": 500, "GrossMargin": 500,
        "Receivables": 100, "CurrentAssets": 300, "PropertyPlantEquipment": 400,
        "Securities": 50, "Assets": 1000, "Depreciation": 50, "SGA": 150,
        "CurrentLiabilities": 200, "LongTermDebt": 300, "NetIncome": 100, "OperatingCashFlow": 120,
        "SharesOutstanding": 1000
    }

    # Fictitious Current Year (Showing signs of revenue manipulation / channel stuffing)
    current_year = {
        "Sales": 1500, "CostOfGoodsSold": 800, "GrossMargin": 700,
        "Receivables": 350,  # Massive jump in receivables (DSRI spikes)
        "CurrentAssets": 600, "PropertyPlantEquipment": 450,
        "Securities": 50, "Assets": 1500, "Depreciation": 60, "SGA": 200,
        "CurrentLiabilities": 300, "LongTermDebt": 400, "NetIncome": 250, "OperatingCashFlow": 50, # Accruals spike
        "WorkingCapital": 300, "TotalLiabilities": 700, "RetainedEarnings": 100, "EBIT": 280, "StockholdersEquity": 800,
        "SharesOutstanding": 1100  # Dilution
    }

    m_results = engine.calculate_m_score(current_year, prior_year)
    z_results = engine.calculate_z_score(current_year)
    sloan_results = engine.calculate_sloan_ratio(current_year, prior_year)
    piotroski_results = engine.calculate_piotroski_f_score(current_year, prior_year)
    leverage_results = engine.calculate_leverage(current_year, prior_year)
    
    print("\n--- Beneish M-Score Test ---")
    print(f"M-Score: {m_results['M_Score']}")
    print(f"Signals Manipulation ( > -2.22 )?: {m_results['Is_Manipulator']}")

    print("\n--- Altman Z-Score Test ---")
    print(f"Z-Score: {z_results['Z_Score']}")
    print(f"Distress Status: {z_results['Status']}")
    
    print("\n--- Sloan Accruals Ratio Test ---")
    print(f"Sloan Ratio: {sloan_results['Sloan_Ratio']}")
    print(f"Earnings Quality: {sloan_results['Status']}")

    print("\n--- Leverage Test ---")
    print(f"Equity Multiplier: {leverage_results['Equity_Multiplier']}x | Debt/Equity: {leverage_results['Debt_to_Equity']}")
    print(f"Status: {leverage_results['Status']} | Trend: {leverage_results['Trend']}")

    print("\n--- Piotroski F-Score Test ---")
    print(f"F-Score: {piotroski_results['F_Score']} / {piotroski_results['Tests_Applied']}")
    print(f"Fundamental Strength: {piotroski_results['Status']}")
    print(f"Categories (Piotroski): {piotroski_results['Categories']}")
    print(f"Absolute Health: {piotroski_results['Absolute_Health']} | Momentum: {piotroski_results['Momentum']}")
    print(f"Interpretation: {piotroski_results['Interpretation']}")

    # Create dummy SEC facts structure to test Benford's Law
    # A natural distribution should favor 1s and 2s. We'll inject an unnatural distribution.
    dummy_facts = {"facts": {"us-gaap": {}}}
    for i in range(100):
        # Unnatural starting digits: heavily favors 7, 8, 9
        val = float(f"{np.random.choice([7,8,9, 1, 2])}{np.random.randint(100, 999)}")
        dummy_facts["facts"]["us-gaap"][f"FakeTag{i}"] = {"units": {"USD": [{"end": "2023-12-31", "val": val}]}}
        
    benford_results = engine.calculate_benford_deviation(dummy_facts, 2023)
    print("\n--- Benford's Law Test ---")
    print(f"Mean Absolute Deviation (MAD): {benford_results['MAD']}")
    print(f"Conformity Status: {benford_results['Status']}")
    print(f"Sample Size: {benford_results['Sample_Size']}")

    verdict = engine.calculate_verdict(
        m_score=m_results, z_score=z_results, sloan=sloan_results,
        piotroski=piotroski_results, leverage=leverage_results,
    )
    print("\n--- Aggregated Verdict (deterministic) ---")
    print(f"[{verdict['Emoji']}] {verdict['Verdict']}  (confidence: {verdict['Confidence']})")
    for r in verdict["Reasons"]:
        print(f"  - {r}")