import os
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Historical fraud / distress corpus for RAG analogies. Each entry describes the MECHANISM and
# the quantitative red-flag signature (so retrieval aligns with the engine's verdict reasons,
# e.g. "DSRI", "accruals", "leverage", "capitalized expenses"). These are factual descriptions of
# public cases used for analogy — no financial figures are fabricated here.
FRAUD_CORPUS = [
    {"id": "enron_2001", "metadata": {"company": "Enron", "year": "2001", "type": "off-balance-sheet"},
     "text": "Enron used special-purpose entities (SPEs) and mark-to-market accounting to hide debt "
             "off the balance sheet and book unrealized future profits as current earnings. Red-flag "
             "signature: surging revenue with weak operating cash flow, ballooning receivables and "
             "soft 'other' assets (high accruals, asset-quality and sales-growth indices), and rising "
             "leverage. Collapsed into bankruptcy in 2001."},
    {"id": "worldcom_2002", "metadata": {"company": "WorldCom", "year": "2002", "type": "expense capitalization"},
     "text": "WorldCom improperly capitalized ~$3.8B of recurring line-cost expenses as fixed assets, "
             "inflating profit and PP&E. Red-flag signature: it does NOT trip receivables/revenue models "
             "like Beneish (the fraud was expense-side), but property/plant grows abnormally and the firm "
             "shows distress on Altman Z. A reminder that expense-capitalization fraud needs distress and "
             "asset-growth checks, not just earnings-manipulation models."},
    {"id": "sunbeam_1997", "metadata": {"company": "Sunbeam", "year": "1997", "type": "channel stuffing"},
     "text": "Under Al Dunlap, Sunbeam used 'bill-and-hold' sales, channel stuffing and cookie-jar "
             "reserves to fabricate a turnaround. Red-flag signature: receivables rising far faster than "
             "sales (high days-sales-in-receivables / DSRI), positive net income with negative operating "
             "cash flow (high Sloan accruals and TATA), and slowing depreciation. The canonical Beneish "
             "M-Score case; restated in 1998, bankrupt in 2001."},
    {"id": "lehman_2008", "metadata": {"company": "Lehman Brothers", "year": "2008", "type": "leverage / distress"},
     "text": "Lehman used 'Repo 105' transactions to temporarily move ~$50B of assets off the balance "
             "sheet at quarter-end, masking extreme leverage (~30x assets-to-equity). Red-flag signature: "
             "leverage is the tell, not earnings manipulation; classic Beneish/Altman models do not apply "
             "cleanly to a broker-dealer. A liquidity run drove the September 2008 bankruptcy."},
    {"id": "waste_mgmt_1998", "metadata": {"company": "Waste Management", "year": "1998", "type": "depreciation"},
     "text": "Waste Management extended the useful lives and inflated salvage values of garbage trucks and "
             "equipment to understate depreciation and overstate profit by ~$1.7B. Red-flag signature: a "
             "falling depreciation rate versus a stable asset base (depreciation index), with earnings "
             "outpacing cash flow."},
    {"id": "healthsouth_2003", "metadata": {"company": "HealthSouth", "year": "2003", "type": "fabricated earnings"},
     "text": "HealthSouth fabricated ~$2.7B in earnings over years to hit analyst targets, booking fake "
             "revenue and nonexistent assets ('dummy' fixed assets). Red-flag signature: earnings far "
             "exceeding operating cash flow (high accruals), implausible asset-quality and margin trends."},
    {"id": "tyco_2002", "metadata": {"company": "Tyco", "year": "2002", "type": "looting / acquisition accounting"},
     "text": "Tyco executives looted the company via unauthorized bonuses and loans, and used aggressive "
             "acquisition ('spring-loading') accounting to manage earnings across hundreds of deals. "
             "Red-flag signature: acquisition-driven revenue growth, soft intangibles, and governance red "
             "flags rather than a single statement ratio."},
    {"id": "satyam_2009", "metadata": {"company": "Satyam", "year": "2009", "type": "fabricated cash"},
     "text": "India's Satyam fabricated ~$1B of cash and bank balances and fake invoices to inflate "
             "revenue and margins. Red-flag signature: fabricated CASH makes a firm look HEALTHIER, so "
             "earnings/distress models read clean — a balance-sheet-fraud blind spot best caught by "
             "auditing cash confirmations, not ratios."},
    {"id": "toshiba_2015", "metadata": {"company": "Toshiba", "year": "2015", "type": "premature revenue"},
     "text": "Toshiba overstated profit by ~$1.2B over seven years via aggressive percentage-of-completion "
             "estimates and deferred costs, driven by top-down profit pressure. Red-flag signature: margins "
             "and accruals drifting up while cash conversion lags."},
    {"id": "valeant_2015", "metadata": {"company": "Valeant", "year": "2015", "type": "channel / revenue recognition"},
     "text": "Valeant used the captive specialty pharmacy Philidor to recognize revenue on products not yet "
             "sold through to patients, alongside acquisition-fueled growth and heavy debt. Red-flag "
             "signature: receivables and channel inventory outrunning real demand (DSRI, sales-growth) plus "
             "high leverage."},
    {"id": "underarmour_2017", "metadata": {"company": "Under Armour", "year": "2017", "type": "revenue timing"},
     "text": "Under Armour pulled forward ~$400M of future sales into current quarters to meet growth "
             "targets (SEC settled 2021). Red-flag signature: receivables and sales-growth indices elevated "
             "relative to cash collection — a milder, timing-based revenue manipulation."},
    {"id": "kraftheinz_2019", "metadata": {"company": "Kraft Heinz", "year": "2019", "type": "procurement accounting"},
     "text": "Kraft Heinz misstated cost of goods sold via improper supplier-rebate/procurement accounting "
             "and took a ~$15B impairment. Red-flag signature: margin and accrual anomalies plus a large "
             "goodwill writedown signaling prior overstatement."},
    {"id": "nikola_2020", "metadata": {"company": "Nikola", "year": "2020", "type": "promotional / disclosure fraud"},
     "text": "Nikola's founder made false claims about working trucks and technology to inflate the stock; "
             "the company was essentially pre-revenue. Red-flag signature: NONE in the financial statements "
             "— quantitative statement models cannot catch promotional/disclosure fraud at a company with "
             "no revenue to manipulate. A limitation case."},
]


class FraudCorpusRAG:
    """
    Local vector database (ChromaDB) of historical fraud/distress cases for RAG analogies.
    Auto-seeds FRAUD_CORPUS on first use so the memo / debate / interrogation can ground
    analogies in real cases.
    """
    def __init__(self):
        self.chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB PersistentClient at {self.chroma_path}")
        self.client = chromadb.PersistentClient(path=self.chroma_path)

        # 'all-MiniLM-L6-v2' — fast, lightweight, CPU-friendly (per the proposal).
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="historical_frauds", embedding_function=self.embedding_fn
        )

        # Populate (or top up) the curated corpus. upsert is idempotent.
        if self.collection.count() < len(FRAUD_CORPUS):
            self.seed_corpus()

    def seed_corpus(self):
        """Upserts the curated fraud-case corpus (idempotent)."""
        self.collection.upsert(
            ids=[c["id"] for c in FRAUD_CORPUS],
            documents=[c["text"] for c in FRAUD_CORPUS],
            metadatas=[c["metadata"] for c in FRAUD_CORPUS],
        )
        logger.info(f"Seeded {len(FRAUD_CORPUS)} fraud cases into ChromaDB.")

    def add_case(self, case_id: str, text: str, metadata: dict = None):
        """Adds/updates a single historical fraud case."""
        self.collection.upsert(documents=[text], metadatas=[metadata or {}], ids=[case_id])
        logger.info(f"Added case {case_id} to ChromaDB.")

    def query_cases(self, query_text: str, n_results: int = 2) -> dict:
        """Returns the most similar historical fraud cases to the query."""
        logger.info(f"Querying ChromaDB for: '{query_text}'")
        return self.collection.query(query_texts=[query_text], n_results=n_results)


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    rag = FraudCorpusRAG()
    print(f"\nCorpus size: {rag.collection.count()} cases")

    for query in ["receivables rising faster than sales, channel stuffing",
                  "extreme leverage off-balance-sheet, broker-dealer distress",
                  "fabricated cash balances make the company look healthy"]:
        print(f"\n--- Query: {query!r} ---")
        results = rag.query_cases(query, n_results=2)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"  [{meta.get('company')}] {doc[:110]}...")
