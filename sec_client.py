import os
import json
import time
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class SECClient:
    """
    Handles interactions with the SEC EDGAR API with local file caching.

    SEC EDGAR fair-access policy allows up to ~10 requests/second per IP and
    blocks abusers (HTTP 403/429). This client stays well under that limit via a
    request throttle, reuses a single connection (requests.Session), caches the
    large company-ticker map to disk, and retries transient errors with backoff.
    """
    BASE_URL_DATA = "https://data.sec.gov/api/xbrl/companyfacts/"
    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    # Refresh the cached ticker map at most once a week (it changes rarely).
    TICKER_MAP_TTL_SECONDS = 7 * 24 * 60 * 60

    def __init__(self):
        self.user_agent = os.getenv("SEC_USER_AGENT")
        self.cache_path = Path(os.getenv("SEC_CACHE_PATH", "./data/sec_cache"))
        self.cache_path.mkdir(parents=True, exist_ok=True)

        if not self.user_agent or "your.email" in self.user_agent:
            logger.warning("SEC_USER_AGENT is not properly configured in .env")

        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate"
        }

        # Single connection reused across all requests (avoids TCP/TLS re-handshake
        # on every fetch in a backtest loop) with the required headers set once.
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Throttle: enforce a minimum gap between network calls. Default 0.15s
        # caps us at <7 req/s, comfortably under SEC's 10/s limit.
        self._min_interval = float(os.getenv("SEC_MIN_REQUEST_INTERVAL", "0.15"))
        self._last_request_ts = 0.0

    def _throttle(self):
        """Sleep just long enough to respect the minimum inter-request interval."""
        elapsed = time.monotonic() - self._last_request_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _request_json(self, url: str, max_retries: int = 3) -> dict:
        """
        Throttled GET that returns parsed JSON, retrying transient errors (429/503)
        with exponential backoff. Honors the Retry-After header when present.
        Returns None on 403 (a User-Agent problem, not worth retrying) or exhaustion.
        """
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if status == 403:
                    logger.error("SEC blocked request (403). Check your SEC_USER_AGENT in .env.")
                    return None
                if status in (429, 503) and attempt < max_retries:
                    retry_after = e.response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                    logger.warning(
                        f"SEC returned {status} (attempt {attempt}/{max_retries}); "
                        f"backing off {delay:.1f}s before retry."
                    )
                    time.sleep(delay)
                    backoff *= 2
                    continue
                logger.error(f"HTTP Error fetching {url}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None
        logger.error(f"Exhausted retries fetching {url}.")
        return None

    def _load_ticker_map(self) -> dict:
        """
        Returns the SEC ticker->CIK map, served from a local disk cache when it is
        present and younger than TICKER_MAP_TTL_SECONDS. This avoids re-downloading
        the ~1 MB map on every uncached company lookup (the biggest rate-limit and
        latency win for batch runs).
        """
        map_file = self.cache_path / "company_tickers.json"

        if map_file.exists():
            age = time.time() - map_file.stat().st_mtime
            if age < self.TICKER_MAP_TTL_SECONDS:
                logger.info("Loading SEC ticker map from cache.")
                with open(map_file, 'r') as f:
                    return json.load(f)

        logger.info("Fetching SEC ticker map...")
        data = self._request_json(self.TICKER_MAP_URL)
        if data is not None:
            with open(map_file, 'w') as f:
                json.dump(data, f)
        return data

    def _get_cik(self, ticker: str) -> str:
        """
        Maps a ticker symbol to a 10-digit zero-padded CIK.
        """
        ticker = ticker.upper().strip()
        data = self._load_ticker_map()
        if not data:
            return None

        for entry in data.values():
            if entry['ticker'] == ticker:
                # SEC CIKs must be 10 digits
                return str(entry['cik_str']).zfill(10)

        logger.error(f"Ticker {ticker} not found in SEC database.")
        return None

    def get_company_facts(self, ticker: str, force_refresh: bool = False) -> dict:
        """
        Fetches company facts from SEC or local cache.
        """
        ticker = ticker.upper().strip()
        cache_file = self.cache_path / f"{ticker}_facts.json"

        if not force_refresh and cache_file.exists():
            logger.info(f"Loading {ticker} facts from cache.")
            with open(cache_file, 'r') as f:
                return json.load(f)

        cik = self._get_cik(ticker)
        if not cik:
            return None

        url = f"{self.BASE_URL_DATA}CIK{cik}.json"
        logger.info(f"Fetching facts for {ticker} (CIK {cik}) from SEC...")
        data = self._request_json(url)
        if data is None:
            return None

        with open(cache_file, 'w') as f:
            json.dump(data, f)

        return data

if __name__ == "__main__":
    # Quick smoke test
    client = SECClient()
    test_ticker = "MSFT"
    facts = client.get_company_facts(test_ticker)
    if facts:
        print(f"Successfully retrieved facts for {test_ticker}")
        print(f"Company Name: {facts.get('entityName')}")
    else:
        print(f"Failed to retrieve facts for {test_ticker}")