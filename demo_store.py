"""
Demo-mode persistence: save completed committee debates (transcript + bull-research brief) to disk
so they survive app restarts and can be replayed INSTANTLY during a live presentation.

A debate is keyed by the app's `debate_key` (company + fiscal year + verdict + research-toggle).
Files live under data/demo_cache/ (gitignored by default; remove that .gitignore line to make saved
demos portable across machines).
"""
import os
import json
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEMO_DIR = Path(os.getenv("DEMO_CACHE_PATH", "./data/demo_cache"))


def _path(key: str) -> Path:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return DEMO_DIR / f"{digest}.json"


def save_debate(key: str, transcript: list, brief: dict = None) -> None:
    """Persist a completed debate. `transcript` is a list of (speaker, text); `brief` is the
    bull-research dict ({brief, sources}) or None."""
    try:
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "transcript": transcript, "brief": brief}
        _path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved demo debate: {key}")
    except Exception as e:  # persistence is a convenience; never break the app over it
        logger.warning(f"Could not save demo debate ({key}): {e}")


def load_debate(key: str) -> dict:
    """Return {'transcript': [[speaker, text], ...], 'brief': {...}|None} or None if not saved."""
    path = _path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Could not load demo debate ({key}): {e}")
        return None
