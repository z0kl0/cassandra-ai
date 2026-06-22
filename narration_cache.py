"""
Per-turn audio cache for committee-debate narration.

Synthesizing a debate's turns with Piper takes a few seconds; caching the WAV clips to disk (keyed by
the app's `debate_key`) makes stage replays instant. Mirrors the shape of demo_store.py. Files live
under data/audio_cache/<hash>/turn_NN.wav (gitignored).
"""
import os
import hashlib
import logging
from pathlib import Path

import voice

logger = logging.getLogger(__name__)

AUDIO_DIR = Path(os.getenv("AUDIO_CACHE_PATH", "./data/audio_cache"))


def _dir(key: str) -> Path:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return AUDIO_DIR / digest


def get_or_build(debate_key: str, transcript: list) -> list:
    """Return per-turn WAV bytes for `transcript` (list of (speaker, text)), in order.
    Synthesizes any missing turn via voice.synth_turn_wav and caches it to disk."""
    folder = _dir(debate_key)
    folder.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, (speaker, text) in enumerate(transcript):
        path = folder / f"turn_{i:02d}.wav"
        if path.exists():
            clips.append(path.read_bytes())
            continue
        wav = voice.synth_turn_wav(text, speaker)
        path.write_bytes(wav)
        clips.append(wav)
    return clips


def clear(debate_key: str) -> None:
    """Delete cached clips for a debate (used on 'Re-run fresh')."""
    folder = _dir(debate_key)
    if folder.exists():
        for f in folder.glob("*.wav"):
            try:
                f.unlink()
            except OSError as e:
                logger.warning(f"Could not remove cached clip {f}: {e}")
