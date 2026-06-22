"""
Optional voice layer for CASSANDRA's "Interrogation Mode" — gated behind ENABLE_VOICE.

Speech-to-text via faster-whisper (local, CPU) and text-to-speech via Piper (local) with an
Edge-TTS online fallback. Heavy deps (faster-whisper, piper-tts) are imported LAZILY inside the
functions so the core app runs fine when voice is disabled / deps aren't installed. STT/TTS acquire
the shared LLM lock so audio work never overlaps model generation (one task at a time on CPU).

Install:  pip install -r requirements-voice.txt   (Edge fallback also needs: pip install edge-tts)
Piper TTS needs a downloaded voice model — set PIPER_VOICE in .env to its .onnx path,
or set TTS_ENGINE=edge to use the online fallback.
"""
import io
import os
import re
import wave
import logging

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ENABLE_VOICE = os.getenv("ENABLE_VOICE", "false").lower() == "true"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")        # tiny | base | small (CPU)
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper").lower()     # piper (local) | edge (online)
PIPER_VOICE = os.getenv("PIPER_VOICE", "")                # path to a Piper .onnx voice model

# Per-persona voices for committee-debate narration. Each entry is (onnx_path, speaker_id); speaker_id
# is only meaningful for multi-speaker models (libritts_r), and is None for single-speaker (joe/norman).
DEBATE_VOICES = {
    "Cassandra (Skeptic)": (os.getenv("PIPER_VOICE_CASSANDRA", "voices/en_US-libritts_r-medium.onnx"),
                            int(os.getenv("PIPER_SPEAKER_CASSANDRA", "0"))),
    "Michael (Bull)": (os.getenv("PIPER_VOICE_MICHAEL", "voices/en_US-joe-medium.onnx"), None),
    "CIO (Ruling)": (os.getenv("PIPER_VOICE_CIO", "voices/en_US-norman-medium.onnx"), None),
}

_whisper_model = None
_piper_cache = {}   # onnx_path -> loaded PiperVoice (one per file, reused across turns)


def voice_enabled() -> bool:
    return ENABLE_VOICE


def _get_whisper():
    """Lazily load + cache the faster-whisper model (int8 on CPU)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel  # lazy: optional dependency
        logger.info(f"Loading faster-whisper model '{WHISPER_MODEL}' (CPU/int8)...")
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe recorded audio (e.g. st.audio_input bytes) to text."""
    from llm import _LLM_LOCK
    model = _get_whisper()
    with _LLM_LOCK:  # never transcribe while the LLM is generating
        segments, _ = model.transcribe(io.BytesIO(audio_bytes))
        return " ".join(seg.text for seg in segments).strip()


def synthesize(text: str) -> bytes:
    """Synthesize speech (WAV/MP3 bytes) for `text` via Piper (local) or Edge (online)."""
    from llm import _LLM_LOCK
    with _LLM_LOCK:
        if TTS_ENGINE == "edge":
            return _edge_tts(text)
        return _piper_tts(text)


def _piper_tts(text: str) -> bytes:
    import wave
    from piper import PiperVoice  # lazy: optional dependency

    if not PIPER_VOICE or not os.path.exists(PIPER_VOICE):
        raise RuntimeError(
            "Piper TTS needs a voice model: set PIPER_VOICE in .env to a downloaded .onnx "
            "voice file, or set TTS_ENGINE=edge for the online fallback."
        )
    voice = PiperVoice.load(PIPER_VOICE)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize(text, wav_file)
    return buf.getvalue()


# ---- Multi-voice debate narration ------------------------------------------------------

# Strip markdown / emoji / stray symbols so Piper reads the words, not the punctuation.
_MD_RE = re.compile(r"[*_`#>]+")
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")


def _clean_for_tts(text: str) -> str:
    text = _MD_RE.sub("", text or "")
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def debate_voices_ready() -> bool:
    """True only if every configured debate voice .onnx file exists on disk (gates the UI)."""
    return all(os.path.exists(path) for path, _ in DEBATE_VOICES.values())


def _load_piper(onnx_path: str):
    """Lazily load + cache one PiperVoice per .onnx file (reused across all turns)."""
    if onnx_path not in _piper_cache:
        from piper import PiperVoice  # lazy: optional dependency
        logger.info(f"Loading Piper voice: {onnx_path}")
        _piper_cache[onnx_path] = PiperVoice.load(onnx_path)
    return _piper_cache[onnx_path]


def synth_turn_wav(text: str, speaker_label: str) -> bytes:
    """Synthesize one debate turn to WAV bytes using that persona's Piper voice.
    Held under the shared LLM lock (narration runs after generation, so the model is idle)."""
    from llm import _LLM_LOCK
    from piper.config import SynthesisConfig

    onnx_path, speaker_id = DEBATE_VOICES[speaker_label]
    voice = _load_piper(onnx_path)
    cfg = SynthesisConfig(speaker_id=speaker_id) if speaker_id is not None else None
    buf = io.BytesIO()
    with _LLM_LOCK:
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(_clean_for_tts(text), wav_file, syn_config=cfg)
    return buf.getvalue()


def _edge_tts(text: str) -> bytes:
    import asyncio
    import edge_tts  # lazy: optional dependency (pip install edge-tts)

    async def _run() -> bytes:
        out = io.BytesIO()
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                out.write(chunk["data"])
        return out.getvalue()

    return asyncio.run(_run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"ENABLE_VOICE={ENABLE_VOICE} | WHISPER_MODEL={WHISPER_MODEL} | TTS_ENGINE={TTS_ENGINE}")
    print("Install voice deps (requirements-voice.txt) and set ENABLE_VOICE=true to use the layer.")
