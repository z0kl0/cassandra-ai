"""Offline tests for the voice/narration helpers (no Piper model or audio device needed)."""
import voice


def test_clean_for_tts_strips_markdown_and_emoji():
    out = voice._clean_for_tts("**Bold** _italic_ `code` 🔻 # Heading\n\nNext line")
    for ch in ("*", "_", "`", "#", "🔻"):
        assert ch not in out
    assert "Bold" in out and "italic" in out and "Next line" in out


def test_debate_voices_ready_returns_bool():
    # No model files in CI -> False, but must always be a clean bool (it gates the UI toggle).
    assert isinstance(voice.debate_voices_ready(), bool)


def test_debate_voices_cover_the_three_personas():
    assert set(voice.DEBATE_VOICES) == {"Cassandra (Skeptic)", "Michael (Bull)", "CIO (Ruling)"}
    # Single-speaker personas carry no speaker id; Cassandra (libritts_r) carries an int id.
    assert voice.DEBATE_VOICES["Michael (Bull)"][1] is None
    assert isinstance(voice.DEBATE_VOICES["Cassandra (Skeptic)"][1], int)
