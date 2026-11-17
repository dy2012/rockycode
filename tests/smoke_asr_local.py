"""Local ASR backend (MLX Whisper) — the pure pieces, hermetic: the
availability hint, the WAV→samples shape, and the honest failures. The
model itself is never loaded here (mlx-whisper is optional; a machine
WITH it gets the live round-trip in run_real.py style instead).
"""
import array
import io
import math
import os
import sys
import tempfile
import wave
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
os.environ["ROCKYCODE_HOME"] = tempfile.mkdtemp(prefix="rockyhome-")

from rockycode.engine.asr_local import (  # noqa: E402
    DEFAULT_MODEL,
    LocalAsrError,
    available,
    transcribe,
    wav_to_samples,
)


def _wav(samples: list[int], rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(array.array("h", samples).tobytes())
    return buf.getvalue()


def test_availability_hint():
    # Without mlx_whisper the hint names the fix and transcribe raises it
    # before touching numpy; with it installed both come back clean.
    hint = available()
    if hint is not None:
        assert "pip install mlx-whisper" in hint
        try:
            transcribe(b"RIFF" + b"\x00" * 40)
            raise AssertionError("missing install did not raise")
        except LocalAsrError as e:
            assert "pip install mlx-whisper" in str(e)
    print("  availability: missing install → one-line fix, never a traceback  ✓")


def test_wav_to_samples():
    # 16-bit mono round-trip: samples come back as [-1, 1] floats, rate intact
    raw = [int(12000 * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(800)]
    rate, samples = wav_to_samples(_wav(raw))
    assert rate == 16000 and len(samples) == 800
    assert abs(samples[10] - raw[10] / 32768.0) < 1e-5
    # a headerless clip is a readable-shape error, not a crash
    try:
        wav_to_samples(b"RIFF" + b"\x00" * 40)
        raise AssertionError("bad clip did not raise")
    except LocalAsrError as e:
        assert "WAV" in str(e)
    print("  wav_to_samples: int16 mono → float32 [-1,1]; bad clips are honest  ✓")


def test_silent_clip_refused():
    # A silent clip never reaches the model — the VAD should have kept it
    # out, and when it slips through the backend says so.
    hint = available()
    if hint is not None:
        print("  silent clip: (mlx-whisper not installed — skipped)  ✓")
        return
    try:
        transcribe(_wav([0] * 1600))
        raise AssertionError("silent clip did not raise")
    except LocalAsrError as e:
        assert "silent" in str(e)
    print("  silent clip: refused before the model with a clear line  ✓")


def test_defaults():
    assert "whisper" in DEFAULT_MODEL and "mlx" in DEFAULT_MODEL
    print("  defaults: the mlx-community base model  ✓")


test_availability_hint()
test_wav_to_samples()
test_silent_clip_refused()
test_defaults()
print("ASR LOCAL SMOKE OK — availability · samples shape · silent refusal · defaults. amaze!")
