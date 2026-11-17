"""Local streaming-quality ASR — MLX Whisper on Apple Silicon (plan C of
the voice roadmap).

Same contract as asr.py (the DeepSeek backend): WAV bytes in, (text,
duration_ms) out — the app just dispatches on config `voice_asr
deepseek|local`. Swapping backends costs one code path and nothing else:
the VAD, the conversation state machine, and the readback are untouched.

Why local: the DeepSeek ASR is whole-file over the network (1-3 s per
utterance); MLX Whisper on an M-series Mac transcribes the same clip in
~0.3-0.8 s, free, offline, and the audio never leaves the machine. The
model downloads once on first use (~150 MB for base; whisper-tiny is
~75 MB — config voice_asr_model picks).

mlx_whisper is a HEAVY optional import (pulls mlx + numpy) — nothing in
this module touches it until transcribe() runs, and a missing install
returns a one-line fix instead of a traceback.
"""
from __future__ import annotations

import array
import importlib.util
import wave
from io import BytesIO
from typing import Optional

# The mlx-community repos carry the converted weights. `base` is the
# sweet spot for Mandarin commands — tiny is faster but sloppier.
DEFAULT_MODEL = "mlx-community/whisper-base-mlx"

_MODEL: Optional[object] = None
_MODEL_NAME: str = ""


class LocalAsrError(Exception):
    """Local ASR failed. Message is user-facing (a fix)."""


def available() -> Optional[str]:
    """None = mlx_whisper is installed; else the one-line install fix."""
    if importlib.util.find_spec("mlx_whisper") is None:
        return ("local ASR needs mlx-whisper — install it: "
                "pip install mlx-whisper (Apple Silicon; ~150 MB model downloads on first use)")
    return None


def wav_to_samples(wav: bytes) -> tuple[int, list[float]]:
    """(sample rate, float32-mono samples in [-1, 1]) from a WAV clip —
    pure stdlib so the shape is smoke-testable without mlx installed."""
    try:
        with wave.open(BytesIO(wav), "rb") as w:
            rate = w.getframerate() or 16_000
            n = w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise LocalAsrError(f"the clip is not a readable WAV: {e}") from e
    samples = array.array("h")
    samples.frombytes(raw[: n * 2])
    if w.getsampwidth() != 2 or w.getnchannels() != 1:
        # a stereo/8-bit clip: trust the rate, feed whatever we got (the
        # recorder always sends 16-bit mono — this is defensive)
        return rate, [s / 32768.0 for s in samples[:n]]
    return rate, [s / 32768.0 for s in samples]


def _model(name: str):
    """Load (and cache) the MLX model — the first call downloads it."""
    global _MODEL, _MODEL_NAME
    import mlx_whisper  # noqa: F401 — availability was checked by the caller
    if _MODEL is None or _MODEL_NAME != name:
        # lazy import INSIDE — mlx is heavy and this module must stay cheap
        import mlx_whisper
        _MODEL = mlx_whisper.load_model(name)
        _MODEL_NAME = name
    return _MODEL


def transcribe(wav: bytes, *, model: str = DEFAULT_MODEL) -> tuple[str, float]:
    """WAV bytes → (text, duration_ms), transcribed LOCALLY. Blocking —
    the app calls it via asyncio.to_thread (whisper holds the CPU/GPE for
    the clip's length; the message pump must never feel it)."""
    hint = available()
    if hint:
        raise LocalAsrError(hint)
    import numpy as np
    rate, samples = wav_to_samples(wav)
    dur_ms = len(samples) / max(rate, 1) * 1000.0
    if dur_ms < 200 or max(abs(s) for s in samples or [0.0]) < 1e-4:
        raise LocalAsrError("the clip is silent — the recorder's VAD should have kept it out")
    try:
        result = _model(model).transcribe(
            np.array(samples, dtype=np.float32),
            language=None,          # auto-detect (zh/yue/en all work)
            fp16=True,              # MLX native — the speed
            initial_prompt=None,
            temperature=0.0,        # deterministic — commands, not poetry
        )
    except Exception as e:  # noqa: BLE001 — a model/download failure is user-facing
        raise LocalAsrError(f"local ASR failed: {type(e).__name__}: {e}") from e
    text = str(result.get("text", "")).strip()
    if not text:
        raise LocalAsrError("the clip transcribed empty — speak a little louder")
    return text, dur_ms
