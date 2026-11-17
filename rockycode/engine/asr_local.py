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

ARCH BRIDGE: mlx ships arm64 wheels only. A machine whose rocky runs on
an x86_64 anaconda python (the classic /usr/local/anaconda3 on Apple
Silicon) can never import it in-process — transcribe() then bridges to a
dedicated arm64 venv as a SUBPROCESS (`python -m rockycode.asr_local`,
same repo code via PYTHONPATH, WAV on stdin, text on stdout). The venv
path is ROCKYCODE_VENV_PYTHON or ~/rockycode-venv/bin/python; create it
with the CLT python: `/usr/bin/python3 -m venv ~/rockycode-venv &&
~/rockycode-venv/bin/pip install mlx-whisper`. An arm64 interpreter
with mlx_whisper importable runs in-process as before.
"""
from __future__ import annotations

import array
import importlib.util
import os
import subprocess
import sys
import wave
from io import BytesIO
from pathlib import Path
from typing import Optional

# The mlx-community repos carry the converted weights. `base` is the
# sweet spot for Mandarin commands — tiny is faster but sloppier.
DEFAULT_MODEL = "mlx-community/whisper-base-mlx"
VENV_ENV = "ROCKYCODE_VENV_PYTHON"
_REPO_ROOT = Path(__file__).resolve().parents[2]  # engine/ → rockycode/ → repo

class LocalAsrError(Exception):
    """Local ASR failed. Message is user-facing (a fix)."""


def available() -> Optional[str]:
    """None = mlx_whisper imports in THIS interpreter; else the fix — the
    in-process install, or (on an x86_64 python) the arm64 venv bridge."""
    if importlib.util.find_spec("mlx_whisper") is not None:
        return None
    if _venv_python() is not None:
        return None  # the bridge covers it
    return ("local ASR needs mlx-whisper — pip install mlx-whisper (Apple Silicon), or "
            "bridge to an arm64 venv: /usr/bin/python3 -m venv ~/rockycode-venv && "
            "~/rockycode-venv/bin/pip install mlx-whisper (model downloads on first use)")


def _venv_python() -> Optional[str]:
    pinned = (os.getenv(VENV_ENV) or "").strip()
    if pinned and os.path.exists(pinned):
        return pinned
    default = Path.home() / "rockycode-venv" / "bin" / "python"
    if default.exists():
        return str(default)
    return None


def _apple_silicon() -> bool:
    """HARDWARE check — an x86_64 python under Rosetta still runs on arm64
    hardware, and that's exactly the case that needs the arch pin:
    os.uname().machine reports the TRANSLATED arch ("x86_64"), which
    fooled the first version into skipping the pin (found live)."""
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"],
                             capture_output=True, timeout=5.0).stdout.strip()
        return out == b"1"
    except (OSError, subprocess.TimeoutExpired):
        return False


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


def transcribe(wav: bytes, *, model: str = DEFAULT_MODEL) -> tuple[str, float]:
    """WAV bytes → (text, duration_ms), transcribed LOCALLY — in-process
    when mlx_whisper imports here, else bridged to the arm64 venv (an
    x86_64 anaconda python can never import mlx). Blocking — the app
    calls it via asyncio.to_thread (whisper holds the GPU for the clip's
    length; the message pump must never feel it).

    mlx-whisper ≥0.4 loads the model INSIDE transcribe (path_or_hf_repo)
    and dropped the language/fp16 kwargs — the call stays on the common
    intersection so both old and new versions work."""
    if importlib.util.find_spec("mlx_whisper") is None:
        venv = _venv_python()
        if venv is None or os.environ.get("ROCKYCODE_ASR_BRIDGED"):
            # the far side of the bridge itself has no mlx either — raising
            # beats bridging to the same broken venv forever
            raise LocalAsrError(available() or "local ASR is not set up")
        return _transcribe_bridged(venv, wav, model)
    import numpy as np
    import mlx_whisper
    rate, samples = wav_to_samples(wav)
    dur_ms = len(samples) / max(rate, 1) * 1000.0
    if dur_ms < 200 or max(abs(s) for s in samples or [0.0]) < 1e-4:
        raise LocalAsrError("the clip is silent — the recorder's VAD should have kept it out")
    try:
        result = mlx_whisper.transcribe(
            np.array(samples, dtype=np.float32),
            path_or_hf_repo=model,
            temperature=0.0,        # deterministic — commands, not poetry
        )
    except Exception as e:  # noqa: BLE001 — a model/download failure is user-facing
        raise LocalAsrError(f"local ASR failed: {type(e).__name__}: {e}") from e
    text = str(result.get("text", "")).strip()
    if not text:
        raise LocalAsrError("the clip transcribed empty — speak a little louder")
    return text, dur_ms


def _transcribe_bridged(venv: str, wav: bytes, model: str) -> tuple[str, float]:
    """The x86_64-anaconda path: run THIS FILE as a script under the arm64
    venv (WAV on stdin, `text\tduration_ms` on stdout). Script mode on
    purpose — `-m rockycode.asr_local` would import the whole rockycode
    package and drag openai/textual into the venv; this module is pure
    stdlib + mlx_whisper, so the script is self-contained."""
    script = str(_REPO_ROOT / "rockycode" / "engine" / "asr_local.py")
    # ARCH PIN: the caller may be an x86_64 python under Rosetta — a child
    # it spawns that is a universal2 binary (the CLT python backing most
    # venvs) inherits the translated slice and runs x86_64, where the arm64
    # mlx/numpy wheels refuse to load ("incompatible architecture", found
    # live). `arch -arm64` forces the native slice on Apple Silicon.
    argv = [venv, script, "--model", model]
    if _apple_silicon():
        argv = ["arch", "-arm64", *argv]
    try:
        proc = subprocess.run(
            argv,
            input=wav, capture_output=True,
            env={**os.environ, "ROCKYCODE_ASR_BRIDGED": "1"},
            timeout=180.0,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise LocalAsrError(f"the local ASR bridge failed to run: {type(e).__name__}: {e}") from e
    if proc.returncode != 0:
        tail = (proc.stderr.decode(errors="replace").strip().splitlines() or ["no error output"])[-1]
        raise LocalAsrError(f"the local ASR bridge failed: {tail}")
    try:
        text, dur = proc.stdout.decode().strip().rsplit("\t", 1)
        return text, float(dur)
    except ValueError as e:
        raise LocalAsrError("the local ASR bridge sent an unreadable reply") from e


if __name__ == "__main__":
    # The bridge's far side: read the WAV from stdin, print text\tduration.
    # Errors go to stderr with exit 1 — the caller turns them into lines.
    try:
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else DEFAULT_MODEL
        wav = sys.stdin.buffer.read()
        text, dur_ms = transcribe(wav, model=model)
        print(f"{text}\t{dur_ms:.0f}")
    except Exception as e:  # noqa: BLE001 — the bridge must never traceback into the UI
        print(f"local ASR failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
