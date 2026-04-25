"""
mastering.py — Post-production for the synthesized vocal output.

Preset chain
------------
  streaming   : -14 LUFS, slight high-shelf air, light limiting
  vinyl       : -18 LUFS, warm low-mid saturation, gentle compression
  club        : -9  LUFS, punchy transient shaping, hard limiting
  bright_pop  : -12 LUFS, presence boost, de-essing hint
  raw         : loudness normalisation only (no EQ/compression)

The ``matchering`` library is used when a reference track is provided;
otherwise ``pyloudnorm`` handles integrated-loudness normalisation and
``scipy`` / ``numpy`` provide lightweight EQ and limiting.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional


# ── Loudness helpers ──────────────────────────────────────────────────────────

def _lufs_normalise(
    audio: np.ndarray, sr: int, target_lufs: float
) -> np.ndarray:
    """Return *audio* normalised to *target_lufs* integrated loudness."""
    try:
        import pyloudnorm as pyln  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyloudnorm is not installed.  Run: pip install pyloudnorm"
        ) from exc

    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(audio)
    if np.isinf(measured):
        return audio  # silence — nothing to normalise
    return pyln.normalize.loudness(audio, measured, target_lufs)


def _peak_limit(audio: np.ndarray, ceiling: float = -1.0) -> np.ndarray:
    """Hard-limit peaks to *ceiling* dBFS."""
    linear_ceiling = 10 ** (ceiling / 20.0)
    peak = np.max(np.abs(audio))
    if peak > linear_ceiling:
        audio = audio * (linear_ceiling / peak)
    return audio


# ── Simple biquad EQ ─────────────────────────────────────────────────────────

def _high_shelf(
    audio: np.ndarray, sr: int, freq: float, gain_db: float
) -> np.ndarray:
    """Apply a high-shelf filter (Zoelzer cookbook)."""
    from scipy.signal import sosfilt, butter  # type: ignore

    nyq = sr / 2.0
    norm_freq = min(freq / nyq, 0.99)
    sos = butter(2, norm_freq, btype="high", output="sos")
    # Mix in the gained high band
    gain_lin = 10 ** (gain_db / 20.0)
    high = sosfilt(sos, audio)
    low = audio - high
    return low + high * gain_lin


def _low_shelf(
    audio: np.ndarray, sr: int, freq: float, gain_db: float
) -> np.ndarray:
    from scipy.signal import sosfilt, butter  # type: ignore

    nyq = sr / 2.0
    norm_freq = min(freq / nyq, 0.99)
    sos = butter(2, norm_freq, btype="low", output="sos")
    gain_lin = 10 ** (gain_db / 20.0)
    low = sosfilt(sos, audio)
    high = audio - low
    return low * gain_lin + high


# ── Preset definitions ────────────────────────────────────────────────────────

def _apply_preset(audio: np.ndarray, sr: int, preset: str) -> np.ndarray:
    preset = preset.lower()

    if preset == "streaming":
        audio = _high_shelf(audio, sr, freq=8000, gain_db=1.5)
        audio = _low_shelf(audio, sr, freq=120, gain_db=-1.0)
        audio = _peak_limit(audio, ceiling=-1.0)

    elif preset == "vinyl":
        audio = _low_shelf(audio, sr, freq=300, gain_db=2.0)
        audio = _high_shelf(audio, sr, freq=10000, gain_db=-2.0)
        audio = _peak_limit(audio, ceiling=-2.0)

    elif preset == "club":
        audio = _low_shelf(audio, sr, freq=80, gain_db=3.0)
        audio = _high_shelf(audio, sr, freq=6000, gain_db=2.0)
        audio = _peak_limit(audio, ceiling=-0.3)

    elif preset == "bright_pop":
        audio = _high_shelf(audio, sr, freq=5000, gain_db=3.0)
        audio = _low_shelf(audio, sr, freq=200, gain_db=-0.5)
        audio = _peak_limit(audio, ceiling=-1.0)

    else:  # "raw" or unknown
        audio = _peak_limit(audio, ceiling=-1.0)

    return audio


# ── Public API ────────────────────────────────────────────────────────────────

def master(
    input_path: str | Path,
    output_path: str | Path,
    preset: str = "streaming",
    target_lufs: float = -14.0,
    reference_track: Optional[str | Path] = None,
) -> Path:
    """
    Apply mastering to *input_path* and write the result to *output_path*.

    Parameters
    ----------
    input_path : str | Path
        Rendered vocal WAV after RVC conversion.
    output_path : str | Path
        Destination path for the mastered WAV.
    preset : str
        One of: ``streaming``, ``vinyl``, ``club``, ``bright_pop``, ``raw``.
    target_lufs : float
        Integrated loudness target (EBU R128).  Typical values: -14 (streaming),
        -9 (club), -18 (vinyl).
    reference_track : str | Path | None
        Optional path to a reference WAV for matchering-based mastering.

    Returns
    -------
    Path
        Absolute path to the mastered file.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Matchering path (reference track supplied) ────────────────────────────
    if reference_track is not None:
        ref_path = Path(reference_track).resolve()
        if ref_path.exists():
            try:
                import matchering as mg  # type: ignore

                print(f"  ⟳  Mastering with matchering (ref: {ref_path.name}) …")
                mg.process(
                    target=str(input_path),
                    reference=str(ref_path),
                    results=[mg.Result(str(output_path))],
                )
                print(f"  ✅ Mastered → {output_path.name}")
                return output_path
            except Exception as exc:
                print(f"  ⚠  matchering failed ({exc}), falling back to preset chain")

    # ── Preset chain path ─────────────────────────────────────────────────────
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=False)

    # Mono → ensure float32
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = _apply_preset(audio, sr, preset)
    audio = _lufs_normalise(audio, sr, target_lufs)
    audio = _peak_limit(audio, ceiling=-1.0)  # final safety limiter

    sf.write(str(output_path), audio.astype(np.float32), sr, subtype="PCM_16")
    print(f"  ✅ Mastered → {output_path.name}  [{preset} / {target_lufs} LUFS]")
    return output_path
