"""
pipeline.py — End-to-end orchestration:

  Lyrics (text)
    → [LyricsParser]     extract structural sections
    → [TTSEngine]        synthesize raw audio per section (Bark singing)
    → [RVCEngine]        convert each clip to user's .PTH voice
    → [pydub concat]     assemble sections with crossfades / silence gaps
    → [Mastering]        loudness, EQ, limiting
    → final WAV

Usage (from Python)
-------------------
    from omni_pipeline import run
    run(lyrics=open("song.txt").read(), config_path="config.yaml")
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment  # type: ignore

from .lyrics_parser import parse_lyrics, LyricSection
from .tts_engine import TTSEngine
from .rvc_engine import RVCEngine
from .mastering import master


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config(config_path: str | Path) -> dict:
    import yaml  # type: ignore

    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def _get_nested_config(cfg: dict, *keys, default=None):
    """Safely walk a nested dict and return the value at the given key path."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


# ── Audio assembly ────────────────────────────────────────────────────────────

def _concat_wavs(
    wav_paths: list[Path],
    output_path: Path,
    crossfade_ms: int = 200,
    silence_ms: int = 400,
) -> Path:
    """Concatenate WAV files with crossfade + silence gaps."""
    if not wav_paths:
        raise ValueError("No WAV files to concatenate")

    combined: Optional[AudioSegment] = None
    for p in wav_paths:
        seg = AudioSegment.from_wav(str(p))
        if combined is None:
            combined = seg
        else:
            silence = AudioSegment.silent(duration=silence_ms)
            combined = combined.append(silence, crossfade=0)
            combined = combined.append(seg, crossfade=crossfade_ms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format="wav")
    return output_path


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    lyrics: str,
    config_path: str | Path = "config.yaml",
    output_name: Optional[str] = None,
) -> Path:
    """
    Run the full vocals pipeline.

    Parameters
    ----------
    lyrics : str
        Raw Suno-style lyric block string.
    config_path : str | Path
        Path to ``config.yaml``.
    output_name : str | None
        Base name for output files (no extension).  Defaults to ``"output"``.

    Returns
    -------
    Path
        Path to the final mastered WAV file.
    """
    config_path = Path(config_path).expanduser().resolve()
    cfg = _load_config(config_path)

    out_dir = Path(_get_nested_config(cfg, "output", "dir", default="output")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = output_name or "output"
    tmp_dir = Path(tempfile.mkdtemp(prefix="omni_"))

    print("\n════════════════════════════════════════")
    print("  OMNI Vocal Pipeline — starting run")
    print(f"  output dir : {out_dir}")
    print(f"  temp dir   : {tmp_dir}")
    print("════════════════════════════════════════\n")

    # ── 1. Parse lyrics ───────────────────────────────────────────────────────
    print("● [1/4] Parsing lyrics …")
    sections = parse_lyrics(lyrics)
    if not sections:
        raise ValueError("No lyric sections found.  Check your input format.")

    print(f"  Found {len(sections)} section(s):")
    for s in sections:
        marker = "⚙  (instrumental — skipped)" if s.is_instrumental else ""
        print(f"    [{s.tag}]  {len(s.lines)} line(s)  {marker}")

    vocal_sections = [s for s in sections if not s.is_instrumental]
    if not vocal_sections:
        raise ValueError("All sections are instrumental — nothing to synthesize.")

    # ── 2. TTS synthesis ──────────────────────────────────────────────────────
    print("\n● [2/4] TTS synthesis …")
    tts_cfg = _get_nested_config(cfg, "tts") or {}
    tts = TTSEngine(
        engine=tts_cfg.get("engine", "bark"),
        voice_preset=tts_cfg.get("bark_voice_preset", "v2/en_speaker_6"),
        singing_mode=tts_cfg.get("singing_mode", True),
        edge_tts_voice=tts_cfg.get("edge_tts_voice", "en-US-JennyNeural"),
    )

    tts_wavs: list[Path] = []
    for i, sec in enumerate(vocal_sections):
        tts_out = tmp_dir / f"tts_{i:02d}_{sec.section_type}.wav"
        print(f"  Synthesizing [{sec.tag}] ({len(sec.lines)} lines) …")
        tts.synthesize_section(sec.lines, tts_out)
        tts_wavs.append(tts_out)

    # ── 3. RVC voice conversion ───────────────────────────────────────────────
    print("\n● [3/4] RVC voice conversion …")
    model_cfg = _get_nested_config(cfg, "model") or {}
    pth_path = _get_nested_config(cfg, "model", "pth_path", default=None)
    if not pth_path:
        raise ValueError(
            "model.pth_path is not set in config.yaml.\n"
            "Set it to the path of your .pth voice model file."
        )

    device_pref = _get_nested_config(cfg, "device", default="auto")
    rvc = RVCEngine(
        pth_path=pth_path,
        f0_method=model_cfg.get("f0_method", "rmvpe"),
        f0_up_key=int(model_cfg.get("f0_up_key", 0)),
        filter_radius=int(model_cfg.get("filter_radius", 3)),
        rms_mix_rate=float(model_cfg.get("rms_mix_rate", 0.25)),
        protect=float(model_cfg.get("protect", 0.33)),
        device=device_pref,
    )

    rvc_wavs: list[Path] = []
    for i, tts_wav in enumerate(tts_wavs):
        rvc_out = tmp_dir / f"rvc_{i:02d}_{vocal_sections[i].section_type}.wav"
        print(f"  Converting [{vocal_sections[i].tag}] …")
        rvc.convert(tts_wav, rvc_out)
        rvc_wavs.append(rvc_out)

    # ── Assemble all sections ─────────────────────────────────────────────────
    assembled_path = tmp_dir / f"{stem}_assembled.wav"
    out_cfg = _get_nested_config(cfg, "output") or {}
    _concat_wavs(
        rvc_wavs,
        assembled_path,
        crossfade_ms=int(out_cfg.get("crossfade_ms", 200)),
        silence_ms=int(out_cfg.get("silence_between_sections_ms", 400)),
    )

    # ── 4. Mastering ──────────────────────────────────────────────────────────
    print("\n● [4/4] Mastering …")
    master_cfg = _get_nested_config(cfg, "mastering") or {}
    final_path = out_dir / f"{stem}_mastered.wav"

    if master_cfg.get("enabled", True):
        ref_track = master_cfg.get("reference_track", None)
        master(
            input_path=assembled_path,
            output_path=final_path,
            preset=master_cfg.get("preset", "streaming"),
            target_lufs=float(master_cfg.get("target_lufs", -14.0)),
            reference_track=ref_track,
        )
    else:
        import shutil
        shutil.copy2(str(assembled_path), str(final_path))
        print(f"  ✅ Mastering disabled — copied assembled file → {final_path.name}")

    print("\n════════════════════════════════════════")
    print(f"  ✅ Done!  Output: {final_path}")
    print("════════════════════════════════════════\n")
    return final_path
