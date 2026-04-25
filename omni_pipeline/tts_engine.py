"""
tts_engine.py — Convert lyric text to raw audio via Bark or Edge-TTS.

Bark is recommended for singing-like synthesis; it wraps each lyric block
in ♪ … ♪ notation so the model renders musical phrasing rather than flat
speech.  Edge-TTS is a lightweight fall-back that produces clean speech
(useful for testing without a GPU).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# ── Lazy imports (heavy models only loaded on first use) ─────────────────────
_bark_loaded = False
_bark_generate = None
_bark_sample_rate = None


def _load_bark() -> None:
    global _bark_loaded, _bark_generate, _bark_sample_rate
    if _bark_loaded:
        return
    try:
        from bark import generate_audio, preload_models, SAMPLE_RATE  # type: ignore
        print("  ⟳  Loading Bark models (first run downloads ~5 GB) …")
        preload_models()
        _bark_generate = generate_audio
        _bark_sample_rate = SAMPLE_RATE
        _bark_loaded = True
        print("  ✅ Bark ready")
    except ImportError as exc:
        raise ImportError(
            "suno-bark is not installed.  Run: pip install suno-bark"
        ) from exc


class TTSEngine:
    """
    Wrapper around a TTS back-end that converts a text string to a WAV file.

    Parameters
    ----------
    engine : str
        ``"bark"`` (default) or ``"edge-tts"``.
    voice_preset : str
        Bark voice preset string, e.g. ``"v2/en_speaker_6"``.
    singing_mode : bool
        When True (default) each phrase is wrapped in ♪ … ♪ so Bark
        renders it as singing rather than flat speech.
    edge_tts_voice : str
        Voice name used when *engine* is ``"edge-tts"``.
    """

    def __init__(
        self,
        engine: str = "bark",
        voice_preset: str = "v2/en_speaker_6",
        singing_mode: bool = True,
        edge_tts_voice: str = "en-US-JennyNeural",
    ) -> None:
        self.engine = engine.lower()
        self.voice_preset = voice_preset
        self.singing_mode = singing_mode
        self.edge_tts_voice = edge_tts_voice

        if self.engine == "bark":
            _load_bark()

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        """
        Synthesize *text* and write the result to *output_path* (WAV).

        Returns
        -------
        Path
            Absolute path to the written WAV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine == "bark":
            return self._bark_synth(text, output_path)
        elif self.engine == "edge-tts":
            return self._edge_synth(text, output_path)
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine!r}")

    def synthesize_section(
        self, lines: list[str], output_path: str | Path
    ) -> Path:
        """
        Synthesize a list of lyric lines as a single audio segment.

        Each line is rendered individually by Bark (for natural phrasing
        between lines) and the resulting clips are concatenated.
        """
        if not lines:
            raise ValueError("synthesize_section received empty lines list")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine == "bark":
            clips: list[np.ndarray] = []
            for line in lines:
                if not line.strip():
                    continue
                prompt = self._make_singing_prompt(line) if self.singing_mode else line
                audio = _bark_generate(  # type: ignore[misc]
                    prompt,
                    history_prompt=self.voice_preset,
                )
                clips.append(audio)
                # Short silence between lines (~0.3 s)
                silence = np.zeros(int(_bark_sample_rate * 0.3), dtype=np.float32)
                clips.append(silence)

            combined = np.concatenate(clips).astype(np.float32)
            sf.write(str(output_path), combined, _bark_sample_rate)
            return output_path

        else:
            # For edge-tts, join lines with a pause hint and synthesize once
            joined = ",  ".join(line for line in lines if line.strip())
            return self._edge_synth(joined, output_path)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_singing_prompt(text: str) -> str:
        """Wrap text in musical note symbols for Bark singing mode."""
        text = text.strip().rstrip(".,!?")
        return f"♪ {text} ♪"

    def _bark_synth(self, text: str, output_path: Path) -> Path:
        prompt = self._make_singing_prompt(text) if self.singing_mode else text
        audio: np.ndarray = _bark_generate(  # type: ignore[misc]
            prompt,
            history_prompt=self.voice_preset,
        )
        sf.write(str(output_path), audio.astype(np.float32), _bark_sample_rate)
        return output_path

    def _edge_synth(self, text: str, output_path: Path) -> Path:
        try:
            import asyncio
            import edge_tts  # type: ignore

            async def _run() -> None:
                communicate = edge_tts.Communicate(text, self.edge_tts_voice)
                # edge-tts writes MP3; we convert via soundfile
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                await communicate.save(tmp_path)

                from pydub import AudioSegment  # type: ignore
                seg = AudioSegment.from_mp3(tmp_path)
                seg = seg.set_frame_rate(44100).set_channels(1)
                seg.export(str(output_path), format="wav")
                os.unlink(tmp_path)

            asyncio.run(_run())
        except ImportError as exc:
            raise ImportError(
                "edge-tts is not installed.  Run: pip install edge-tts"
            ) from exc
        return output_path
