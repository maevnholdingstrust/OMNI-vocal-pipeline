"""
rvc_engine.py — RVC voice conversion using only a .PTH model file.

No .index file is required.  Set ``index_rate=0`` (the default) and
``rvc_python`` skips the FAISS index entirely.

Dependencies
------------
    pip install rvc-python

The ``rvc-python`` package bundles the RVC inference stack (HuBERT,
RMVPE pitch extractor, HiFi-GAN vocoder) without the full WebUI repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def _detect_device(preference: str = "auto") -> str:
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preference


class RVCEngine:
    """
    Thin wrapper around ``rvc_python.infer.RVCInference``.

    Parameters
    ----------
    pth_path : str | Path
        Path to the ``.pth`` voice model file.
    f0_method : str
        Pitch detection algorithm: ``"rmvpe"`` (default), ``"harvest"``,
        ``"crepe"``, or ``"pm"``.
    f0_up_key : int
        Semitone shift (+/-).  0 = no shift.
    filter_radius : int
        Median filter window for pitch smoothing (1-7).  3 recommended.
    rms_mix_rate : float
        Mix ratio for output volume envelope (0.0-1.0).
    protect : float
        Consonant-protection strength (0.0-0.5).  0.33 recommended.
    device : str
        ``"auto"`` (default), ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``.
    """

    def __init__(
        self,
        pth_path: str | Path,
        f0_method: str = "rmvpe",
        f0_up_key: int = 0,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        device: str = "auto",
    ) -> None:
        self.pth_path = Path(pth_path).expanduser().resolve()
        if not self.pth_path.exists():
            raise FileNotFoundError(
                f"Voice model not found: {self.pth_path}\n"
                "Set model.pth_path in config.yaml to your .pth file."
            )

        self.f0_method = f0_method
        self.f0_up_key = f0_up_key
        self.filter_radius = filter_radius
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect
        self.device = _detect_device(device)

        self._rvc: Optional[object] = None  # lazy-loaded

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._rvc is not None:
            return
        try:
            from rvc_python.infer import RVCInference  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rvc-python is not installed.  Run: pip install rvc-python"
            ) from exc

        print(f"  ⟳  Loading RVC model from {self.pth_path} …")
        rvc = RVCInference(device=self.device)
        rvc.load_model(str(self.pth_path))
        self._rvc = rvc
        print("  ✅ RVC model ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Run RVC voice conversion on a WAV file.

        Parameters
        ----------
        input_path : str | Path
            Raw vocal WAV produced by the TTS engine.
        output_path : str | Path
            Destination path for the converted WAV.

        Returns
        -------
        Path
            Absolute path to the written output file.
        """
        self._load()

        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._rvc.infer_file(  # type: ignore[union-attr]
            input_path=str(input_path),
            output_path=str(output_path),
            f0_method=self.f0_method,
            f0_up_key=self.f0_up_key,
            index_rate=0,          # no .index file — disabled
            filter_radius=self.filter_radius,
            resample_sr=0,         # preserve original SR
            rms_mix_rate=self.rms_mix_rate,
            protect=self.protect,
        )
        return output_path
