#!/usr/bin/env python3
"""
run_pipeline.py — CLI entry point for the OMNI Vocal Pipeline.

Usage
-----
  # Basic (reads lyrics from a .txt file):
  python run_pipeline.py lyrics.txt

  # Inline lyrics:
  python run_pipeline.py --lyrics "[Verse 1]\\nLine 1\\nLine 2\\n[Chorus]\\nChorus line"

  # Custom config and output name:
  python run_pipeline.py lyrics.txt --config my_config.yaml --name my_song

  # Show help:
  python run_pipeline.py --help

Lyrics file format
------------------
  [Verse 1]
  I walk alone under neon skies
  The city never sleeps, neither do I

  [Chorus]
  Echoes of the night, calling out my name
  We're burning bright, and we'll never be the same

  [Bridge]
  In the silence I can hear you calling

Supported section tags: Intro, Verse, Pre-Chorus, Chorus, Hook,
  Bridge, Break, Refrain, Drop, Build, Tag, Outro, Instrumental, Ad-lib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="OMNI Vocal Pipeline — lyrics → synthesized vocals in your voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "lyrics_file",
        nargs="?",
        type=Path,
        metavar="LYRICS_FILE",
        help="Path to a .txt file containing Suno-style lyric blocks.",
    )
    src.add_argument(
        "--lyrics",
        "-l",
        type=str,
        metavar="TEXT",
        help="Lyrics as an inline string (use \\n for newlines).",
    )
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.yaml"),
        metavar="CONFIG",
        help="Path to config.yaml (default: ./config.yaml).",
    )
    p.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        metavar="OUTPUT_NAME",
        help="Base name for output files, e.g. 'my_song' → my_song_mastered.wav.",
    )
    p.add_argument(
        "--pth",
        type=Path,
        default=None,
        metavar="PTH_PATH",
        help="Override the .pth model path from config.yaml.",
    )
    p.add_argument(
        "--preset",
        choices=["streaming", "vinyl", "club", "bright_pop", "raw"],
        default=None,
        metavar="PRESET",
        help="Override mastering preset (streaming|vinyl|club|bright_pop|raw).",
    )
    p.add_argument(
        "--no-master",
        action="store_true",
        help="Skip mastering — output raw RVC vocal only.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Override compute device (auto|cpu|cuda|cuda:0|mps).",
    )
    return p


def _patch_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of the loaded config dict."""
    if args.pth:
        cfg.setdefault("model", {})["pth_path"] = str(args.pth)
    if args.preset:
        cfg.setdefault("mastering", {})["preset"] = args.preset
    if args.no_master:
        cfg.setdefault("mastering", {})["enabled"] = False
    if args.device:
        cfg["device"] = args.device
    return cfg


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Resolve lyrics string ─────────────────────────────────────────────────
    if args.lyrics_file:
        lyrics_path = args.lyrics_file.expanduser().resolve()
        if not lyrics_path.exists():
            parser.error(f"Lyrics file not found: {lyrics_path}")
        lyrics = lyrics_path.read_text(encoding="utf-8")
        stem = args.name or lyrics_path.stem
    else:
        lyrics = args.lyrics.replace("\\n", "\n")
        stem = args.name or "output"

    # ── Resolve config ────────────────────────────────────────────────────────
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        parser.error(
            f"config.yaml not found: {config_path}\n"
            "Copy the sample config.yaml to the same directory and edit it."
        )

    # ── Apply CLI overrides to config ─────────────────────────────────────────
    import yaml  # type: ignore

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    cfg = _patch_config(cfg, args)

    # Write a temporary patched config so pipeline.run() can read it
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp_config = tmp.name

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        from omni_pipeline.pipeline import run

        out = run(lyrics=lyrics, config_path=tmp_config, output_name=stem)
        print(f"\n🎤 Vocal render complete → {out}\n")
    finally:
        import os

        os.unlink(tmp_config)


if __name__ == "__main__":
    main()
