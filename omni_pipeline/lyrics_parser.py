"""
lyrics_parser.py — Parse Suno-style lyric blocks into structured sections.

Supported block tags (case-insensitive):
  [Intro]  [Verse]  [Verse 1]  [Pre-Chorus]  [Chorus]
  [Hook]   [Bridge] [Break]    [Outro]        [Instrumental]
  [Ad-lib] [Refrain] [Drop]    [Build]        [Tag]

Inline annotations like (ad-lib: woo) or [laughs] are stripped from
the spoken text but preserved in the metadata for the renderer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── Section ordering weights (for informational use) ─────────────────────────
_SECTION_ORDER = [
    "intro", "verse", "pre-chorus", "chorus", "hook",
    "bridge", "break", "refrain", "drop", "build", "tag", "outro",
    "instrumental", "ad-lib",
]

# ── Regex patterns ────────────────────────────────────────────────────────────
_BLOCK_TAG_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$", re.MULTILINE)
_INLINE_ANNOTATION_RE = re.compile(
    r"\((?:ad.?lib|laugh|spoken|scream|whisper)[^)]*\)",
    re.IGNORECASE,
)
_MUSICAL_NOTE_RE = re.compile(r"[♪♫✦~*]")


@dataclass
class LyricSection:
    """A single structural section of a song."""
    tag: str            # raw tag text, e.g. "Verse 1"
    section_type: str   # normalised type, e.g. "verse"
    index: int          # 1-based occurrence index for this type
    lines: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Plain text of this section (stripped of inline annotations)."""
        cleaned = []
        for line in self.lines:
            line = _INLINE_ANNOTATION_RE.sub("", line)
            line = _MUSICAL_NOTE_RE.sub("", line)
            line = line.strip()
            if line:
                cleaned.append(line)
        return " ".join(cleaned)

    @property
    def is_instrumental(self) -> bool:
        # Both 'instrumental' and 'break' are treated as non-vocal sections
        # and are skipped during synthesis (no TTS output generated).
        return self.section_type in ("instrumental", "break")


def _normalise_type(tag: str) -> str:
    """Map a raw tag string to a canonical section type."""
    tag_lower = tag.lower().strip()
    for key in _SECTION_ORDER:
        if tag_lower.startswith(key):
            return key
    # fall-through: treat unknown tags as generic verse
    return "verse"


def parse_lyrics(raw: str) -> List[LyricSection]:
    """
    Parse a Suno-style lyric block string into an ordered list of
    :class:`LyricSection` objects.

    Parameters
    ----------
    raw : str
        Multiline string with ``[Tag]`` headers and lyric lines.

    Returns
    -------
    list[LyricSection]
        Ordered list of sections ready for TTS rendering.

    Example
    -------
    >>> sections = parse_lyrics(\"\"\"
    ... [Verse 1]
    ... I walk alone under neon skies
    ... The city never sleeps, neither do I
    ...
    ... [Chorus]
    ... Echoes of the night, calling out my name
    ... \"\"\")
    >>> sections[0].section_type
    'verse'
    >>> sections[1].section_type
    'chorus'
    """
    sections: List[LyricSection] = []
    type_counts: dict[str, int] = {}

    # Split into (tag, body) pairs
    parts = _BLOCK_TAG_RE.split(raw)

    # parts = [preamble, tag1, body1, tag2, body2, ...]
    # If the file starts without a tag, parts[0] is preamble text — skip it.
    if len(parts) < 3:
        # No block tags found — treat entire input as a single verse
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        if lines:
            sections.append(
                LyricSection(
                    tag="Verse",
                    section_type="verse",
                    index=1,
                    lines=lines,
                )
            )
        return sections

    # Iterate over tag+body pairs (skip index 0 which is preamble)
    it = iter(parts[1:])
    for tag, body in zip(it, it):
        stype = _normalise_type(tag)
        type_counts[stype] = type_counts.get(stype, 0) + 1

        lines = [l.strip() for l in body.strip().splitlines() if l.strip()]

        sections.append(
            LyricSection(
                tag=tag.strip(),
                section_type=stype,
                index=type_counts[stype],
                lines=lines,
            )
        )

    return sections


def pretty_print(sections: List[LyricSection]) -> None:
    """Print parsed sections to stdout for debugging."""
    for sec in sections:
        print(f"\n[{sec.tag}]  (type={sec.section_type}, #{sec.index})")
        for line in sec.lines:
            print(f"  {line}")
