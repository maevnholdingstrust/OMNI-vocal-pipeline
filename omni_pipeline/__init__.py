"""OMNI Vocal Pipeline — local lyrics-to-audio production."""

from .lyrics_parser import parse_lyrics


def run(*args, **kwargs):
    """Lazy-import wrapper so torch/bark are only loaded when actually used."""
    from .pipeline import run as _run  # noqa: PLC0415
    return _run(*args, **kwargs)


__all__ = ["run", "parse_lyrics"]
