"""Helpers for style parsing and application."""
from typing import Callable, List, Tuple

__all__ = ["parse_styles", "apply_styles"]

def parse_styles(text: str, get_modo: Callable[[], str], get_armon: Callable[[], str]) -> Tuple[List[str], List[str], List[str]]:
    """Return mode, harmonisation and inversion lists for each chord."""
    segmentos_raw = [s.strip() for s in text.split("|") if s.strip()]
    segmentos: List[str] = []
    for seg in segmentos_raw:
        if seg == "%":
            if not segmentos:
                continue
            segmentos.append(segmentos[-1])
        else:
            segmentos.append(seg)
    num_chords = len(segmentos)
    modos = [get_modo()] * num_chords
    arms = [get_armon()] * num_chords
    invs = [None] * num_chords
    return modos, arms, invs


def apply_styles(base_text: str) -> str:
    """Return ``base_text`` unchanged."""
    return base_text
