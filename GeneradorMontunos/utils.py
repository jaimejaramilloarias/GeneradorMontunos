from __future__ import annotations

"""Miscellaneous helper functions used across the GUI."""

from typing import Callable, Iterable, List, Optional, Tuple
import re
import pretty_midi

__all__ = [
    "RE_BAR_CLEAN",
    "limpiar_inversion",
    "apply_manual_edits",
    "calc_default_inversions",
    "normalise_bars",
    "clean_tokens",
]

RE_BAR_CLEAN = re.compile(r"\|\s*\|+")


def limpiar_inversion(valor: str) -> str:
    """Remove duplicated inversion suffixes like ``root_root``."""
    if "_" in valor:
        valor = valor.split("_")[0]
    return valor


def apply_manual_edits(pm: pretty_midi.PrettyMIDI, edits: Iterable[dict]) -> None:
    """Apply recorded manual edits to a ``PrettyMIDI`` object."""
    inst = pm.instruments[0]
    for ed in edits:
        typ = ed.get("type", "modify")
        if typ == "modify":
            for n in inst.notes:
                if abs(n.start - ed["start"]) < 1e-6 and abs(n.end - ed["end"]) < 1e-6:
                    n.pitch = ed["pitch"]
                    break
        elif typ == "add":
            inst.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=ed["pitch"],
                    start=ed["start"],
                    end=ed["end"],
                )
            )
        elif typ == "delete":
            for n in list(inst.notes):
                if (
                    abs(n.start - ed["start"]) < 1e-6
                    and abs(n.end - ed["end"]) < 1e-6
                    and n.pitch == ed["pitch"]
                ):
                    inst.notes.remove(n)
                    break
    inst.notes.sort(key=lambda n: n.start)


def calc_default_inversions(
    asignaciones,
    inversion_getter: Callable[[], str],
    get_bass_pitch: Callable[[str, str], int],
    ajustar_rango_flexible: Callable[[Optional[int], int], int],
    seleccionar_inversion: Callable[[Optional[int], str], Tuple[str, int]],
) -> List[str]:
    """Return default inversions for ``asignaciones`` using the salsa helpers."""
    invs: List[str] = []
    voz = None
    for idx, data in enumerate(asignaciones):
        cif = data[0]
        inv_for = data[3] if len(data) > 3 else None
        if idx == 0:
            inv = inv_for or limpiar_inversion(inversion_getter())
            pitch = get_bass_pitch(cif, inv)
            pitch = ajustar_rango_flexible(voz, pitch)
        else:
            if inv_for:
                inv = inv_for
                pitch = get_bass_pitch(cif, inv)
                pitch = ajustar_rango_flexible(voz, pitch)
            else:
                inv, pitch = seleccionar_inversion(voz, cif)
        invs.append(inv)
        voz = pitch
    return invs


def normalise_bars(text: str) -> str:
    """Remove duplicated barlines and tidy spacing."""
    lines = []
    for ln in text.splitlines():
        ln = RE_BAR_CLEAN.sub("|", ln)
        parts = [p.strip() for p in ln.split("|")]
        ln = " | ".join(p for p in parts if p)
        lines.append(ln)
    return "\n".join(lines)


def clean_tokens(txt: str) -> str:
    """Return *txt* unchanged. Placeholder for future logic."""
    return txt
