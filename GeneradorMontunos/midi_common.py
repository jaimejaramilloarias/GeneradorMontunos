from __future__ import annotations

"""Shared MIDI helper utilities used by both modes."""

from pathlib import Path
from typing import List
import random
import logging
import pretty_midi

__all__ = [
    "NOTAS_BASE",
    "leer_midi_referencia",
    "obtener_posiciones_referencia",
    "detectar_notas_base",
    "construir_posiciones_secuenciales",
    "construir_posiciones_por_ventanas",
]

# Baseline notes present in the reference MIDI to be replaced by generated voicings
NOTAS_BASE = [55, 57, 60, 64]  # G3, A3, C4, E4

logger = logging.getLogger(__name__)


def leer_midi_referencia(midi_path: Path):
    """Load reference MIDI and return its notes and the PrettyMIDI object."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    instrumento = pm.instruments[0]
    notes = sorted(instrumento.notes, key=lambda n: n.start)
    for n in notes:
        nombre = pretty_midi.note_number_to_name(int(n.pitch))
        logger.debug("%s (%s)", n.pitch, nombre)
    logger.debug("Total de notas: %s", len(notes))
    return notes, pm


def detectar_notas_base(notes) -> List[int]:
    """Return the sorted set of pitches present in ``notes``."""
    base = sorted({int(n.pitch) for n in notes if n.pitch > 0})
    logger.debug("Notas base detectadas: %s", base)
    return base


def obtener_posiciones_referencia(
    notes, notas_base: List[int] | None = None
) -> tuple[list[dict], list[int]]:
    """Return rhythmic info for the reference template and its base pitches."""

    if notas_base is None:
        notas_base = detectar_notas_base(notes)

    posiciones = []
    for n in notes:
        pitch = int(n.pitch)
        if pitch in [int(p) for p in notas_base]:
            posiciones.append(
                {
                    "pitch": pitch,
                    "start": n.start,
                    "end": n.end,
                    "velocity": n.velocity,
                }
            )
            nombre = pretty_midi.note_number_to_name(pitch)
            logger.debug("Nota base %s (%s) inicio %s", pitch, nombre, n.start)
    posiciones.sort(key=lambda x: (x["start"], x["pitch"]))
    logger.debug("Notas base encontradas: %s", len(posiciones))
    ejemplo = [(p["pitch"], p["start"]) for p in posiciones[:10]]
    logger.debug("Ejemplo primeros 10: %s", ejemplo)
    return posiciones, notas_base


def construir_posiciones_secuenciales(
    posiciones_base: List[dict],
    total_cor_dest: int,
    total_cor_ref: int,
    grid_seg: float,
    *,
    inicio_cor: int = 0,
) -> List[dict]:
    """Build note positions repeating the reference sequentially."""

    grupos_ref: List[List[dict]] = [[] for _ in range(total_cor_ref)]
    for pos in posiciones_base:
        idx = int(round(pos["start"] / grid_seg))
        if 0 <= idx < total_cor_ref:
            grupos_ref[idx].append(
                {
                    "pitch": pos["pitch"],
                    "start": pos["start"] - idx * grid_seg,
                    "end": pos["end"] - idx * grid_seg,
                    "velocity": pos["velocity"],
                }
            )

    posiciones: List[dict] = []
    for dest_idx in range(total_cor_dest):
        ref_idx = (inicio_cor + dest_idx) % total_cor_ref
        for nota in grupos_ref[ref_idx]:
            posiciones.append(
                {
                    "pitch": nota["pitch"],
                    "start": round(dest_idx * grid_seg + nota["start"], 6),
                    "end": round(dest_idx * grid_seg + nota["end"], 6),
                    "velocity": nota["velocity"],
                }
            )

    posiciones.sort(key=lambda x: (x["start"], x["pitch"]))
    return posiciones


def construir_posiciones_por_ventanas(
    posiciones_base: List[dict],
    total_cor_dest: int,
    total_cor_ref: int,
    grid_seg: float,
    *,
    inicio_cor: int = 0,
    compases_ventana: int = 4,
    aleatorio: bool = True,
) -> List[dict]:
    """Build note positions choosing fixed-size windows from the reference."""

    inicio_cor = inicio_cor % total_cor_ref

    ventana_cor = compases_ventana * 8
    num_ventanas = max(1, total_cor_ref // ventana_cor)

    grupos_ref: List[List[dict]] = [[] for _ in range(total_cor_ref)]
    for pos in posiciones_base:
        idx = int(round(pos["start"] / grid_seg))
        if 0 <= idx < total_cor_ref:
            grupos_ref[idx].append(
                {
                    "pitch": pos["pitch"],
                    "start": pos["start"] - idx * grid_seg,
                    "end": pos["end"] - idx * grid_seg,
                    "velocity": pos["velocity"],
                }
            )

    posiciones: List[dict] = []
    start_block = inicio_cor // ventana_cor
    end_block = (inicio_cor + total_cor_dest - 1) // ventana_cor
    num_blocks = end_block - start_block + 1

    if aleatorio:
        orden_inicial = list(range(num_ventanas))
        random.shuffle(orden_inicial)
        ventanas_por_bloque = orden_inicial[:num_blocks]
        while len(ventanas_por_bloque) < num_blocks:
            ventanas_por_bloque.append(random.randint(0, num_ventanas - 1))
    else:
        ventanas_por_bloque = [
            (start_block + i) % num_ventanas for i in range(num_blocks)
        ]

    for dest_idx in range(total_cor_dest):
        cor_global = inicio_cor + dest_idx + 1
        pos_relativa = (cor_global - 1) % ventana_cor
        bloque = (cor_global - 1) // ventana_cor - start_block
        ventana = ventanas_por_bloque[bloque]
        ref_idx = (ventana * ventana_cor + pos_relativa) % total_cor_ref
        for nota in grupos_ref[ref_idx]:
            posiciones.append(
                {
                    "pitch": nota["pitch"],
                    "start": round(dest_idx * grid_seg + nota["start"], 6),
                    "end": round(dest_idx * grid_seg + nota["end"], 6),
                    "velocity": nota["velocity"],
                }
            )

    posiciones.sort(key=lambda x: (x["start"], x["pitch"]))
    return posiciones
