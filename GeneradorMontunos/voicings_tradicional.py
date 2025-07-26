# -*- coding: utf-8 -*-
"""Utilities for generating piano voicings."""

from typing import List, Optional, Tuple
import logging

# ---------------------------------------------------------------------------
# Pitch range limits for the generated voicings.  Notes are adjusted so that
# they remain within this interval when building the linked voicings.
# These limits should only affect the base voicings; harmonisation later on
# (octaves, double octaves, tenths or sixths) may exceed ``RANGO_MAX``.
# ---------------------------------------------------------------------------
RANGO_MIN = 53  # F3
RANGO_MAX = 67  # G4
RANGO_EXTRA = 4  # flexible extension above and below
SALTO_MAX = 8  # maximum leap in semitones

logger = logging.getLogger(__name__)

# ==========================================================================
# Dictionaries for chord suffixes and note names
# These are used to parse chord symbols and build chord voicings
# ===========================================================================

INTERVALOS_TRADICIONALES = {
    '6':      [0, 4, 7, 9],     # 1 3 5 6
    '7':      [0, 4, 7, 10],    # 1 3 5 b7
    '∆':      [0, 4, 7, 11],    # 1 3 5 7
    'm6':     [0, 3, 7, 9],     # 1 b3 5 6
    'm7':     [0, 3, 7, 10],    # 1 b3 5 b7
    'm∆':     [0, 3, 7, 11],    # 1 b3 5 7
    '+7':     [0, 4, 8, 10],    # 1 3 #5 b7
    '∆sus4':  [0, 5, 7, 11],    # 1 4 5 7
    '∆sus2':  [0, 2, 7, 11],    # 1 2 5 7
    '7sus4':  [0, 5, 7, 10],    # 1 4 5 b7
    '7sus2':  [0, 2, 7, 10],    # 1 2 5 b7
    'º7':     [0, 3, 6, 9],     # 1 b3 b5 bb7 (bb7 = 6ma mayor)
    'º∆':     [0, 3, 6, 11],    # 1 b3 b5 7
    'm7(b5)':      [0, 3, 6, 10],    # 1 b3 b5 b7
    '7(b5)':  [0, 4, 6, 10],    # 1 3 b5 b7
    '7(b9)':  [0, 4, 7, 10, 13],  # 1 3 5 b7 b9
    '+7(b9)': [0, 4, 8, 10, 13],  # 1 3 #5 b7 b9
    '7(b5)b9': [0, 4, 6, 10, 13],  # 1 3 b5 b7 b9
    '7sus4(b9)': [0, 5, 7, 10, 13],  # 1 4 5 b7 b9
    '∆(b5)':  [0, 4, 6, 11],    # 1 3 b5 7
}

NOTAS = {
    'C':0,  'B#':0,
    'C#':1, 'Db':1,
    'D':2,
    'D#':3,'Eb':3,
    'E':4, 'Fb':4,
    'F':5, 'E#':5,
    'F#':6,'Gb':6,
    'G':7,
    'G#':8,'Ab':8,
    'A':9,
    'A#':10,'Bb':10,
    'B':11, 'Cb':11,
}

# ==========================================================================
# Chord parsing and linked voicing generation
# ==========================================================================

def parsear_nombre_acorde(nombre: str) -> Tuple[int, str]:
    """Parse a chord name into root MIDI pitch class and suffix."""
    import re

    base = re.sub(r"/[1357]$", "", nombre)

    m = re.match(
        r'^([A-G][b#]?)(m6|m7|m∆|m|6|7|∆sus4|∆sus2|∆|\+7|º7|º∆|m7\(b5\)|7sus4|7sus2|7\(b5\)|7\(b9\)|\+7\(b9\)|7\(b5\)b9|7sus4\(b9\)|∆\(b5\))?$',
        base,
    )
    if not m:
        raise ValueError(f"Acorde no reconocido: {nombre}")
    root, suf = m.group(1), m.group(2) or '∆'
    return NOTAS[root], suf


def _ajustar_octava(pitch: int) -> int:
    """Confine ``pitch`` within ``RANGO_MIN`` .. ``RANGO_MAX`` by octaves."""

    while pitch < RANGO_MIN:
        pitch += 12
    while pitch > RANGO_MAX:
        pitch -= 12
    return pitch


def _ajustar_octava_flexible(pitch: int, prev: Optional[int]) -> int:
    """Adjust ``pitch`` preferring the fixed range but allowing a small extension."""

    def clamp(p: int, lo: int, hi: int) -> int:
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        return p

    base = clamp(pitch, RANGO_MIN, RANGO_MAX)
    if prev is None or abs(base - prev) <= SALTO_MAX:
        return base

    ext = clamp(pitch, RANGO_MIN - RANGO_EXTRA, RANGO_MAX + RANGO_EXTRA)
    if abs(ext - prev) < abs(base - prev):
        return ext
    return base


def generar_voicings_enlazados_tradicional(progresion: List[str]) -> List[List[int]]:
    """Generate linked four‑note voicings in the traditional style.

    The bass voice is **never** the root (interval ``0``) nor the third
    (interval ``x``) of the chord.  Only the intervals ``y`` or ``z`` can be
    placed in the lowest voice.  The chosen bass is the option (``y`` or ``z``)
    closest to the previous bass note.  The rest of the chord tones are stacked
    above in ascending order.
    """

    import pretty_midi
    import re

    referencia = [55, 57, 60, 64]  # default positions for the four voices
    voicings: List[List[int]] = []
    bajo_anterior = referencia[0]

    def ajustar(pc: int, target: int) -> int:
        """Return ``pc`` adjusted near ``target`` without range limits."""
        pitch = target + ((pc - target) % 12)
        if abs(pitch - target) > abs(pitch - 12 - target):
            pitch -= 12
        return pitch

    inv_map = {"1": "root", "3": "third", "5": "fifth", "7": "seventh"}

    for nombre in progresion:
        m = re.search(r"/([1357])$", nombre)
        inv_forzado = None
        if m:
            inv_forzado = inv_map[m.group(1)]
            nombre = nombre[: m.start()]

        root, suf = parsear_nombre_acorde(nombre)
        ints = INTERVALOS_TRADICIONALES[suf]
        pcs = [(root + i) % 12 for i in ints]

        # ------------------------------------------------------------------
        # Elegir el bajo solamente entre las notas correspondientes a ``y``
        # o ``z``.  La fundamental y la tercera nunca se usan en la voz baja.
        # ------------------------------------------------------------------
        if inv_forzado:
            idx_map = {"root": 0, "third": 1, "fifth": 2, "seventh": 3}
            idx = idx_map.get(inv_forzado, 0)
            pc_bajo = pcs[idx]
            bajo_tmp = ajustar(pc_bajo, bajo_anterior)
            bajo = _ajustar_octava_flexible(bajo_tmp, bajo_anterior)
            restantes_pcs = [p for i, p in enumerate(pcs) if i != idx][:3]
            bajo_intervalo = inv_forzado
        else:
            pc_y, pc_z = pcs[2], pcs[3]
            cand_y = _ajustar_octava_flexible(ajustar(pc_y, bajo_anterior), bajo_anterior)
            cand_z = _ajustar_octava_flexible(ajustar(pc_z, bajo_anterior), bajo_anterior)
            if abs(cand_y - bajo_anterior) <= abs(cand_z - bajo_anterior):
                bajo = cand_y
                restantes_pcs = [pcs[0], pcs[1], pc_z]
                bajo_intervalo = "y"
            else:
                bajo = cand_z
                restantes_pcs = [pcs[0], pcs[1], pc_y]
                bajo_intervalo = "z"

        notas_restantes: List[int] = []
        for pc, ref in zip(restantes_pcs, referencia[1:]):
            pitch = ajustar(pc, ref)
            # Asegura que todas las notas queden por encima del bajo
            while pitch <= bajo:
                pitch += 12
            pitch = _ajustar_octava(pitch)
            while pitch <= bajo:
                pitch += 12
            notas_restantes.append(pitch)

        voicing = sorted([bajo] + notas_restantes)

        # --------------------------------------------------------------
        # Special case for chords rooted on C: optionally shift the
        # entire voicing one octave up if that yields a smoother bass
        # connection.
        # --------------------------------------------------------------
        if root % 12 == 0:
            alt_voicing = [n + 12 for n in voicing]
            if abs(alt_voicing[0] - bajo_anterior) < abs(voicing[0] - bajo_anterior):
                voicing = alt_voicing
                bajo = voicing[0]

        voicings.append(voicing)
        nombres = [pretty_midi.note_number_to_name(n) for n in voicing]
        logger.debug(
            "%s: %s - bajo %s (%s)",
            nombre,
            nombres,
            pretty_midi.note_number_to_name(voicing[0]),
            bajo_intervalo,
        )
        bajo_anterior = voicing[0]

    return voicings

# Future voicing strategies for other modes can be added here
