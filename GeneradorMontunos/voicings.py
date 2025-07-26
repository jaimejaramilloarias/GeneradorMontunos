# -*- coding: utf-8 -*-
"""Utilities for generating piano voicings."""

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pitch range limits for the generated voicings.  Notes are adjusted so that
# they remain within this interval when building the linked voicings.
# These limits should only affect the base voicings; harmonisation later on
# (octaves, double octaves, tenths or sixths) may exceed ``RANGO_MAX``.
# ---------------------------------------------------------------------------
RANGO_MIN = 53  # F3
RANGO_MAX = 67  # G4

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
    'm7(b5)': [0, 3, 6, 10],    # 1 b3 b5 b7
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
    """Parse a chord name into root MIDI pitch class and suffix.

    Extra modifiers like ``(b6)`` or ``(b13)`` are ignored for the
    purposes of voicing generation but may be used elsewhere.
    """

    import re

    # Strip forced inversion notation at the end (e.g. C∆/3)
    base = re.sub(r"/[1357]$", "", nombre)
    # Remove optional extensions that are not part of the base dictionary
    base = re.sub(r"\(b6\)|\(b13\)", "", base)

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

SALTO_MAX = 8  # interval in semitones (minor sixth)


def generar_voicings_enlazados_tradicional(progresion: List[str]) -> List[List[int]]:
    """Generate linked four‑note voicings emulating the pre‑salsa behaviour.

    Each chord is evaluated in several octaves and the bass note chosen is the
    chord tone closest to the previous bass.  Only the first four intervals of
    the chord definition are used so any ninths present in the symbol are
    ignored.
    """

    voicings: List[List[int]] = []
    bajo_anterior = 43  # G2

    for nombre in progresion:
        root, suf = parsear_nombre_acorde(nombre)
        intervalos = INTERVALOS_TRADICIONALES[suf][:4]
        notas_base = [root + i for i in intervalos]

        candidatos: list[tuple[int, int, list[int], int]] = []
        for o in range(1, 5):  # octavas razonables para graves
            acorde = [n + 12 * o for n in notas_base]
            for idx_bajo, n in enumerate(acorde):
                distancia = abs(n - bajo_anterior)
                candidatos.append((distancia, n, acorde, idx_bajo))

        candidatos_comunes = [c for c in candidatos if c[1] == bajo_anterior]
        if candidatos_comunes:
            mejor = min(candidatos_comunes, key=lambda x: x[0])
        else:
            mejor = min(candidatos, key=lambda x: x[0])

        nuevo_bajo = mejor[1]
        acorde = mejor[2]
        idx_bajo = mejor[3]

        resto = acorde[:idx_bajo] + acorde[idx_bajo + 1 :]
        resto.sort()
        voicing = [nuevo_bajo] + resto
        voicings.append(voicing)
        bajo_anterior = nuevo_bajo

    return voicings

# Future voicing strategies for other modes can be added here
