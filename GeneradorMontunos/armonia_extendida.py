"""Modo de armonía extendida para la aplicación."""

from pathlib import Path
import re
import pretty_midi

# Mapeo de nombres de nota a su clase de tono MIDI
NOTA_A_MIDI = {
    'C': 0,
    'C#': 1,
    'Db': 1,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'Gb': 6,
    'G': 7,
    'G#': 8,
    'Ab': 8,
    'A': 9,
    'A#': 10,
    'Bb': 10,
    'B': 11,
    'Cb': 11,
    'B#': 0,
}

# Diccionario completo de intervalos relativos a la fundamental
DICCIONARIO_EXTENDIDA = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7(#9)": [0, 4, 7, 10, 3],
    "∆9": [0, 4, 7, 11, 2],
    "∆9(#11)": [0, 4, 7, 11, 2, 6],
    "∆(#9)#11": [0, 4, 7, 11, 3, 6],
    "+": [0, 4, 8],
    "∆(#9)": [0, 4, 7, 11, 3],
    "m11": [0, 3, 7, 10, 2, 5],
    "º": [0, 3, 6],
    "m6(9)": [0, 3, 7, 9, 2],
    "m9(#11)": [0, 3, 7, 10, 2, 6],
    "sus4": [0, 5, 7],
    "m9": [0, 3, 7, 10, 2],
    "m∆11": [0, 3, 7, 11, 2, 5],
    "(b5)": [0, 4, 6],
    "m∆9": [0, 3, 7, 11, 2],
    "m∆#11": [0, 3, 7, 11, 2, 6],
    "6": [0, 4, 7, 9],
    "+7(b9)": [0, 4, 8, 10, 1],
    "+7(b9)#11": [0, 4, 8, 10, 1, 6],
    "13(b5)": [0, 4, 6, 10, 2, 5, 9],
    "7": [0, 4, 7, 10],
    "+9": [0, 4, 8, 10, 3],
    "+9(#11)": [0, 4, 8, 10, 3, 6],
    "13(b5)#9": [0, 4, 6, 10, 3, 5, 9],
    "∆": [0, 4, 7, 11],
    "m6": [0, 3, 7, 9],
    "+∆9": [0, 4, 8, 11, 2],
    "+∆9(#11)": [0, 4, 8, 11, 2, 6],
    "m7": [0, 3, 7, 10],
    "+∆(#9)": [0, 4, 8, 11, 3],
    "+∆(#9)#11": [0, 4, 8, 11, 3, 6],
    "m∆": [0, 3, 7, 11],
    "º7(9)": [0, 3, 6, 9, 2],
    "º7(11)": [0, 3, 6, 9, 2, 5],
    "+7": [0, 4, 8, 10],
    "m7(b5)9": [0, 3, 6, 10, 2],
    "ø11": [0, 3, 6, 10, 2, 5],
    "+∆": [0, 4, 8, 11],
    "º∆9": [0, 3, 6, 11, 2],
    "º∆11": [0, 3, 6, 11, 2, 5],
    "º7": [0, 3, 6, 9],
    "9sus4": [0, 5, 7, 10, 2],
    "13(b9)": [0, 4, 7, 10, 1, 6, 9],
    "m7(b5)": [0, 3, 6, 10],
    "7sus4(b9)": [0, 5, 7, 10, 1],
    "13": [0, 4, 7, 10, 2, 6, 9],
    "º∆": [0, 3, 6, 11],
    "∆9sus4": [0, 5, 7, 11, 2],
    "13(#9)": [0, 4, 7, 10, 3, 6, 9],
    "7sus4": [0, 5, 7, 10],
    "7(b5)b9": [0, 4, 6, 10, 1],
    "7(b9)b13": [0, 4, 7, 10, 1, 6, 8],
    "∆sus4": [0, 5, 7, 11],
    "9(b5)": [0, 4, 6, 10, 2],
    "9(b13)": [0, 4, 7, 10, 2, 6, 8],
    "7(b5)": [0, 4, 6, 10],
    "7(b5)#9": [0, 4, 6, 10, 3],
    "7(#9)b13": [0, 4, 7, 10, 3, 6, 8],
    "∆(b5)": [0, 4, 6, 11],
    "6(9)#11": [0, 4, 7, 9, 2, 6],
    "∆13": [0, 4, 7, 11, 2, 6, 9],
    "6(9)": [0, 4, 7, 9, 2],
    "7(b9)#11": [0, 4, 7, 10, 1, 6],
    "∆13(#11)": [0, 4, 7, 11, 2, 6, 9],
    "7(b9)": [0, 4, 7, 10, 1],
    "9(#11)": [0, 4, 7, 10, 2, 6],
    "13(#11)": [0, 4, 7, 10, 2, 6, 9],
    "9": [0, 4, 7, 10, 2],
    "7(#9)#11": [0, 4, 7, 10, 3, 6],
    "∆13(#9)": [0, 4, 7, 11, 3, 6, 9]
}

# Fórmulas de voicing para cada tipo de acorde e inversión
VOICINGS_EXTENDIDA = {
    "triada": {
        "fundamental": [0, 1, 2, 12, 13, 14, 24],
        "1a_inv": [1, 2, 12, 13, 14, 24, 25],
        "2a_inv": [2, 12, 13, 14, 24, 25, 26],
    },
    "7": {
        "fundamental": [0, 2, 3, 12, 13, 14, 24],
        "1a_inv": [1, 3, 12, 13, 14, 15, 25],
        "2a_inv": [2, 12, 13, 14, 15, 25, 26],
    },
    "9": {
        "fundamental": [0, 2, 3, 13, 14, 15, 16],
        "1a_inv": [1, 12, 4, 13, 14, 15, 16],
        "2a_inv": [2, 12, 13, 14, 15, 16, 26],
    },
    "11": {
        "fundamental": [0, 2, 3, 13, 5, 15, 16],
        "1a_inv": [1, 12, 4, 5, 15, 16, 17],
        "2a_inv": [2, 12, 13, 15, 16, 17, 27],
    },
    "13": {
        "fundamental": [0, 2, 3, 4, 14, 6, 16],
        "1a_inv": [1, 2, 12, 4, 5, 6, 16],
        "2a_inv": [2, 12, 13, 15, 16, 5, 6],
    },
}

def get_midi_numbers(base_note, intervalos, voicing_pattern, octava_base=3):
    base_midi = NOTA_A_MIDI[base_note] + 12 * octava_base
    res = []
    for v in voicing_pattern:
        idx = v % 12  # index en la lista de intervalos (ajusta si tienes más de 12 notas, pero para 7 siempre vale)
        octave = v // 12
        semitonos = intervalos[idx] + 12 * octave
        midi_nota = base_midi + semitonos
        res.append(midi_nota)
    return res


def parsear_nombre_acorde(nombre: str) -> tuple[str, str]:
    """Devuelve ``(root, sufijo)`` a partir de ``nombre``."""
    m = re.match(r"^([A-G](?:b|#)?)(.*)$", nombre)
    if not m:
        raise ValueError(f"Acorde no reconocido: {nombre}")
    root, suf = m.groups()
    suf = suf or ""
    if suf not in DICCIONARIO_EXTENDIDA:
        raise ValueError(f"Sufijo desconocido: {suf}")
    return root, suf


def procesar_progresion(texto: str, *, inicio_cor: int = 0):
    """Analiza la progresión y asigna corcheas a cada acorde."""

    from salsa import _siguiente_grupo, _indice_para_corchea

    import re

    segmentos_raw = [s.strip() for s in texto.split("|") if s.strip()]
    segmentos: list[str] = []
    for seg in segmentos_raw:
        if seg == "%":
            if not segmentos:
                raise ValueError("% no puede ir en el primer compás")
            segmentos.append(segmentos[-1])
        else:
            segmentos.append(seg)

    resultado: list[tuple[str, list[int], str | None]] = []
    indice_patron = _indice_para_corchea(inicio_cor)
    posicion = 0

    for seg in segmentos:
        tokens = [t for t in seg.split() if t]
        acordes: list[tuple[str, str | None]] = []
        for tok in tokens:
            m = re.match(r"^(.*)/([135])$", tok)
            inv = None
            if m:
                tok, codigo = m.groups()
                inv_map = {"1": "root", "3": "third", "5": "fifth"}
                inv = inv_map[codigo]
            acordes.append((tok, inv))

        if len(acordes) == 1:
            g1 = _siguiente_grupo(indice_patron)
            g2 = _siguiente_grupo(indice_patron + 1)
            dur = g1 + g2
            indices = list(range(posicion, posicion + dur))
            nombre, inv = acordes[0]
            resultado.append((nombre, indices, inv))
            posicion += dur
            indice_patron += 2
        elif len(acordes) == 2:
            g1 = _siguiente_grupo(indice_patron)
            indices1 = list(range(posicion, posicion + g1))
            posicion += g1
            indice_patron += 1

            g2 = _siguiente_grupo(indice_patron)
            indices2 = list(range(posicion, posicion + g2))
            posicion += g2
            indice_patron += 1

            (n1, i1), (n2, i2) = acordes
            resultado.append((n1, indices1, i1))
            resultado.append((n2, indices2, i2))
        elif len(acordes) == 0:
            continue
        else:
            raise ValueError(f"Cada segmento debe contener uno o dos acordes: {seg}")

    return resultado, len(segmentos)

def montuno_armonia_extendida(
    progresion_texto: str,
    midi_ref: Path,
    output: Path,
    inversion_inicial: str = "root",
    *,
    inicio_cor: int = 0,
    inversiones_manual: list[str] | None = None,
    return_pm: bool = False,
    variante: str = "A",
    asignaciones_custom: list[tuple[str, list[int], str | None]] | None = None,
) -> pretty_midi.PrettyMIDI | None:
    """Genera montuno usando las reglas de armonía extendida."""

    from salsa import (
        seleccionar_inversion,
        get_bass_pitch,
        _ajustar_rango_flexible,
    )
    import midi_utils

    if asignaciones_custom is None:
        asignaciones, compases = procesar_progresion(
            progresion_texto, inicio_cor=inicio_cor
        )
    else:
        asignaciones = asignaciones_custom
        compases = (
            (max(i for _, idxs, *_ in asignaciones for i in idxs) + 7) // 8
            if asignaciones
            else 0
        )

    if inversiones_manual is None:
        inversiones = []
        voz_grave_anterior = None
        for idx, (cifrado, _, inv_forzado) in enumerate(asignaciones):
            if idx == 0:
                inv = inv_forzado or inversion_inicial
                pitch = get_bass_pitch(cifrado, inv)
                pitch = _ajustar_rango_flexible(voz_grave_anterior, pitch)
            else:
                if inv_forzado:
                    inv = inv_forzado
                    pitch = get_bass_pitch(cifrado, inv)
                    pitch = _ajustar_rango_flexible(voz_grave_anterior, pitch)
                else:
                    inv, pitch = seleccionar_inversion(voz_grave_anterior, cifrado)
            inversiones.append(inv)
            voz_grave_anterior = pitch
    else:
        inversiones = inversiones_manual

    inv_map = {"root": "fundamental", "third": "1a_inv", "fifth": "2a_inv"}
    tipo_map = {3: "triada", 4: "7", 5: "9", 6: "11", 7: "13"}

    voicings = []
    asign_simple = []
    for (nombre, idxs, _), inv in zip(asignaciones, inversiones):
        root, suf = parsear_nombre_acorde(nombre)
        intervalos = DICCIONARIO_EXTENDIDA[suf]
        tipo = tipo_map[len(intervalos)]
        pattern = VOICINGS_EXTENDIDA[tipo][inv_map[inv]]
        voicings.append(get_midi_numbers(root, intervalos, pattern))
        asign_simple.append((nombre, idxs, ""))

    return midi_utils.exportar_montuno(
        midi_ref,
        voicings,
        asign_simple,
        compases,
        output,
        inicio_cor=inicio_cor,
        return_pm=return_pm,
    )
