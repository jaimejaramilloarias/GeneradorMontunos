"""Modo de armonía extendida para la aplicación."""

from pathlib import Path
import re
import pretty_midi
from typing import Optional, Tuple, List, Dict

from salsa import SALTO_MAX, _ajustar_rango_flexible
from midi_common import obtener_posiciones_referencia
from midi_utils import (
    _grid_and_bpm,
    _cortar_notas_superpuestas,
    _recortar_notas_a_limite,
)

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
    "7(#9)": [0, 4, 7, 10, 15],                    # 3→15
    "∆9": [0, 4, 7, 11, 14],                       # 2→14
    "∆9(#11)": [0, 4, 7, 11, 14, 18],              # 2,6→14,18
    "∆(#9)#11": [0, 4, 7, 11, 15, 18],             # 3,6→15,18
    "+": [0, 4, 8],
    "∆(#9)": [0, 4, 7, 11, 15],                    # 3→15
    "m11": [0, 3, 7, 10, 14, 17],                  # 2,5→14,17
    "º": [0, 3, 6],
    "m6(9)": [0, 3, 7, 9, 14],                     # 2→14
    "m9(#11)": [0, 3, 7, 10, 14, 18],              # 2,6→14,18
    "sus4": [0, 5, 7],                             # 5→17
    "m9": [0, 3, 7, 10, 14],                       # 2→14
    "m∆11": [0, 3, 7, 11, 14, 17],                 # 2,5→14,17
    "(b5)": [0, 4, 6],
    "m∆9": [0, 3, 7, 11, 14],                      # 2→14
    "m∆#11": [0, 3, 7, 11, 14, 18],                # 2,6→14,18
    "6": [0, 4, 7, 9],
    "+7(b9)": [0, 4, 8, 10, 13],                   # 1→13
    "+7(b9)#11": [0, 4, 8, 10, 13, 18],            # 1,6→13,18
    "13(b5)": [0, 4, 6, 10, 14, 17, 21],           # 2,5,9→14,17,21
    "7": [0, 4, 7, 10],
    "+9": [0, 4, 8, 10, 15],                       # 3→15
    "+9(#11)": [0, 4, 8, 10, 15, 18],              # 3,6→15,18
    "13(b5)#9": [0, 4, 6, 10, 15, 17, 21],         # 3,5,9→15,17,21
    "∆": [0, 4, 7, 11],
    "m6": [0, 3, 7, 9],
    "+∆9": [0, 4, 8, 11, 14],                      # 2→14
    "+∆9(#11)": [0, 4, 8, 11, 14, 18],             # 2,6→14,18
    "m7": [0, 3, 7, 10],
    "+∆(#9)": [0, 4, 8, 11, 15],                   # 3→15
    "+∆(#9)#11": [0, 4, 8, 11, 15, 18],            # 3,6→15,18
    "m∆": [0, 3, 7, 11],
    "º7(9)": [0, 3, 6, 9, 14],                     # 2→14
    "º7(11)": [0, 3, 6, 9, 14, 17],                # 2,5→14,17
    "+7": [0, 4, 8, 10],
    "m7(b5)9": [0, 3, 6, 10, 14],                  # 2→14
    "ø11": [0, 3, 6, 10, 14, 17],                  # 2,5→14,17
    "+∆": [0, 4, 8, 11],
    "º∆9": [0, 3, 6, 11, 14],                      # 2→14
    "º∆11": [0, 3, 6, 11, 14, 17],                 # 2,5→14,17
    "º7": [0, 3, 6, 9],
    "9sus4": [0, 5, 7, 10, 14],                    # 2→14
    "13(b9)": [0, 4, 7, 10, 13, 18, 21],           # 1,6,9→13,18,21
    "m7(b5)": [0, 3, 6, 10],
    "7sus4(b9)": [0, 5, 7, 10, 13],                # 1→13
    "13": [0, 4, 7, 10, 14, 18, 21],               # 2,6,9→14,18,21
    "º∆": [0, 3, 6, 11],
    "∆9sus4": [0, 5, 7, 11, 14],                   # 2→14
    "13(#9)": [0, 4, 7, 10, 15, 18, 21],           # 3,6,9→15,18,21
    "7sus4": [0, 5, 7, 10],
    "7(b5)b9": [0, 4, 6, 10, 13],                  # 1→13
    "7(b9)b13": [0, 4, 7, 10, 13, 18, 20],         # 1,6,8→13,18,20
    "∆sus4": [0, 5, 7, 11],
    "9(b5)": [0, 4, 6, 10, 14],                    # 2→14
    "9(b13)": [0, 4, 7, 10, 14, 18, 20],           # 2,6,8→14,18,20
    "7(b5)": [0, 4, 6, 10],
    "7(b5)#9": [0, 4, 6, 10, 15],                  # 3→15
    "7(#9)b13": [0, 4, 7, 10, 15, 18, 20],         # 3,6,8→15,18,20
    "∆(b5)": [0, 4, 6, 11],
    "6(9)#11": [0, 4, 7, 9, 14, 18],               # 2,6→14,18
    "∆13": [0, 4, 7, 11, 14, 18, 21],              # 2,6,9→14,18,21
    "6(9)": [0, 4, 7, 9, 14],                      # 2→14
    "7(b9)#11": [0, 4, 7, 10, 13, 18],             # 1,6→13,18
    "∆13(#11)": [0, 4, 7, 11, 14, 18, 21],         # 2,6,9→14,18,21
    "7(b9)": [0, 4, 7, 10, 13],                    # 1→13
    "9(#11)": [0, 4, 7, 10, 14, 18],               # 2,6→14,18
    "13(#11)": [0, 4, 7, 10, 14, 18, 21],          # 2,6,9→14,18,21
    "9": [0, 4, 7, 10, 14],                        # 2→14
    "7(#9)#11": [0, 4, 7, 10, 15, 18],             # 3,6→15,18
    "∆13(#9)": [0, 4, 7, 11, 15, 18, 21]           # 3,6,9→15,18,21
}


# Fórmulas de voicing para cada tipo de acorde e inversión
VOICINGS_EXTENDIDA = {
    'triada': {
        'fundamental': [
            ('n1',    'C3'),
            ('n2',    'E3'),
            ('n3',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n3+12', 'G4'),
            ('n1+24', 'C5'),
        ],
        '1a_inv': [
            ('n2',    'E3'),
            ('n3',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n3+12', 'G4'),
            ('n1+24', 'C5'),
            ('n2+24', 'E5'),
        ],
        '2a_inv': [
            ('n3',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n3+12', 'G4'),
            ('n1+24', 'C5'),
            ('n2+24', 'E5'),
            ('n3+24', 'G5'),
        ],
    },
    '7_6': {
        'fundamental': [
            ('n1',    'C3'),
            ('n3',    'E3'),
            ('n4',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n3+12', 'G4'),
            ('n1+24', 'C5'),
        ],
        '1a_inv': [
            ('n2',    'E3'),
            ('n4',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n3+12', 'G4'),
            ('n4+12', 'C5'),
            ('n2+24', 'E5'),
        ],
        '2a_inv': [
            ('n3',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n3+12', 'G4'),
            ('n4+12', 'C5'),
            ('n2+24', 'E5'),
            ('n3+24', 'G5'),
        ],
    },
    '9': {
        'fundamental': [
            ('n1',    'C3'),
            ('n3',    'E3'),
            ('n4',    'G3'),
            ('n2+12', 'C4'),
            ('n3+12', 'E4'),
            ('n4+12', 'G4'),
            ('n5+12', 'C5'),
        ],
        '1a_inv': [
            ('n2',    'E3'),
            ('n1+12', 'G3'),
            ('n5',    'C4'),
            ('n2+12', 'E4'),
            ('n3+12', 'G4'),
            ('n4+12', 'C5'),
            ('n5+12', 'E5'),
        ],
        '2a_inv': [
            ('n3',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n3+12', 'G4'),
            ('n4+12', 'C5'),
            ('n5+12', 'E5'),
            ('n3+24', 'G5'),
        ],
    },
    '11': {
        'fundamental': [
            ('n1',    'C3'),
            ('n3',    'E3'),
            ('n4',    'G3'),
            ('n2+12', 'C4'),
            ('n6',    'E4'),
            ('n4+12', 'G4'),
            ('n5+12', 'C5'),
        ],
        '1a_inv': [
            ('n2',    'E3'),
            ('n1+12', 'G3'),
            ('n5',    'C4'),
            ('n6',    'E4'),
            ('n4+12', 'G4'),
            ('n5+12', 'C5'),
            ('n6+12', 'E5'),
        ],
        '2a_inv': [
            ('n3',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n4+12', 'G4'),
            ('n5+12', 'C5'),
            ('n6+12', 'E5'),
            ('n4+24', 'G5'),
        ],
    },
    '13': {
        'fundamental': [
            ('n1',    'C3'),
            ('n3',    'E3'),
            ('n4',    'G3'),
            ('n5',    'C4'),
            ('n3+12', 'E4'),
            ('n7',    'G4'),
            ('n5+12', 'C5'),
        ],
        '1a_inv': [
            ('n2',    'E3'),
            ('n3',    'G3'),
            ('n1+12', 'C4'),
            ('n5',    'E4'),
            ('n6',    'G4'),
            ('n7',    'C5'),
            ('n5+12', 'E5'),
        ],
        '2a_inv': [
            ('n3',    'G3'),
            ('n1+12', 'C4'),
            ('n2+12', 'E4'),
            ('n4+12', 'G4'),
            ('n5+12', 'C5'),
            ('n6',    'E5'),
            ('n7',    'G5'),
        ],
    },
}

def get_midi_numbers(base_note, intervalos, voicing_pattern, octava_base=4):
    """Convierte ``voicing_pattern`` en notas MIDI absolutas.

    ``intervalos`` debe contener las distancias desde la fundamental que
    definen el acorde (por ejemplo ``[0, 3, 7, 10]`` para ``m7``).  Los
    elementos de ``voicing_pattern`` indican qué intervalo usar (``n1``,
    ``n2``, ...), opcionalmente desplazado por octavas con ``+12``.
    ``base_note`` puede ser la nota en texto (``"C"``) o la clase de
    tono MIDI como entero.
    """

    if isinstance(base_note, str):
        base_root = NOTA_A_MIDI[base_note]
    else:
        base_root = int(base_note) % 12
    base_root += 12 * octava_base
    resultado = []
    for token, _ in voicing_pattern:
        m = re.match(r"n(\d+)(?:\+(\d+))?", token)
        if not m:
            raise ValueError(f"Patrón inválido: {token}")
        idx = int(m.group(1)) - 1
        if idx >= len(intervalos):
            raise ValueError(
                f"La plantilla requiere n{idx + 1} pero el acorde solo tiene {len(intervalos)} intervalos"
            )
        extra = int(m.group(2) or 0)
        midi = base_root + intervalos[idx] + extra
        resultado.append(midi)
    return resultado


def get_midi_numbers_debug(base_note, intervalos, voicing_pattern, octava_base=4):
    """Como :func:`get_midi_numbers` pero conserva la información de depuración."""

    if isinstance(base_note, str):
        base_root = NOTA_A_MIDI[base_note]
    else:
        base_root = int(base_note) % 12
    base_root += 12 * octava_base
    resultado = []
    debug = []
    for token, ref in voicing_pattern:
        m = re.match(r"n(\d+)(?:\+(\d+))?", token)
        if not m:
            raise ValueError(f"Patrón inválido: {token}")
        idx = int(m.group(1)) - 1
        extra = int(m.group(2) or 0)
        if idx >= len(intervalos):
            raise ValueError(
                f"La plantilla requiere n{idx + 1} pero el acorde solo tiene {len(intervalos)} intervalos"
            )
        midi = base_root + intervalos[idx] + extra
        resultado.append(midi)
        debug.append((ref, token, midi))
    return resultado, debug


def parsear_nombre_acorde(nombre: str) -> tuple[int, str]:
    """Devuelve ``(root_pc, sufijo)`` a partir de ``nombre``.

    La notación ``/3`` o ``/5`` al final para forzar una inversión no forma
    parte del sufijo, por lo que se elimina antes de validar.
    """

    base = re.sub(r"/[1357]$", "", nombre)

    m = re.match(r"^([A-G](?:b|#)?)(.*)$", base)
    if not m:
        raise ValueError(f"Acorde no reconocido: {nombre}")
    root_str, suf = m.groups()
    suf = suf or ""
    if suf not in DICCIONARIO_EXTENDIDA:
        raise ValueError(f"Sufijo desconocido: {suf}")
    return NOTA_A_MIDI[root_str], suf


def get_bass_pitch_ext(cifrado: str, inversion: str) -> int:
    """Return the MIDI pitch for the bass note of ``cifrado``."""

    root_pc, suf = parsear_nombre_acorde(cifrado)
    ints = DICCIONARIO_EXTENDIDA[suf]
    if inversion == "root":
        return root_pc + 12 * 3
    elif inversion == "third":
        return (root_pc + ints[1]) % 12 + 12 * 3
    elif inversion == "fifth":
        return (root_pc + ints[2]) % 12 + 12 * 3
    else:
        raise ValueError(f"Inversión desconocida: {inversion}")


def seleccionar_inversion_ext(anterior: Optional[int], cifrado: str) -> Tuple[str, int]:
    """Choose the inversion with the smallest leap from ``anterior``."""

    mejores: List[Tuple[int, str, int]] = []
    for inv in ["root", "third", "fifth"]:
        pitch = get_bass_pitch_ext(cifrado, inv)
        pitch = _ajustar_rango_flexible(anterior, pitch)
        distancia = 0 if anterior is None else abs(pitch - anterior)
        mejores.append((distancia, inv, pitch))

    mejores.sort()
    mejor = mejores[0]
    if anterior is not None and mejor[2] == anterior:
        for opcion in mejores[1:]:
            dist, inv, pitch = opcion
            if dist <= SALTO_MAX and pitch != anterior:
                return inv, pitch
    return mejor[1], mejor[2]


def _extraer_grupos(posiciones_base: List[dict], total_cor_ref: int, grid_seg: float) -> List[List[dict]]:
    """Agrupa ``posiciones_base`` por corchea."""

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

    return grupos_ref


def _cargar_grupos_por_inversion_ext(
    plantillas: dict,
) -> Tuple[dict, List[int], Dict[str, int], int, float, float]:
    """Devuelve notas agrupadas por corchea para cada inversión."""

    grupos_por_inv: Dict[str, List[List[Dict]]] = {}
    total_cor_ref = None
    grid = bpm = None
    notas_base_set: set[int] = set()
    offsets: Dict[str, int] = {}
    base_root: List[int] | None = None
    for inv, pm in plantillas.items():
        cor_ref, g, b = _grid_and_bpm(pm)
        if grid is None:
            grid = g
            bpm = b
            total_cor_ref = cor_ref
        notes = pm.instruments[0].notes
        posiciones_base, base = obtener_posiciones_referencia(notes)
        notas_base_set.update(base)
        if base_root is None:
            base_root = base
            offsets[inv] = 0
        else:
            offsets[inv] = base[0] - base_root[0]
        grupos_por_inv[inv] = _extraer_grupos(posiciones_base, cor_ref, grid)

    return grupos_por_inv, sorted(notas_base_set), offsets, total_cor_ref, grid, bpm


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
    inversiones_manual: list[tuple[str, int]] | None = None,
    return_pm: bool = False,
    variante: str = "A",
    asignaciones_custom: list[tuple[str, list[int], str | None]] | None = None,
    voicing_offsets: list[int] | None = None,
) -> pretty_midi.PrettyMIDI | None:
    """Genera montuno usando las reglas de armonía extendida."""

    from salsa import _ajustar_rango_flexible
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
        offsets: list[int] = []
        voz_grave_anterior = None
        for idx, (cifrado, _, inv_forzado) in enumerate(asignaciones):
            if idx == 0:
                inv = inv_forzado or inversion_inicial
                pitch = get_bass_pitch_ext(cifrado, inv)
                pitch = _ajustar_rango_flexible(voz_grave_anterior, pitch)
            else:
                if inv_forzado:
                    inv = inv_forzado
                    pitch = get_bass_pitch_ext(cifrado, inv)
                    pitch = _ajustar_rango_flexible(voz_grave_anterior, pitch)
                else:
                    inv, pitch = seleccionar_inversion_ext(voz_grave_anterior, cifrado)
            inversiones.append(inv)
            offsets.append(0)
            voz_grave_anterior = pitch
    else:
        inversiones = [inv for inv, _ in inversiones_manual]
        offsets = [off for _, off in inversiones_manual]

    inv_map = {"root": "fundamental", "third": "1a_inv", "fifth": "2a_inv"}
    tipo_map = {3: "triada", 4: "7_6", 5: "9", 6: "11", 7: "13"}

    voicings = []
    offs = []
    asign_simple = []
    for idx_i, ((nombre, idxs, _), inv) in enumerate(zip(asignaciones, inversiones)):
        root, suf = parsear_nombre_acorde(nombre)
        intervalos = DICCIONARIO_EXTENDIDA[suf]
        tipo = tipo_map[len(intervalos)]
        pattern = VOICINGS_EXTENDIDA[tipo][inv_map[inv]]
        voicings.append(get_midi_numbers(root, intervalos, pattern))
        offs.append(offsets[idx_i] * 12)
        asign_simple.append((nombre, idxs, ""))

    if voicing_offsets is not None:
        offs = voicing_offsets

    # --------------------------------------------------------------
    # Carga de las plantillas por inversión y construcción de grupos
    # --------------------------------------------------------------
    parts = midi_ref.stem.split("_")
    base = "_".join(parts[:2]) if len(parts) >= 2 else midi_ref.stem
    plantillas = {}
    for inv in ["root", "third", "fifth"]:
        path = midi_ref.parent / f"{base}_{inv}_2chords.mid"
        plantillas[inv] = pretty_midi.PrettyMIDI(str(path))

    grupos_por_inv, notas_base, offsets_inv, total_ref_cor, grid, bpm = _cargar_grupos_por_inversion_ext(plantillas)

    total_dest_cor = max(i for _, idxs, _ in asignaciones for i in idxs) + 1

    mapa: Dict[int, int] = {}
    limites: Dict[int, int] = {}
    for i, (_, idxs, _) in enumerate(asignaciones):
        for ix in idxs:
            mapa[ix] = i
        limites[i] = idxs[-1] + 1

    inv_por_cor: Dict[int, str] = {}
    for idx, (_, idxs, _) in enumerate(asignaciones):
        for ix in idxs:
            inv_por_cor[ix] = inversiones[idx]

    posiciones: List[dict] = []
    for cor in range(total_dest_cor):
        inv = inv_por_cor.get(cor)
        if inv is None:
            continue
        ref_idx = (inicio_cor + cor) % total_ref_cor
        idx_acorde = mapa[cor]
        limite_cor = limites[idx_acorde]
        for pos in grupos_por_inv[inv][ref_idx]:
            inicio = cor * grid + pos["start"]
            fin = cor * grid + pos["end"]
            end = min(fin, limite_cor * grid)
            if end <= inicio:
                continue
            posiciones.append(
                {
                    "pitch": pos["pitch"] - offsets_inv.get(inv, 0),
                    "start": inicio,
                    "end": end,
                    "velocity": pos["velocity"],
                }
            )

    if offs:
        voicings = [
            [p + offs[i] for p in v] for i, v in enumerate(voicings)
        ]

    notas_finales = midi_utils.generar_notas_mixtas(
        posiciones,
        voicings,
        asign_simple,
        grid,
        notas_base=notas_base,
        parse_fn=parsear_nombre_acorde,
        interval_dict=DICCIONARIO_EXTENDIDA,
    )

    limite = total_dest_cor * grid
    notas_finales = _cortar_notas_superpuestas(notas_finales)
    notas_finales = _recortar_notas_a_limite(notas_finales, limite)
    if limite > 0:
        has_start = any(n.start <= 0 < n.end and n.pitch > 0 for n in notas_finales)
        has_end = any(
            n.pitch > 0 and n.start < limite and n.end > limite - grid for n in notas_finales
        )
        if not has_start:
            notas_finales.append(
                pretty_midi.Note(
                    velocity=1,
                    pitch=0,
                    start=0.0,
                    end=min(grid, limite),
                )
            )
        if not has_end:
            notas_finales.append(
                pretty_midi.Note(
                    velocity=1,
                    pitch=0,
                    start=max(0.0, limite - grid),
                    end=limite,
                )
            )

    pm_out = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(
        program=plantillas["root"].instruments[0].program,
        is_drum=plantillas["root"].instruments[0].is_drum,
        name=plantillas["root"].instruments[0].name,
    )
    inst.notes = notas_finales
    pm_out.instruments.append(inst)
    if return_pm:
        return pm_out

    pm_out.write(str(output))
    return None
