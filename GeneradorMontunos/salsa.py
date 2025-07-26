# -*- coding: utf-8 -*-
# salsa.py
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pretty_midi

from voicings import parsear_nombre_acorde, INTERVALOS_TRADICIONALES
from midi_utils import (
    _grid_and_bpm,
    procesar_progresion_en_grupos,
    _cortar_notas_superpuestas,
    _recortar_notas_a_limite,
    _siguiente_grupo,
)

# ========================

# ========================
# Inversiones disponibles
# ========================
INVERSIONS = ["root", "third", "fifth"]

# Notas que funcionan como aproximaciones en las plantillas de salsa.  Si el
# acorde cambia justo al inicio de la figura se ajustan al sonido estructural
# más cercano.
APPROACH_NOTES = {"D", "A", "B", "D#", "F", "G#", "C#"}

# Switch to enable adjusting approach notes to structural tones when a chord
# change occurs at the beginning of the pattern.  Set to ``True`` to keep the
# current behaviour.  When ``False``, approach notes remain unchanged.
CONVERTIR_APROX_A_ESTRUCT = False


def _ajustar_a_estructural_mas_cercano(note_name: str, cifrado: str, pitch: int) -> int:
    """Devuelve la fundamental, tercera o quinta más cercana a ``pitch``."""

    root, suf = parsear_nombre_acorde(cifrado)
    ints = INTERVALOS_TRADICIONALES[suf]
    octave = int(note_name[-1])

    def midi(interval: int) -> int:
        return root + interval + 12 * (octave + 1)

    opc1 = midi(0)
    opc2 = midi(5 if "sus" in suf else ints[1])
    opc3 = midi(ints[2])
    candidatos = [opc1, opc2, opc3]
    return min(candidatos, key=lambda p: abs(p - pitch))


# ========================
# Función para elegir la mejor inversión para cada acorde
# ========================

RANGO_BAJO_MIN = 48  # C3
RANGO_BAJO_MAX = 67  # G4
RANGO_EXTRA = 4  # flexible extension
SALTO_MAX = 8  # minor sixth in semitones


def _ajustar_rango(pitch: int) -> int:
    """Confine ``pitch`` within ``RANGO_BAJO_MIN`` .. ``RANGO_BAJO_MAX``."""

    while pitch < RANGO_BAJO_MIN:
        pitch += 12
    while pitch > RANGO_BAJO_MAX:
        pitch -= 12
    return pitch


def _ajustar_rango_ext(pitch: int) -> int:
    """Confine ``pitch`` within the extended range."""

    while pitch < RANGO_BAJO_MIN - RANGO_EXTRA:
        pitch += 12
    while pitch > RANGO_BAJO_MAX + RANGO_EXTRA:
        pitch -= 12
    return pitch


def _ajustar_rango_flexible(prev_pitch: Optional[int], pitch: int) -> int:
    """Prefer the fixed range but allow a small extension for smoother leaps."""

    base = _ajustar_rango(pitch)
    base = _ajustar_salto(prev_pitch, base)
    base = _ajustar_rango(base)
    base = _ajustar_salto(prev_pitch, base)
    if prev_pitch is None:
        return base
    dist_base = abs(base - prev_pitch)
    if dist_base <= SALTO_MAX:
        return base

    ext = _ajustar_rango_ext(pitch)
    ext = _ajustar_salto(prev_pitch, ext)
    ext = _ajustar_rango_ext(ext)
    ext = _ajustar_salto(prev_pitch, ext)
    return ext


def _ajustar_salto(prev_pitch: Optional[int], pitch: int) -> int:
    """Shift ``pitch`` by octaves so the leap from ``prev_pitch`` is <= ``SALTO_MAX``."""

    if prev_pitch is None:
        return pitch
    while pitch - prev_pitch > SALTO_MAX:
        pitch -= 12
    while prev_pitch - pitch > SALTO_MAX:
        pitch += 12
    return pitch


def get_bass_pitch(cifrado: str, inversion: str) -> int:
    """Devuelve la nota MIDI de la voz grave para el acorde e inversión dada."""
    root, suf = parsear_nombre_acorde(cifrado)
    ints = INTERVALOS_TRADICIONALES[suf]
    if inversion == "root":
        return root + 12 * 3  # C3 por default
    elif inversion == "third":
        return (root + ints[1]) % 12 + 12 * 3  # Tercera en C3, E3, etc.
    elif inversion == "fifth":
        return (root + ints[2]) % 12 + 12 * 3  # Quinta en G3, etc.
    else:
        raise ValueError(f"Inversión desconocida: {inversion}")


def seleccionar_inversion(anterior: Optional[int], cifrado: str) -> tuple[str, int]:
    """Selecciona la inversión que genere el salto más corto (<= SALTO_MAX)."""

    mejores: list[tuple[int, str, int]] = []
    for inv in INVERSIONS:
        pitch = get_bass_pitch(cifrado, inv)
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


# ========================
# Traducción de notas plantilla → acorde cifrado
# ========================


def traducir_nota(note_name: str, cifrado: str) -> Tuple[int, bool]:
    """Traduce ``note_name`` según las reglas del modo salsa.

    Devuelve el ``pitch`` calculado y un flag indicando si la nota es de
    aproximación.
    """

    root, suf = parsear_nombre_acorde(cifrado)
    ints = INTERVALOS_TRADICIONALES[suf]

    is_minor = ints[1] - ints[0] == 3 or "m" in suf or "m7(b5)" in suf or "º" in suf
    has_b9 = "b9" in cifrado
    has_b13 = "b13" in cifrado
    has_b5 = "b5" in cifrado
    extra_b6 = "(b6)" in cifrado
    extra_b13 = "(b13)" in cifrado

    name = note_name[:-1]
    octave = int(note_name[-1])

    def midi(interval: int) -> int:
        return root + interval + 12 * (octave + 1)

    interval = None
    es_aprox = False

    if name == "C":
        interval = 0
    elif name == "E":
        interval = 5 if "sus" in suf else ints[1]
    elif name == "G":
        interval = ints[2]
    elif name == "D":
        if has_b5:
            interval = 1
        else:
            interval = 1 if has_b9 else 2
        es_aprox = True
    elif name == "A":
        if has_b9 or has_b13 or has_b5 or extra_b6 or extra_b13:
            interval = 8
        else:
            interval = 9
        es_aprox = True
    elif name == "B":
        if suf.endswith("6") and "7" not in suf:
            interval = 11
        else:
            interval = ints[3] if len(ints) > 3 else 11
        es_aprox = True
    elif name == "D#":
        third_int = 3 if is_minor else 4
        interval = third_int - 1
        es_aprox = True
    elif name == "F":
        interval = 5
        es_aprox = True
    elif name == "G#":
        interval = ints[2] - 1
        es_aprox = True
    elif name == "C#":
        interval = 11 if has_b9 else 1
        es_aprox = True
    else:
        return pretty_midi.note_name_to_number(note_name), False

    return midi(interval), es_aprox


def _extraer_grupos_con_nombres(
    posiciones_base: List[dict], total_cor_ref: int, grid_seg: float
) -> List[List[dict]]:
    """Agrupa ``posiciones_base`` por corchea conservando el nombre."""

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
                    "name": pos["name"],
                }
            )

    return grupos_ref


def _cargar_grupos_por_inversion(
    plantillas: Dict[str, pretty_midi.PrettyMIDI],
) -> tuple[dict[str, List[List[dict]]], int, float, float]:
    """Devuelve notas agrupadas por corchea para cada inversión."""

    grupos_por_inv: dict[str, List[List[dict]]] = {}
    total_cor_ref = None
    grid = bpm = None
    for inv, pm in plantillas.items():
        cor_ref, g, b = _grid_and_bpm(pm)
        if grid is None:
            grid = g
            bpm = b
            total_cor_ref = cor_ref
        posiciones_base: List[dict] = []
        for n in pm.instruments[0].notes:
            posiciones_base.append(
                {
                    "pitch": int(n.pitch),
                    "start": n.start,
                    "end": n.end,
                    "velocity": n.velocity,
                    "name": pretty_midi.note_number_to_name(int(n.pitch)),
                }
            )
        grupos_por_inv[inv] = _extraer_grupos_con_nombres(
            posiciones_base, cor_ref, grid
        )
    return grupos_por_inv, total_cor_ref, grid, bpm


def _indice_para_corchea(cor: int) -> int:
    idx = 0
    pos = 0
    while pos < cor:
        pos += _siguiente_grupo(idx)
        idx += 1
    return idx


def procesar_progresion_salsa(
    texto: str,
    armonizacion_default: str | None = None,
    *,
    inicio_cor: int = 0,
) -> Tuple[List[Tuple[str, List[int], str, Optional[str]]], int]:
    """Procesa la progresión reconociendo extensiones específicas de salsa."""

    import re

    segmentos_raw = [s.strip() for s in texto.split("|") if s.strip()]

    # Expand symbol '%' to repeat the previous measure
    segmentos: List[str] = []
    for seg in segmentos_raw:
        if seg == "%":
            if not segmentos:
                raise ValueError("% no puede ir en el primer comp\u00e1s")
            segmentos.append(segmentos[-1])
        else:
            segmentos.append(seg)

    num_compases = len(segmentos)

    resultado: List[Tuple[str, List[int], str, Optional[str]]] = []
    indice_patron = _indice_para_corchea(inicio_cor)
    posicion = 0
    arm_actual = (armonizacion_default or "").capitalize()
    inv_forzado: Optional[str] = None

    def procesar_token(token: str) -> tuple[str | None, str | None]:
        """Return ``(chord, inversion)`` parsed from ``token``.

        The global ``arm_actual`` is updated if the token contains a
        harmonisation marker.  ``inversion`` may be ``None`` if no forced
        inversion was found.
        """

        nonlocal arm_actual
        inversion: Optional[str] = None

        arm_map = {
            "8": "Octavas",
            "15": "Doble octava",
            "10": "D\u00e9cimas",
            "13": "Treceavas",
        }

        while True:
            # Strip optional mode/style token (e.g. ``[TRAD]``)
            m = re.match(r"^\[[A-Z]+\](.*)$", token)
            if m:
                token = m.group(1)
                if not token:
                    return None, inversion
                continue

            m = re.match(r"^\((8|10|13|15)\)(.*)$", token)
            if m:
                codigo, token = m.groups()
                arm_actual = arm_map[codigo]
                continue
            break

        m = re.match(r"^(.*)/([1357])$", token)
        if m:
            token, codigo = m.groups()
            inv_map = {"1": "root", "3": "third", "5": "fifth", "7": "seventh"}
            inversion = inv_map[codigo]

        if not token:
            return None, inversion

        return token, inversion

    for seg in segmentos:
        tokens = [t for t in seg.split() if t]
        acordes: list[tuple[str, str, Optional[str]]] = []
        for tok in tokens:
            nombre, inv_local = procesar_token(tok)
            if nombre is None:
                if inv_local is not None:
                    inv_forzado = inv_local
                continue
            acordes.append((nombre, arm_actual, inv_local or inv_forzado))
            inv_forzado = None
        if len(acordes) == 1:
            g1 = _siguiente_grupo(indice_patron)
            g2 = _siguiente_grupo(indice_patron + 1)
            dur = g1 + g2
            indices = list(range(posicion, posicion + dur))
            nombre, arm, inv = acordes[0]
            resultado.append((nombre, indices, arm, inv))
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

            (n1, a1, i1), (n2, a2, i2) = acordes
            resultado.append((n1, indices1, a1, i1))
            resultado.append((n2, indices2, a2, i2))
        elif len(acordes) == 0:
            continue
        else:
            raise ValueError("Cada segmento debe contener uno o dos acordes: " f"{seg}")

    return resultado, num_compases


# ========================
# Función principal para el modo salsa
# ========================


def montuno_salsa(
    progresion_texto: str,
    midi_ref: Path,
    output: Path,
    inversion_inicial: str = "root",
    *,
    inicio_cor: int = 0,
    inversiones_manual: list[str] | None = None,
    return_pm: bool = False,
    variante: str = "A",   # <-- NUEVO parámetro
) -> pretty_midi.PrettyMIDI | None:
    """Genera montuno estilo salsa enlazando acordes e inversiones.

    ``inversion_inicial`` determina la posición del primer acorde y guía el
    enlace de los siguientes. ``inicio_cor`` indica la corchea global donde
    comienza este segmento para que la plantilla se alinee siempre con la
    progresión completa.
    """
    # Procesa la progresión. Cada compás puede contener uno o dos acordes
    print("[DEBUG] Texto que llega a procesar_progresion_salsa (Salsa):", repr(progresion_texto))
    asignaciones, compases = procesar_progresion_salsa(
        progresion_texto, inicio_cor=inicio_cor
    )

    # --------------------------------------------------------------
    # Selección de la inversión para cada acorde enlazando la voz grave
    # o usando la lista proporcionada por la interfaz
    # --------------------------------------------------------------
    if inversiones_manual is None:
        inversiones = []
        voz_grave_anterior = None
        for idx, (cifrado, _, _, inv_forzado) in enumerate(asignaciones):
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

    # Carga los midis de referencia una única vez por inversión y
    # construye las posiciones repetidas para toda la progresión
    plantillas: dict[str, pretty_midi.PrettyMIDI] = {}
    parts = midi_ref.stem.split("_")
    base = "_".join(parts[:2]) if len(parts) >= 2 else midi_ref.stem
    if len(parts) >= 4:
        variante = parts[-1]
    for inv in INVERSIONS:
        path = midi_ref.parent / f"{base}_{inv}_{variante}.mid"
        plantillas[inv] = pretty_midi.PrettyMIDI(str(path))

    # Número real de corcheas en la progresión según el patrón de clave
    total_dest_cor = max(i for _, idxs, _, _ in asignaciones for i in idxs) + 1

    grupos_por_inv, total_ref_cor, grid, bpm = _cargar_grupos_por_inversion(plantillas)
    pm_ref = plantillas[inversion_inicial]
    offset_ref = 0

    # Mapa corchea -> índice de acorde y límites de cada acorde
    mapa: Dict[int, int] = {}
    limites: Dict[int, int] = {}
    for i, (_, idxs, _, _) in enumerate(asignaciones):
        for ix in idxs:
            mapa[ix] = i
        limites[i] = idxs[-1] + 1

    inv_por_cor: Dict[int, str] = {}
    for idx, (_, idxs, _, _) in enumerate(asignaciones):
        for ix in idxs:
            inv_por_cor[ix] = inversiones[idx]

    notas_finales: List[pretty_midi.Note] = []
    for cor in range(total_dest_cor):
        inv = inv_por_cor.get(cor)
        if inv is None:
            continue
        idx_acorde = mapa[cor]
        acorde, _, _, _ = asignaciones[idx_acorde]
        grupos_act = grupos_por_inv
        ref_idx = (inicio_cor + cor + offset_ref) % total_ref_cor
        for pos in grupos_act[inv][ref_idx]:
            pitch, es_aprox = traducir_nota(pos["name"], acorde)
            comienzo = asignaciones[idx_acorde][1][0]
            if CONVERTIR_APROX_A_ESTRUCT and es_aprox and cor == comienzo:
                pitch = _ajustar_a_estructural_mas_cercano(
                    pos["name"], cifrado=acorde, pitch=pitch
                )
            inicio = cor * grid + pos["start"]
            fin = cor * grid + pos["end"]
            fin_limite = limites[idx_acorde] * grid
            end = min(fin, fin_limite)
            if end <= inicio:
                continue
            notas_finales.append(
                pretty_midi.Note(
                    velocity=pos["velocity"],
                    pitch=pitch,
                    start=inicio,
                    end=end,
                )
            )

    # ------------------------------------------------------------------
    # Ajuste final de duración y bpm igual que en el modo tradicional
    # ------------------------------------------------------------------
    limite = total_dest_cor * grid
    notas_finales = _cortar_notas_superpuestas(notas_finales)
    notas_finales = _recortar_notas_a_limite(notas_finales, limite)
    if limite > 0:
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
        program=pm_ref.instruments[0].program,
        is_drum=pm_ref.instruments[0].is_drum,
        name=pm_ref.instruments[0].name,
    )
    inst.notes = notas_finales
    pm_out.instruments.append(inst)

    if return_pm:
        return pm_out

    pm_out.write(str(output))
