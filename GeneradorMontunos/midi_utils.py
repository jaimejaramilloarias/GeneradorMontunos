# -*- coding: utf-8 -*-
"""Helpers for reading, manipulating and exporting MIDI files."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import pretty_midi
import random
import logging
from voicings import parsear_nombre_acorde, INTERVALOS_TRADICIONALES
from midi_common import (
    NOTAS_BASE,
    leer_midi_referencia,
    obtener_posiciones_referencia,
    construir_posiciones_secuenciales,
    construir_posiciones_por_ventanas,
)

# All reference MIDI loops have the same length (32 bars with 8 eighth-notes
# each). Tempo information is ignored so the default player tempo is used.
NORMALIZED_BPM = 200.0  # Unused but kept for compatibility


logger = logging.getLogger(__name__)



# ==========================================================================
# MIDI export utilities
# ==========================================================================


def aplicar_voicings_a_referencia(
    posiciones: List[dict],
    voicings: List[List[int]],
    asignaciones: List[Tuple[str, List[int]]],
    grid_seg: float,
    *,
    notas_base: List[int] = NOTAS_BASE,
    debug: bool = False,
) -> Tuple[List[pretty_midi.Note], int]:
    """Reemplaza las notas de referencia por los voicings generados.

    Devuelve la lista de nuevas notas y el último índice de corchea utilizado.
    """

    # Mapeo corchea → índice de voicing
    mapa: Dict[int, int] = {}
    max_idx = -1
    for i, data in enumerate(asignaciones):
        idxs = data[1]
        for ix in idxs:
            mapa[ix] = i
            if ix > max_idx:
                max_idx = ix

    nuevas_notas: List[pretty_midi.Note] = []

    for pos in posiciones:
        corchea = int(round(pos["start"] / grid_seg))
        if corchea not in mapa:
            if debug:
                logger.debug("Corchea %s: silencio", corchea)
            continue  # silencio
        voicing = sorted(voicings[mapa[corchea]])
        orden = notas_base.index(pos["pitch"])  # posición dentro del voicing
        # Preserve the velocity of the reference note so dynamics match
        nueva_nota = pretty_midi.Note(
            velocity=pos["velocity"],
            pitch=voicing[orden],
            start=pos["start"],
            end=pos["end"],
        )
        if debug:
            logger.debug("Corchea %s: nota base %s -> %s", corchea, pos['pitch'], nueva_nota.pitch)
        nuevas_notas.append(nueva_nota)

    return nuevas_notas, max_idx


def _arm_octavas(notas: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
    """Duplicate each note one octave above."""

    resultado: List[pretty_midi.Note] = []
    for n in notas:
        resultado.append(n)
        if n.pitch > 0:
            resultado.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch + 12,
                    start=n.start,
                    end=n.end,
                )
            )
    return resultado


def _arm_doble_octava(notas: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
    """Create notes an octave below and above, without keeping the original."""

    resultado: List[pretty_midi.Note] = []
    for n in notas:
        if n.pitch > 0:
            # Copy the velocity from the original note
            resultado.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch - 12,
                    start=n.start,
                    end=n.end,
                )
            )
            resultado.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch + 12,
                    start=n.start,
                    end=n.end,
                )
            )
    return resultado


def _arm_por_parejas(
    posiciones: List[dict],
    voicings: List[List[int]],
    asignaciones: List[Tuple[str, List[int]]],
    grid_seg: float,
    salto: int,
    *,
    debug: bool = False,
) -> List[pretty_midi.Note]:
    """Generate notes in parallel motion (décimas or sixths).

    Each chord ``voicing`` is walked sequentially using the eighth-note
    positions assigned to it.  ``salto`` determines the pairing pattern:
    ``1`` produces décimas (third + octave) and ``2`` produces sixths.
    The rhythmic information (start, end and velocity) is taken from the
    reference ``posiciones`` list.
    """

    # Map each eighth index to the corresponding voicing/chord
    mapa: Dict[int, int] = {}
    for i, data in enumerate(asignaciones):
        idxs = data[1]
        for ix in idxs:
            mapa[ix] = i

    # Counter so each chord advances through its voicing in parallel
    contadores: Dict[int, int] = {}

    resultado: List[pretty_midi.Note] = []
    for pos in posiciones:
        corchea = int(round(pos["start"] / grid_seg))
        if corchea not in mapa:
            if debug:
                logger.debug("Corchea %s: silencio", corchea)
            continue

        idx_voicing = mapa[corchea]
        paso = contadores.get(idx_voicing, 0)
        contadores[idx_voicing] = paso + 1

        voicing = sorted(voicings[idx_voicing])

        if salto == 1:  # décimas
            principal = voicing[paso % 4]
            agregada = voicing[(paso + 1) % 4] + 12
        else:  # antiguas sextas
            principal = voicing[(paso + 1) % 4]
            agregada = voicing[paso % 4] + 12

        # Ensure the upper note never sits in the same octave as the
        # principal voice.  This avoids "collapsed" intervals when the
        # voicing spans less than an octave.
        while agregada <= principal:
            agregada += 12

        for pitch in (principal, agregada):
            resultado.append(
                pretty_midi.Note(
                    velocity=pos["velocity"],
                    pitch=pitch,
                    start=pos["start"],
                    end=pos["end"],
                )
            )

        if debug:
            logger.debug("Corchea %s: paso %s -> %s / %s", corchea, paso, principal, agregada)

    return resultado


def _arm_decimas_intervalos(
    posiciones: List[dict],
    voicings: List[List[int]],
    asignaciones: List[Tuple[str, List[int]]],
    grid_seg: float,
    *,
    debug: bool = False,
) -> List[pretty_midi.Note]:
    """Harmonize in parallel tenths following fixed functional pairs.

    Before processing the MIDI positions each chord is analysed so every
    pitch can be labelled as fundamental, third, fifth, sixth or seventh.
    The added note is then obtained with the exact interval mandated by the
    specification:

    * F → 3 (+12)
    * 3 → 5 (+12)
    * 5 → 7 (+12) or M7 (+12) on sixth chords
    * 6 or diminished 7 → F (+24)
    * 7 → 9 (+24)

    Velocity and timing from the reference are preserved verbatim.
    """

    # ------------------------------------------------------------------
    # Build a map from eighth index to voicing index and gather information
    # about each chord so that every pitch can be classified by function.
    # ``info`` stores the root pitch class, the four intervals of the chord
    # and flags indicating whether it is a sixth chord or a diminished
    # seventh.
    # ------------------------------------------------------------------
    mapa: Dict[int, int] = {}
    for i, data in enumerate(asignaciones):
        idxs = data[1]
        for ix in idxs:
            mapa[ix] = i

    info: List[Dict] = []
    for data in asignaciones:
        nombre = data[0]
        root_pc, suf = parse_fn(nombre)
        ints = interval_dict[suf]
        is_sixth = suf.endswith("6") and "7" not in suf
        is_dim7 = suf == "º7"
        info.append(
            {
                "root_pc": root_pc,
                "intervals": ints,
                "is_sixth": is_sixth,
                "is_dim7": is_dim7,
                "suf": suf,
            }
        )

    contadores: Dict[int, int] = {}
    offsets: Dict[int, int] = {}
    bajo_anterior: Optional[int] = None
    arm_anterior: Optional[str] = None
    resultado: List[pretty_midi.Note] = []

    for pos in posiciones:
        corchea = int(round(pos["start"] / grid_seg))
        if corchea not in mapa:
            if debug:
                logger.debug("Corchea %s: silencio", corchea)
            continue

        idx = mapa[corchea]
        paso = contadores.get(idx, 0)
        contadores[idx] = paso + 1

        datos = info[idx]
        voicing = sorted(voicings[idx])
        base = voicing[paso % 4]
        root_pc = datos["root_pc"]
        ints = datos["intervals"]
        is_sixth = datos["is_sixth"]
        is_dim7 = datos["is_dim7"]
        suf = datos["suf"]
        suf = datos["suf"]

        # --------------------------------------------------------------
        # Identify the function of ``base`` comparing its pitch class
        # against the intervals of the current chord.
        # --------------------------------------------------------------
        pc = base % 12
        func = None
        base_int = None
        if pc == (root_pc + ints[0]) % 12:
            func = "F"
            base_int = ints[0]
            target_int = ints[1]
        elif pc == (root_pc + ints[1]) % 12:
            func = "3"
            base_int = ints[1]
            target_int = ints[2]
        elif pc == (root_pc + ints[2]) % 12:
            func = "5"
            base_int = ints[2]
            target_int = 11 if is_sixth else ints[3]
        elif pc == (root_pc + ints[3]) % 12:
            base_int = ints[3]
            if is_sixth or is_dim7:
                func = "6"
                target_int = ints[0]
            else:
                func = "7"
                if suf in ("7(b9)", "+7(b9)", "7(b5)b9", "7sus4(b9)"):
                    target_int = ints[4]
                else:
                    target_int = 2
        else:
            base_int = pc
            target_int = pc

        # --------------------------------------------------------------
        # Compute the required interval (15 or 16 semitones) based on
        # ``base_int`` and ``target_int``.  ``target_int`` is expected to be
        # higher than ``base_int`` within the chord definition.  The added
        # note is placed exactly ``diff`` semitones above ``base``.
        # --------------------------------------------------------------
        diff = (target_int - base_int) + (24 if func in ("6", "7") else 12)
        # If the added note is the flat nine, force a minor tenth (15 semitones)
        # above the principal voice even if it exceeds the usual range.
        if func == "7" and target_int == 13:
            diff = (target_int - base_int) + 12
        agregada = base + diff

        if debug:
            logger.debug(
                "Corchea %s: paso %s %s %s (%s) -> %s",
                corchea,
                paso,
                asignaciones[idx][0],
                pretty_midi.note_number_to_name(base),
                func,
                pretty_midi.note_number_to_name(agregada),
            )

        for pitch in (base, agregada):
            resultado.append(
                pretty_midi.Note(
                    velocity=pos["velocity"],
                    pitch=pitch,
                    start=pos["start"],
                    end=pos["end"],
                )
            )

    return resultado


def _arm_treceavas_intervalos(
    posiciones: List[dict],
    voicings: List[List[int]],
    asignaciones: List[Tuple[str, List[int]]],
    grid_seg: float,
    *,
    debug: bool = False,
) -> List[pretty_midi.Note]:
    """Generate inverted tenths resulting in thirteenths below.

    This uses the same functional logic as :func:`_arm_decimas_intervalos` but
    the pair of voices is inverted: the principal note is raised an octave and
    the added voice is placed a thirteenth (20 or 21 semitones) below it.
    """

    mapa: Dict[int, int] = {}
    for i, data in enumerate(asignaciones):
        idxs = data[1]
        for ix in idxs:
            mapa[ix] = i

    info: List[Dict] = []
    for data in asignaciones:
        nombre = data[0]
        root_pc, suf = parse_fn(nombre)
        ints = interval_dict[suf]
        is_sixth = suf.endswith("6") and "7" not in suf
        is_dim7 = suf == "º7"
        info.append(
            {
                "root_pc": root_pc,
                "intervals": ints,
                "is_sixth": is_sixth,
                "is_dim7": is_dim7,
                "suf": suf,
            }
        )

    contadores: Dict[int, int] = {}
    resultado: List[pretty_midi.Note] = []

    for pos in posiciones:
        corchea = int(round(pos["start"] / grid_seg))
        if corchea not in mapa:
            if debug:
                logger.debug("Corchea %s: silencio", corchea)
            continue

        idx = mapa[corchea]
        paso = contadores.get(idx, 0)
        contadores[idx] = paso + 1

        datos = info[idx]
        voicing = sorted(voicings[idx])
        base = voicing[paso % 4]
        root_pc = datos["root_pc"]
        ints = datos["intervals"]
        is_sixth = datos["is_sixth"]
        is_dim7 = datos["is_dim7"]

        pc = base % 12
        func = None
        base_int = None
        if pc == (root_pc + ints[0]) % 12:
            func = "F"
            base_int = ints[0]
            target_int = ints[1]
        elif pc == (root_pc + ints[1]) % 12:
            func = "3"
            base_int = ints[1]
            target_int = ints[2]
        elif pc == (root_pc + ints[2]) % 12:
            func = "5"
            base_int = ints[2]
            target_int = 11 if is_sixth else ints[3]
        elif pc == (root_pc + ints[3]) % 12:
            base_int = ints[3]
            if is_sixth or is_dim7:
                func = "6"
                target_int = ints[0]
            else:
                func = "7"
                if suf in ("7(b9)", "+7(b9)", "7(b5)b9", "7sus4(b9)"):
                    target_int = ints[4]
                else:
                    target_int = 2
        else:
            base_int = pc
            target_int = pc

        diff = (target_int - base_int) + (24 if func in ("6", "7") else 12)
        # Si la nota agregada es la novena menor, se fuerza una «décima menor»
        # (15 semitonos) por encima de la voz principal aunque se supere el
        # registro habitual.
        if func == "7" and target_int == 13:
            diff = (target_int - base_int) + 12
        agregada = base + diff

        principal = base + 12
        inferior = agregada - 24

        if debug:
            logger.debug(
                "Corchea %s: paso %s -> %s / %s",
                corchea,
                paso,
                pretty_midi.note_number_to_name(principal),
                pretty_midi.note_number_to_name(inferior),
            )

        for pitch in (principal, inferior):
            resultado.append(
                pretty_midi.Note(
                    velocity=pos["velocity"],
                    pitch=pitch,
                    start=pos["start"],
                    end=pos["end"],
                )
            )

    return resultado


def _arm_noop(notas: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
    """Placeholder for future harmonization types."""

    return notas


# Armonizaciones simples que no dependen del contexto del voicing
_ARMONIZADORES = {
    "octavas": _arm_octavas,
    "doble octava": _arm_doble_octava,
}


def _ajustar_salto(prev_pitch: Optional[int], pitch: int) -> int:
    """Return ``pitch`` transposed by octaves so the leap from ``prev_pitch``
    is less than an octave."""

    if prev_pitch is None:
        return pitch
    while pitch - prev_pitch >= 12:
        pitch -= 12
    while prev_pitch - pitch >= 12:
        pitch += 12
    return pitch


def generar_notas_mixtas(
    posiciones: List[dict],
    voicings: List[List[int]],
    asignaciones: List[Tuple[str, List[int], str]],
    grid_seg: float,
    *,
    notas_base: List[int] = NOTAS_BASE,
    debug: bool = False,
    parse_fn=parsear_nombre_acorde,
    interval_dict=INTERVALOS_TRADICIONALES,
) -> List[pretty_midi.Note]:
    """Generate notes applying per-chord harmonisation.

    ``asignaciones`` debe contener tuplas ``(acorde, indices, armonizacion)``.
    """

    mapa: Dict[int, int] = {}
    armonias: Dict[int, str] = {}
    for i, data in enumerate(asignaciones):
        idxs = data[1]
        arm = data[2]
        for ix in idxs:
            mapa[ix] = i
        armonias[i] = (arm or "").lower()

    info: List[Dict] = []
    for data in asignaciones:
        nombre = data[0]
        root_pc, suf = parse_fn(nombre)
        ints = interval_dict[suf]
        is_sixth = suf.endswith("6") and "7" not in suf
        is_dim7 = suf == "º7"
        info.append(
            {
                "root_pc": root_pc,
                "intervals": ints,
                "is_sixth": is_sixth,
                "is_dim7": is_dim7,
                "suf": suf,
            }
        )

    contadores: Dict[int, int] = {}
    offsets: Dict[int, int] = {}
    bajo_anterior: Optional[int] = None
    arm_anterior: Optional[str] = None
    resultado: List[pretty_midi.Note] = []

    for pos in posiciones:
        corchea = int(round(pos["start"] / grid_seg))
        if corchea not in mapa:
            if debug:
                logger.debug("Corchea %s: silencio", corchea)
            continue

        idx = mapa[corchea]
        arm = armonias.get(idx, "")
        paso = contadores.get(idx, 0)
        contadores[idx] = paso + 1
        voicing = sorted(voicings[idx])

        if arm in ("décimas", "treceavas"):
            datos = info[idx]
            base = voicing[paso % 4]
            root_pc = datos["root_pc"]
            ints = datos["intervals"]
            is_sixth = datos["is_sixth"]
            is_dim7 = datos["is_dim7"]
            suf = datos["suf"]

            pc = base % 12
            if pc == (root_pc + ints[0]) % 12:
                base_int = ints[0]
                target_int = ints[1]
                func = "F"
            elif pc == (root_pc + ints[1]) % 12:
                base_int = ints[1]
                target_int = ints[2]
                func = "3"
            elif pc == (root_pc + ints[2]) % 12:
                base_int = ints[2]
                target_int = 11 if is_sixth else ints[3]
                func = "5"
            elif pc == (root_pc + ints[3]) % 12:
                base_int = ints[3]
                if is_sixth or is_dim7:
                    target_int = ints[0]
                    func = "6"
                else:
                    if suf in ("7(b9)", "+7(b9)", "7(b5)b9", "7sus4(b9)"):
                        target_int = ints[4]
                    else:
                        target_int = 2
                    func = "7"
            else:
                base_int = pc
                target_int = pc
                func = "?"

            diff = (target_int - base_int) + (24 if func in ("6", "7") else 12)
            # For flat nine the interval is forced to a minor tenth (15 semitones)
            # above the principal voice even if it breaks range limits.
            if func == "7" and target_int == 13:
                diff = (target_int - base_int) + 12
            agregada = base + diff

            if arm == "décimas":
                notas = [base, agregada]
            else:  # treceavas
                notas = [base + 12, agregada - 24]
        else:
            # Procesamiento estandar del voicing base
            orden = notas_base.index(pos["pitch"])
            base_pitch = voicing[orden]

            if arm == "octavas":
                notas = [base_pitch, base_pitch + 12]
            elif arm == "doble octava":
                notas = []
                if base_pitch > 0:
                    notas.extend([base_pitch - 12, base_pitch + 12])
            else:
                notas = [base_pitch]

        offset = offsets.get(idx, 0)
        if paso == 0:
            bajo = min(notas)
            if arm != arm_anterior:
                ajustado = _ajustar_salto(bajo_anterior, bajo)
                offset = ajustado - bajo
                offsets[idx] = offset
                bajo_anterior = ajustado
                arm_anterior = arm
            else:
                offsets[idx] = offset
                bajo_anterior = bajo + offset
                arm_anterior = arm
        else:
            offset = offsets.get(idx, 0)

        if debug and paso == 0:
            logger.debug(
                "Corchea %s: paso %s -> %s", corchea, paso, [p + offset for p in notas]
            )

        for pitch in notas:
            resultado.append(
                pretty_midi.Note(
                    velocity=pos["velocity"],
                    pitch=pitch + offset,
                    start=pos["start"],
                    end=pos["end"],
                )
            )

    return resultado


def generar_notas_voicing_completo(
    posiciones: List[dict],
    voicings: List[List[int]],
    asignaciones: List[Tuple[str, List[int], str]],
    grid_seg: float,
    *,
    debug: bool = False,
) -> List[pretty_midi.Note]:
    """Generate notes reproducing each voicing in full for every eighth note."""

    mapa: Dict[int, int] = {}
    for i, (_, idxs, _) in enumerate(asignaciones):
        for ix in idxs:
            mapa[ix] = i

    primer_evento: Dict[int, dict] = {}
    for pos in posiciones:
        cor = int(round(pos["start"] / grid_seg))
        if cor not in primer_evento or pos["start"] < primer_evento[cor]["start"]:
            primer_evento[cor] = pos

    resultado: List[pretty_midi.Note] = []
    for cor, idx in mapa.items():
        pos = primer_evento.get(cor)
        if pos is None:
            continue
        for pitch in voicings[idx]:
            resultado.append(
                pretty_midi.Note(
                    velocity=pos["velocity"],
                    pitch=pitch,
                    start=pos["start"],
                    end=pos["end"],
                )
            )
        if debug:
            logger.debug("Corchea %s -> %s", cor, voicings[idx])

    return resultado


def aplicar_armonizacion(
    notas: List[pretty_midi.Note], opcion: str
) -> List[pretty_midi.Note]:
    """Apply the selected harmonization option using ``_ARMONIZADORES``."""

    funcion = _ARMONIZADORES.get(opcion.lower())
    if funcion is None:
        return notas
    return funcion(notas)


def _grid_and_bpm(pm: pretty_midi.PrettyMIDI) -> Tuple[int, float, float]:
    """Return the reference length, eighth duration and BPM.

    The project assumes all reference templates span exactly 32 bars
    (``256`` eighth-notes).  Tempo data in the files is ignored and a
    constant tempo of ``120`` BPM is used for every template so the
    resulting grid is always identical.
    """

    bpm = 120.0
    grid = 60.0 / bpm / 2  # seconds per eighth note
    cor = 256
    return cor, grid, bpm


def normalize_tempo(pm: pretty_midi.PrettyMIDI, target_bpm: float = NORMALIZED_BPM) -> pretty_midi.PrettyMIDI:
    """Return ``pm`` unchanged.

    Tempo normalization has been disabled because all templates already lack
    tempo messages.  The caller may still invoke this function for
    compatibility, but no processing is performed.
    """

    return pm


def _recortar_notas_a_limite(
    notas: List[pretty_midi.Note], limite: float
) -> List[pretty_midi.Note]:
    """Recorta las notas para que no se extiendan más allá de ``limite``.

    Cualquier nota que termine después del instante indicado se acorta para
    que su atributo ``end`` coincida exactamente con ``limite``.  Las notas
    cuyo ``start`` es posterior al límite se descartan.
    """

    recortadas: List[pretty_midi.Note] = []
    for n in notas:
        if n.start >= limite:
            continue
        if n.end > limite:
            n.end = limite
        recortadas.append(n)
    return recortadas


def _cortar_notas_superpuestas(notas: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
    """Shorten notes to avoid overlaps at the same pitch.

    If two consecutive notes share the same ``pitch`` and the first note
    extends beyond the start of the second, the first note is truncated so
    that it ends exactly when the following one begins.  This prevents MIDI
    artefacts caused by overlapping identical pitches.
    """

    agrupadas: Dict[int, List[pretty_midi.Note]] = {}
    for n in sorted(notas, key=lambda x: (x.pitch, x.start)):
        lista = agrupadas.setdefault(n.pitch, [])
        if lista and lista[-1].end > n.start:
            lista[-1].end = n.start
        lista.append(n)

    resultado = [n for lst in agrupadas.values() for n in lst]
    resultado.sort(key=lambda x: (x.start, x.pitch))
    return resultado


def exportar_montuno(
    midi_referencia_path: Path,
    voicings: List[List[int]],
    asignaciones: List[Tuple[str, List[int], str]],
    num_compases: int,
    output_path: Path,
    armonizacion: Optional[str] = None,
    *,
    inicio_cor: int = 0,
    debug: bool = False,
    return_pm: bool = False,
    aleatorio: bool = False,
    voicing_offsets: Optional[List[int]] = None,
    parse_fn=parsear_nombre_acorde,
    interval_dict=INTERVALOS_TRADICIONALES,
    full_voicing: bool = False,
) -> Optional[pretty_midi.PrettyMIDI]:
    """Generate a new MIDI file with the given voicings.

    The resulting notes are trimmed so the output stops after the last
    eighth-note of the progression. ``inicio_cor`` is the global eighth-note
    index where this segment begins.  ``armonizacion`` specifies how notes
    should be duplicated.  Set ``return_pm`` to ``True`` to return the
    generated object instead of writing it to disk.  When ``full_voicing`` is
    ``True`` the complete list of pitches for each voicing is inserted at
    every eighth-note position instead of mapping notes one by one.
    """
    notes, pm = leer_midi_referencia(midi_referencia_path)
    posiciones_base, notas_base = obtener_posiciones_referencia(notes)
    total_cor_ref, grid, bpm = _grid_and_bpm(pm)

    if debug:
        logger.debug("Asignacion de acordes a corcheas:")
        for acorde, idxs, arm, *_ in asignaciones:
            logger.debug("  %s (%s): %s", acorde, arm, idxs)

    if asignaciones:
        total_dest_cor = max(i for _, idxs, *_ in asignaciones for i in idxs) + 1
    else:
        total_dest_cor = num_compases * 8
    limite_cor = total_dest_cor
    # --------------------------------------------------------------
    # The reference must align with the absolute eighth-note position of
    # the progression so changes of mode or template never break the
    # continuity. ``inicio_cor`` indicates the global index where this
    # segment starts; use it modulo the reference length to pick the
    # correct starting point.
    # --------------------------------------------------------------
    inicio_ref = inicio_cor % total_cor_ref
    if aleatorio:
        posiciones = construir_posiciones_por_ventanas(
            posiciones_base,
            limite_cor,
            total_cor_ref,
            grid,
            inicio_cor=inicio_ref,
            compases_ventana=2,
            aleatorio=True,
        )
    else:
        posiciones = construir_posiciones_secuenciales(
            posiciones_base,
            limite_cor,
            total_cor_ref,
            grid,
            inicio_cor=inicio_ref,
        )

    limite = limite_cor * grid

    if voicing_offsets:
        voicings = [
            [p + voicing_offsets[i] for p in v]
            for i, v in enumerate(voicings)
        ]

    if full_voicing:
        nuevas_notas = generar_notas_voicing_completo(
            posiciones, voicings, asignaciones, grid, debug=debug
        )
    else:
        nuevas_notas = generar_notas_mixtas(
            posiciones,
            voicings,
            asignaciones,
            grid,
            notas_base=notas_base,
            debug=debug,
            parse_fn=parse_fn,
            interval_dict=interval_dict,
        )

    # Avoid overlapping notes at the same pitch which can cause MIDI
    # artefacts by trimming preceding notes when necessary.
    nuevas_notas = _cortar_notas_superpuestas(nuevas_notas)

    # ------------------------------------------------------------------
    # Ajuste final de duracion: todas las notas se recortan para que
    # terminen, como maximo, en la ultima corchea programada.
    # ------------------------------------------------------------------
    nuevas_notas = _recortar_notas_a_limite(nuevas_notas, limite)

    if limite > 0:
        has_start = any(n.start <= 0 < n.end and n.pitch > 0 for n in nuevas_notas)
        has_end = any(
            n.pitch > 0
            and n.start < limite
            and n.end > limite - grid
            for n in nuevas_notas
        )
        if not has_start:
            nuevas_notas.append(
                pretty_midi.Note(
                    velocity=1,
                    pitch=21,
                    start=0.0,
                    end=min(grid, limite),
                )
            )
        if not has_end:
            nuevas_notas.append(
                pretty_midi.Note(
                    velocity=1,
                    pitch=21,
                    start=max(0.0, limite - grid),
                    end=limite,
                )
            )

    pm_out = pretty_midi.PrettyMIDI()
    inst_out = pretty_midi.Instrument(
        program=pm.instruments[0].program,
        is_drum=pm.instruments[0].is_drum,
        name=pm.instruments[0].name,
    )
    inst_out.notes = nuevas_notas
    pm_out.instruments.append(inst_out)
    if return_pm:
        return pm_out

    pm_out.write(str(output_path))


# ==========================================================================
# Traditional rhythmic grouping
# ==========================================================================

# ---------------------------------------------------------------------------
# Rhythmic pattern configuration
# ---------------------------------------------------------------------------
# ``PRIMER_BLOQUE`` y ``PATRON_REPETIDO`` definen el esquema de agrupación de
# corcheas utilizado por el modo tradicional.  El primer bloque se utiliza tal
# cual una única vez y a partir de entonces se repite ``PATRON_REPETIDO`` de
# forma indefinida.  Para cambiar el patrón basta con modificar estas dos
# listas.
PRIMER_BLOQUE: List[int] = [3, 4, 4, 3]
PATRON_REPETIDO: List[int] = [5, 4, 4, 3]

# ``PATRON_GRUPOS`` se mantiene solo como referencia para visualizar los
# primeros valores calculados con la configuración actual.
PATRON_GRUPOS: List[int] = PRIMER_BLOQUE + PATRON_REPETIDO * 3


def _siguiente_grupo(indice: int) -> int:
    """Devuelve la longitud del grupo de corcheas según ``indice``.

    Los cuatro primeros valores provienen de ``PRIMER_BLOQUE`` y, a partir de
    ahí, se repite ``PATRON_REPETIDO`` tantas veces como sea necesario.
    """
    if indice < len(PRIMER_BLOQUE):
        return PRIMER_BLOQUE[indice]
    indice -= len(PRIMER_BLOQUE)
    return PATRON_REPETIDO[indice % len(PATRON_REPETIDO)]


def _indice_para_corchea(cor: int) -> int:
    """Return the pattern index corresponding to ``cor`` eighth-notes."""

    idx = 0
    pos = 0
    while pos < cor:
        pos += _siguiente_grupo(idx)
        idx += 1
    return idx


def procesar_progresion_en_grupos(
    texto: str,
    armonizacion_default: Optional[str] = None,
    *,
    inicio_cor: int = 0,
) -> Tuple[List[Tuple[str, List[int], str]], int]:
    """Asignar corcheas por compases según las barras ``|``.

    Un segmento con un solo acorde ocupa dos grupos consecutivos de corcheas.
    Si contiene dos acordes cada uno recibe un grupo. Cualquier segmento con
    más de dos acordes genera un ``ValueError``.
    """

    import re

    segmentos_raw = [s.strip() for s in texto.split("|") if s.strip()]

    segmentos: List[str] = []
    for seg in segmentos_raw:
        if seg == "%":
            if not segmentos:
                raise ValueError("% no puede ir en el primer compás")
            segmentos.append(segmentos[-1])
        else:
            segmentos.append(seg)

    resultado: List[Tuple[str, List[int], str]] = []
    indice_patron = _indice_para_corchea(inicio_cor)
    posicion = 0

    arm_actual = (armonizacion_default or "").capitalize()

    def procesar_token(token: str) -> Tuple[Optional[str], Optional[str]]:
        nonlocal arm_actual

        m = re.match(r"^\[[A-Z]+\](.*)$", token)
        if m:
            token = m.group(1)
            if not token:
                return None, None

        m = re.match(r"^\((8|10|13|15)\)(.*)$", token)
        if m:
            codigo, token = m.groups()
            arm_map = {
                "8": "Octavas",
                "15": "Doble octava",
                "10": "Décimas",
                "13": "Treceavas",
            }
            arm_actual = arm_map[codigo]
            if not token:
                return None, None

        if not token:
            return None, None

        return token, arm_actual

    for seg in segmentos:
        tokens = [t for t in seg.split() if t]
        acordes: List[tuple[str, str]] = []
        for tok in tokens:
            nombre, arm = procesar_token(tok)
            if nombre is None:
                continue
            acordes.append((nombre, arm or ""))
        if len(acordes) == 1:
            g1 = _siguiente_grupo(indice_patron)
            g2 = _siguiente_grupo(indice_patron + 1)
            dur = g1 + g2
            indices = list(range(posicion, posicion + dur))
            nombre, arm = acordes[0]
            resultado.append((nombre, indices, arm))
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

            (n1, a1), (n2, a2) = acordes
            resultado.append((n1, indices1, a1))
            resultado.append((n2, indices2, a2))
        elif len(acordes) == 0:
            continue
        else:
            raise ValueError(f"Cada segmento debe contener uno o dos acordes: {seg}")

    for acorde, idxs, arm in resultado:
        logger.debug("%s (%s): %s", acorde, arm, idxs)

    num_compases = len(segmentos)
    return resultado, num_compases
