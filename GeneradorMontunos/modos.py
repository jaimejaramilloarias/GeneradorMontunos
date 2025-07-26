# -*- coding: utf-8 -*-
"""Definition of the available montuno generation modes."""

from pathlib import Path

import pretty_midi
from typing import List, Optional, Tuple


from voicings_tradicional import generar_voicings_enlazados_tradicional
from midi_utils_tradicional import (
    exportar_montuno,
    procesar_progresion_en_grupos,
)
from salsa import montuno_salsa


# ==========================================================================
# Traditional mode
# ==========================================================================

def montuno_tradicional(
    progresion_texto: str,
    midi_ref: Path,
    output: Path,
    armonizacion: Optional[str] = None,
    *,
    inicio_cor: int = 0,
    return_pm: bool = False,
    aleatorio: bool = False,
    armonizaciones_custom: Optional[List[str]] = None,
    asignaciones_custom: Optional[List[Tuple[str, List[int], str]]] = None,
) -> Optional[pretty_midi.PrettyMIDI]:
    """Generate a montuno in the traditional style.

    ``armonizacion`` especifica la forma de duplicar las notas generadas. Por
    ahora solo se aplica la opción "Octavas".
    """
    if asignaciones_custom is None:
        asignaciones, compases = procesar_progresion_en_grupos(
            progresion_texto, armonizacion, inicio_cor=inicio_cor
        )
    else:
        asignaciones = asignaciones_custom
        compases = (
            (max(i for _, idxs, _ in asignaciones for i in idxs) + 7) // 8
            if asignaciones
            else 0
        )
    if armonizaciones_custom is not None:
        for idx, arm in enumerate(armonizaciones_custom):
            if idx < len(asignaciones):
                nombre, idxs, _ = asignaciones[idx]
                asignaciones[idx] = (nombre, idxs, arm)
    acordes = [a for a, _, _ in asignaciones]
    voicings = generar_voicings_enlazados_tradicional(acordes)
    return exportar_montuno(
        midi_ref,
        voicings,
        asignaciones,
        compases,
        output,
        armonizacion=armonizacion,
        inicio_cor=inicio_cor,
        return_pm=return_pm,
        aleatorio=aleatorio,
    )


MODOS_DISPONIBLES = {
    "Tradicional": montuno_tradicional,
    "Salsa": montuno_salsa,
}
