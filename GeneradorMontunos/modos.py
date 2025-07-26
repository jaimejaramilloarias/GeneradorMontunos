# -*- coding: utf-8 -*-
"""Definition of the available montuno generation modes."""

from pathlib import Path

import pretty_midi


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
    armonizacion: str | None = None,
    *,
    inicio_cor: int = 0,
    return_pm: bool = False,
    aleatorio: bool = False,
    armonizaciones_custom: list[str] | None = None,
) -> pretty_midi.PrettyMIDI | None:
    """Generate a montuno in the traditional style.

    ``armonizacion`` especifica la forma de duplicar las notas generadas. Por
    ahora solo se aplica la opción "Octavas".
    """
    asignaciones, compases = procesar_progresion_en_grupos(
        progresion_texto, armonizacion, inicio_cor=inicio_cor
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
