from pathlib import Path
import pretty_midi

# Notas a números MIDI
NOTA_A_MIDI = {
    'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4, 'F':5, 'F#':6, 'Gb':6,
    'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'Bb':10, 'B':11
}

# Diccionario de intervalos por cifrado (ejemplos, expande según tu tabla)
DICCIONARIO_EXTENDIDA = {
    # 'cifrado': [fundamental, tercera, quinta, séptima, novena, onceava, treceava]
    'maj':  [0, 4, 7],
    '7':    [0, 4, 7, 10],
    '13':   [0, 4, 7, 10, 14, 17, 21],
    # agrega los tuyos aquí...
}

# Fórmulas de voicing (ejemplo para 13ª, ajusta y expande según tu tabla)
VOICINGS_EXTENDIDA = {
    '13': {
        'fundamental':  [0, 2, 3, 4, 0+12, 1+12, 2+12, 0+24],
        '1a_inv':       [1, 2, 3, 4, 0+12, 1+12, 2+12, 0+24],
        '2a_inv':       [2, 3, 4, 5, 0+12, 1+12, 2+12, 0+24],
    },
    # agrega para triadas, 7ª, 9ª, 11ª...
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
) -> pretty_midi.PrettyMIDI | None:
    """
    Modo 'Armonía extendida' usando lógica de intervalos y voicing modular.
    """
    # Ejemplo: procesa la progresión y obtiene lista de acordes e inversiones
    # Suponiendo que tienes: lista_de_acordes = [('G', '13', '2a_inv'), ...]
    lista_de_acordes = []  # Rellena con tu parser

    for (fundamental, cifrado, inversion) in lista_de_acordes:
        intervalos = DICCIONARIO_EXTENDIDA[cifrado]
        voicing = VOICINGS_EXTENDIDA[cifrado][inversion]
        notas_midi = get_midi_numbers(fundamental, intervalos, voicing, octava_base=3)
        # ... genera tus notas midi y añade al pretty_midi (estructura igual a modo salsa para exportar)

    # Arma el midi como en los otros modos y devuelve el pm_out
    return None  # o pm_out
