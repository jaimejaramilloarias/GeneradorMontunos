import sys
sys.path.append('GeneradorMontunos')
import pretty_midi
from armonia_extendida import (
    VOICINGS_EXTENDIDA,
    DICCIONARIO_EXTENDIDA,
    parsear_nombre_acorde,
    get_midi_numbers_debug,
)

inv_map = {"fundamental": "fundamental", "1a": "1a_inv", "2a": "2a_inv"}

tipo_map = {3: "triada", 4: "7_6", 5: "9", 6: "11", 7: "13"}

def mostrar(cifrado, inversion):
    root, suf = parsear_nombre_acorde(cifrado)
    intervalos = DICCIONARIO_EXTENDIDA[suf]
    tipo = tipo_map[len(intervalos)]
    pattern = VOICINGS_EXTENDIDA[tipo][inversion]
    notas, dbg = get_midi_numbers_debug(root, intervalos, pattern)
    nombres = [pretty_midi.note_number_to_name(n) for n in notas]
    print(f"{cifrado} {inversion} -> {nombres}")
    for ref, token, midi in dbg:
        name = pretty_midi.note_number_to_name(midi)
        print(f"  {ref}: {token} -> {name}({midi})")
    return notas

def main():
    errores = []
    acordes = [
        ("C", "fundamental"),
        ("C", "1a_inv"),
        ("C", "2a_inv"),
        ("C7", "fundamental"),
        ("C7", "1a_inv"),
        ("C7", "2a_inv"),
        ("C13", "fundamental"),
        ("C13", "1a_inv"),
        ("C13", "2a_inv"),
    ]
    for cifrado, inv in acordes:
        try:
            mostrar(cifrado, inv)
        except Exception as exc:
            errores.append(str(exc))
    if errores:
        print("Errores:\n" + "\n".join(errores))
    else:
        print("VOICINGS_EXTENDIDA OK")

if __name__ == "__main__":
    main()
