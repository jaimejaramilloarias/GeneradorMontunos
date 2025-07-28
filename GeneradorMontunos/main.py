# -*- coding: utf-8 -*-
"""Simple GUI for montuno generation."""

from pathlib import Path
from tempfile import TemporaryDirectory
from tkinter import PhotoImage
from tkinter import colorchooser
import tkinter.font as tkfont
import customtkinter as ctk
import re
import logging
import os
from customtkinter import (
    CTk,
    CTkFrame as Frame,
    CTkLabel as Label,
    CTkButton as Button,
    CTkRadioButton as Radiobutton,
    CTkCanvas as Canvas,
    CTkScrollbar as Scrollbar,
    CTkToplevel as Toplevel,
    CTkOptionMenu as OptionMenu,
    CTkComboBox as ComboBox,
    CTkTextbox as Text,
    CTkEntry,
)
from tkinter import StringVar

# Attempt to use ``customtkinter`` if available.  The rest of the file falls
# back to the standard ``tkinter`` widgets when the import fails so the
# dependency remains optional.
try:
    import customtkinter as ctk
except Exception:  # pragma: no cover - optional dependency
    ctk = None
import threading
import time
import mido
import pretty_midi
import pygame.midi
from typing import Dict, List, Optional, Tuple, Set

from autocomplete import ChordAutocomplete

import midi_utils
import midi_utils_tradicional
import salsa
import style_utils
from utils import (
    limpiar_inversion,
    apply_manual_edits,
    calc_default_inversions,
    normalise_bars,
    RE_BAR_CLEAN,
    clean_tokens,
)
from ui_config import (
    COLORS,
    get_entry_font,
    get_general_font,
    get_measure_font,
    load_preferences,
    save_preferences,
)

from modos import MODOS_DISPONIBLES

# Base directory of the project to build absolute paths to resources.
BASE_DIR = Path(__file__).resolve().parent

# Short labels for drop down menus
ARMONIZACION_LABELS = {
    "Octavas": "Oct",
    "Doble octava": "D. Oct",
    "Décimas": "10a",
    "Treceavas": "13a",
}
ARMONIZACION_INV = {v: k for k, v in ARMONIZACION_LABELS.items()}

MODOS_LABELS = {"Tradicional": "Trad.", "Salsa": "Salsa", "Armonía extendida": "Ext."}
MODOS_INV = {v: k for k, v in MODOS_LABELS.items()}

# ---------------------------------------------------------------------------
# Regular expressions compiled once for efficiency
# ---------------------------------------------------------------------------
CHORD_RE = re.compile(
    r"(?<![A-Za-z0-9#bº°+∆m7(b5)])([A-G](?:b|#)?[a-zA-Z0-9º°+∆m7(b5)]*(?:\([^)]*\))*)"
)
CHORD_CURSOR_RE = re.compile(r"(?:^|[\s|])([A-G](?:b|#)?[A-Za-z0-9º°+∆]*(?:\([^)]*\))*)")

# Width in pixels for each eighth-note cell in the piano roll
CELL_WIDTH = 40

# Maximum height in pixels for the chord input box
CHORD_INPUT_MAX_HEIGHT = 80

# Global undo/redo stacks used for the universal undo feature
UNDO_STACK: List[Dict] = []
REDO_STACK: List[Dict] = []
_suppress_undo = False
_updating = False

# Keep a persistent MIDI output to avoid repeated initialisation errors
MIDI_PORT = None

logger = logging.getLogger(__name__)


class _MultiPort:
    """Simple wrapper to send MIDI messages to multiple ports."""

    def __init__(self, ports: List[pygame.midi.Output]):
        self.ports = ports

    def note_on(self, *args, **kwargs) -> None:  # pragma: no cover - UI code
        for p in self.ports:
            try:
                p.note_on(*args, **kwargs)
            except Exception:
                pass

    def note_off(self, *args, **kwargs) -> None:  # pragma: no cover - UI code
        for p in self.ports:
            try:
                p.note_off(*args, **kwargs)
            except Exception:
                pass

    def write_short(self, *args) -> None:  # pragma: no cover - UI code
        for p in self.ports:
            try:
                p.write_short(*args)
            except Exception:
                pass

    def close(self) -> None:  # pragma: no cover - UI code
        for p in self.ports:
            try:
                p.close()
            except Exception:
                pass


# Current playhead position in eighth-note units for preview playback
PLAYHEAD_POS = 0.0
PLAYHEAD_LINE = None
PLAYHEAD_HEIGHT = 0


# Opciones de armonización disponibles.  También se pueden alternar dentro
# de la progresión escribiendo "(8)", "(15)", "(10)" o "(13)" antes del acorde.
ARMONIZACIONES = ["Octavas", "Doble octava", "Décimas", "Treceavas"]

# Variaciones disponibles para cada clave.  Para añadir más, simplemente
# amplía esta lista (por ejemplo, ["A", "B", "C", "D", "E"]).  Asegúrate de
# incluir los archivos MIDI correspondientes siguiendo el patrón
# ``<prefijo>_<variacion>.mid`` dentro de ``reference_midi_loops``.
VARIACIONES = ["A", "B", "C", "D"]

# Inversion options for salsa mode. Values map directly to the MIDI templates
INVERSIONES = [
    ("", "root"),
    ("", "third"),
    ("", "fifth"),
]
INV_ORDER = [v for _, v in INVERSIONES]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers and state used when parsing/applying styles.  ``get_modo`` and
# ``get_armon`` are simple functions returning the currently selected mode and
# harmonisation.  They are reassigned inside ``main`` to hook into the UI,
# allowing these utilities to be tested standalone.
# ---------------------------------------------------------------------------

current_inversions: List[str] = []
chord_styles: List[str] = []
chord_armos: List[str] = []

_current_modo = "Tradicional"
_current_armon = "Octavas"

def get_modo() -> str:
    return _current_modo

def get_armon() -> str:
    return _current_armon

def _parse_styles(text: str):
    """Delegate to ``style_utils.parse_styles`` with the current getters."""
    return style_utils.parse_styles(text, get_modo, get_armon)

def _apply_styles(base_text: str) -> str:
    """Return ``base_text`` unchanged via :func:`style_utils.apply_styles`."""
    return style_utils.apply_styles(base_text)

# ---------------------------------------------------------------------------
# Configuration of the available "claves".  Each entry defines the reference
# MIDI file and the rhythmic pattern to use.  Add new claves here in the
# future, following the same structure.
# ---------------------------------------------------------------------------
CLAVES = {
    "Clave 2-3": {
        "midi_prefix": "tradicional_2-3",
        "primer_bloque": [3, 4, 4, 3],
        "patron_repetido": [5, 4, 4, 3],
    },
    "Clave 3-2": {
        "midi_prefix": "tradicional_3-2",
        "primer_bloque": [3, 3, 5, 4],
        "patron_repetido": [4, 3, 5, 4],
    },
}

# ---------------------------------------------------------------------------
# Global counter for the generated montunos so output files have
# sequential names.
# ---------------------------------------------------------------------------
CONTADOR_MONTUNO = 1


class NumberedChordInput(Frame):
    """Compound widget with a chord Text area and measure numbers."""

    def __init__(self, master=None, **kw):
        super().__init__(master)
        self.numbers = Canvas(self, width=0, highlightthickness=0)
        self.numbers.pack(side="left", fill="y")
        opts = dict(kw)
        opts.setdefault("wrap", "word")
        opts.setdefault("height", 1)
        self.text = ChordAutocomplete(self, **opts)
        self.text.pack(side="left", fill="both", expand=True)
        try:
            line_h = tkfont.Font(font=self.text.cget("font")).metrics("linespace")
        except Exception:
            line_h = 20
        self._max_lines = max(1, int(CHORD_INPUT_MAX_HEIGHT / line_h))
        cur_h = int(self.text.cget("height"))
        if cur_h > self._max_lines:
            self.text.configure(height=self._max_lines)
        self.text.configure(
            yscrollcommand=self._on_text_scroll,
        )
        self.text.bind("<KeyRelease>", self._update_numbers, add=True)
        self.text.bind("<MouseWheel>", self._on_scroll, add=True)
        self.text.bind(
            "<Button-1>",
            lambda e: self.after_idle(self._update_numbers),
            add=True,
        )
        # Allow resizing the widget by middle-click dragging on its bottom edge
        self.bind("<ButtonPress-2>", self._start_resize)
        self.bind("<B2-Motion>", self._perform_resize)
        self.bind("<ButtonRelease-2>", self._end_resize)
        self.text.bind("<ButtonPress-2>", self._start_resize)
        self.text.bind("<B2-Motion>", self._perform_resize)
        self.text.bind("<ButtonRelease-2>", self._end_resize)
        self.numbers.bind("<ButtonPress-2>", self._start_resize)
        self.numbers.bind("<B2-Motion>", self._perform_resize)
        self.numbers.bind("<ButtonRelease-2>", self._end_resize)
        self._update_numbers()

        self._resizing = False
        self._resize_start_y = 0
        self._resize_start_h = int(self.text.cget("height"))

    def _on_text_scroll(self, *args):
        self.numbers.yview(*args)
        self._update_numbers()
        return "break"

    def _on_scroll(self, event):
        self.text.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._update_numbers()
        return "break"

    def _start_resize(self, event) -> None:
        margin = 5
        if event.y >= self.text.winfo_height() - margin:
            self._resizing = True
            self._resize_start_y = event.y_root
            self._resize_start_h = int(self.text.cget("height"))
            self.configure(cursor="sb_v_double_arrow")

    def _perform_resize(self, event) -> None:
        if not self._resizing:
            return
        try:
            line_h = tkfont.Font(font=self.text.cget("font")).metrics("linespace")
        except Exception:
            line_h = 20
        delta = int((event.y_root - self._resize_start_y) / line_h)
        new_h = max(1, self._resize_start_h + delta)
        new_h = min(new_h, self._max_lines)
        self.text.configure(height=new_h)
        self.after_idle(self._update_numbers)

    def _end_resize(self, event) -> None:
        if self._resizing:
            self._resizing = False
            self.configure(cursor="")

    def _update_numbers(self, event=None):
        font = self.text.cget("font")
        try:
            line_h = tkfont.Font(font=font).metrics("linespace")
        except Exception:
            line_h = 20
        text = self.text.get("1.0", "end-1c")
        lines = text.splitlines() or [""]
        self.numbers.delete("all")
        y = (len(lines)) * line_h + 2
        self.numbers.configure(scrollregion=(0, 0, 0, y))


def generar(
    status_var: StringVar,
    clave_var: StringVar,
    variacion_var: StringVar,
    inversion_var: StringVar,
    texto: Text,
    modo_combo: ComboBox,
    armon_combo: ComboBox,
    *,
    inversiones_custom: Optional[List[str]] = None,
    armonias_custom: Optional[List[str]] = None,
    return_pm: bool = False,
    output_path: Optional[Path] = None,
    override_text: Optional[str] = None,
    manual_edits: Optional[List[Dict]] = None,
    seed: Optional[int] = None,
    bpm: Optional[float] = None,
) -> Optional[pretty_midi.PrettyMIDI]:
    if seed is not None:
        import random
        old_state = random.getstate()
        random.seed(seed)
    else:
        old_state = None
    clave = clave_var.get()
    cfg = CLAVES.get(clave)
    if cfg is None:
        status_var.set(f"Clave no soportada: {clave}")
        return

    # Apply the rhythmic pattern for the selected clave
    modo_nombre = MODOS_INV.get(modo_combo.get(), modo_combo.get())

    for mod in (midi_utils_tradicional, midi_utils):
        mod.PRIMER_BLOQUE = cfg["primer_bloque"]
        mod.PATRON_REPETIDO = cfg["patron_repetido"]
        mod.PATRON_GRUPOS = mod.PRIMER_BLOQUE + mod.PATRON_REPETIDO * 3

    global CONTADOR_MONTUNO
    variacion = variacion_var.get()
    inversion = limpiar_inversion(inversion_var.get())

    progresion_texto = override_text if override_text is not None else texto.get("1.0", "end")
    progresion_texto = " ".join(progresion_texto.split())  # limpia espacios extra
    if not progresion_texto.strip():
        status_var.set("Ingresa una progresión de acordes")
        return

    try:
        asignaciones_all, _ = salsa.procesar_progresion_salsa(progresion_texto)
    except Exception as e:
        status_var.set(str(e))
        return

    num_acordes = len(asignaciones_all)
    estilos_loc = list(chord_styles)
    if len(estilos_loc) < num_acordes:
        estilos_loc.extend([modo_nombre] * (num_acordes - len(estilos_loc)))
    elif len(estilos_loc) > num_acordes:
        estilos_loc = estilos_loc[:num_acordes]

    if not asignaciones_all:
        status_var.set("Progresión vacía")
        return

    segmentos_info: List[Tuple[str, int, int]] = []
    start = 0
    modo_actual = estilos_loc[0]
    for i in range(1, num_acordes):
        if estilos_loc[i] != modo_actual:
            segmentos_info.append((modo_actual, start, i))
            start = i
            modo_actual = estilos_loc[i]
    segmentos_info.append((modo_actual, start, num_acordes))

    segmentos: List[Tuple[str, List[Tuple], int, List[int]]] = []
    for modo, a, b in segmentos_info:
        asign_abs = asignaciones_all[a:b]
        start_cor = asign_abs[0][1][0]
        rel = [
            (n, [i - start_cor for i in idxs], arm, inv)
            for n, idxs, arm, inv in asign_abs
        ]
        segmentos.append((modo, rel, start_cor, list(range(a, b))))

    modo_tag = (
        "salsa"
        if len({m for m, _, _, _ in segmentos}) == 1 and segmentos[0][0] == "Salsa"
        else "tradicional"
        if len({m for m, _, _, _ in segmentos}) == 1 and segmentos[0][0] == "Tradicional"
        else "mixto"
    )
    clave_tag = cfg["midi_prefix"].split("_")[1]
    output_dir = Path.home() / "Desktop" / "montunos"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output = output_dir / f"{modo_tag}_{clave_tag}_{CONTADOR_MONTUNO}.mid"
        CONTADOR_MONTUNO += 1
    else:
        output = Path(output_path)

    import pretty_midi

    notas_finales = []
    cur_bpm = 120.0
    inst_params = None
    max_cor = 0

    with TemporaryDirectory() as tmpdir:
        inv_idx = 0
        for idx, (modo_seg, asign_seg, start_cor, idxs_seg) in enumerate(segmentos):
            funcion = MODOS_DISPONIBLES.get(modo_seg)
            if funcion is None:
                status_var.set(f"Modo no soportado: {modo_seg}")
                return

            if modo_seg == "Salsa":
                sufijos = ['A', 'B', 'C', 'D']
                idx = (seed or 0) % 4
                sufijo = sufijos[idx]
                clave_tag_m = cfg["midi_prefix"].split("_")[1]
                inversion = limpiar_inversion(inversion_var.get())
                midi_ref_seg = BASE_DIR / "reference_midi_loops" / f"salsa_{clave_tag_m}_{inversion}_{sufijo}.mid"
                arg_extra = inversion

            else:
                midi_ref_seg = BASE_DIR / "reference_midi_loops" / f"{cfg['midi_prefix']}_{variacion}.mid"
                arg_extra = ARMONIZACION_INV.get(armon_combo.get(), armon_combo.get())

            if not midi_ref_seg.exists():
                status_var.set(f"No se encontró {midi_ref_seg}")
                return

            tmp_path = Path(tmpdir) / f"seg{idx}.mid"
            kwargs = {"asignaciones_custom": asign_seg}
            if modo_seg == "Salsa":
                if inversiones_custom is not None:
                    inv_seg = inversiones_custom[inv_idx : inv_idx + len(asign_seg)]
                    inv_idx += len(asign_seg)
                    if inv_seg:
                        kwargs["inversiones_manual"] = inv_seg
            else:
                if inversiones_custom is not None:
                    inv_seg = inversiones_custom[inv_idx : inv_idx + len(asign_seg)]
                    inv_idx += len(asign_seg)
                    if inv_seg:
                        suf_map = {"root": "1", "third": "3", "fifth": "5", "seventh": "7"}
                        asign_mod = []
                        for (nombre, idxs, arm, *rest), inv in zip(asign_seg, inv_seg):
                            if inv and inv != "root":
                                nombre = f"{nombre}/{suf_map.get(inv, '1')}"
                            asign_mod.append((nombre, idxs, arm))
                        asign_seg = asign_mod
                        kwargs["asignaciones_custom"] = asign_seg
                if armonias_custom is not None:
                    armon_seg = [armonias_custom[i] for i in idxs_seg]
                else:
                    armon_seg = [chord_armos[i] for i in idxs_seg]
                if modo_seg == "Tradicional":
                    kwargs["armonizaciones_custom"] = armon_seg
                    kwargs["aleatorio"] = True
            try:
                funcion(
                    "",
                    midi_ref_seg,
                    tmp_path,
                    arg_extra,
                    inicio_cor=start_cor,
                    return_pm=False,
                    **kwargs,
                )
            except Exception as e:
                status_var.set(str(e))
                return

            pm = pretty_midi.PrettyMIDI(str(tmp_path))
            seg_bpm = 120.0
            inst = pm.instruments[0]
            if inst_params is None:
                inst_params = (inst.program, inst.is_drum, inst.name)

            escala = 1.0
            grid_seg = 60.0 / seg_bpm / 2
            seg_cor = int(round(pm.get_end_time() / grid_seg))
            start = start_cor * (60.0 / cur_bpm / 2)
            for n in pm.instruments[0].notes:
                if n.pitch in (0, 21):
                    continue
                notas_finales.append(
                    pretty_midi.Note(
                        velocity=n.velocity,
                        pitch=n.pitch,
                        start=n.start * escala + start,
                        end=n.end * escala + start,
                    )
                )
            if start_cor + seg_cor > max_cor:
                max_cor = start_cor + seg_cor

    grid = 60.0 / cur_bpm / 2
    final_offset = max_cor * grid
    if final_offset > 0 and not return_pm:
        has_start = any(n.start <= 0 < n.end and n.pitch > 0 for n in notas_finales)
        has_end = any(
            n.pitch > 0 and n.start < final_offset and n.end > final_offset - grid for n in notas_finales
        )
        if not has_start:
            notas_finales.append(
                pretty_midi.Note(
                    velocity=1,
                    pitch=0,
                    start=0.0,
                    end=min(grid, final_offset),
                )
            )
        if not has_end:
            notas_finales.append(
                pretty_midi.Note(
                    velocity=1,
                    pitch=0,
                    start=max(0.0, final_offset - grid),
                    end=final_offset,
                )
            )

    pm_out = pretty_midi.PrettyMIDI()
    inst_out = pretty_midi.Instrument(
        program=inst_params[0], is_drum=inst_params[1], name=inst_params[2]
    )
    inst_out.notes = notas_finales
    pm_out.instruments.append(inst_out)
    if manual_edits:
        apply_manual_edits(pm_out, manual_edits)
    if return_pm:
        if old_state is not None:
            import random
            random.setstate(old_state)
        return pm_out
    try:
        pm_out.write(str(output))
        status_var.set(f"MIDI generado: {output}")
    except Exception as e:
        status_var.set(f"Error: {e}")
    if old_state is not None:
        import random
        random.setstate(old_state)
    return None


def main():
    root = CTk()
    ctk.set_appearance_mode("dark")
    root.title("Generador de Montunos")

    fonts_cfg = load_preferences()
    root.configure(fg_color=COLORS['bg'])

    # Load button icons and store references on the root to avoid garbage
    # collection.  The raw files are quite large (512x512) so they are
    # downscaled on load to avoid oversized buttons in the UI.
    icon_dir = Path(__file__).parent / "icons"

    # Icons are stored at a rather large resolution.  Downscale them on load
    # so buttons are not excessively big.  The default size was 24px which is a
    # bit large, reduce it by roughly 30% for a sleeker look.
    def _load_icon(name: str, size: int = 16) -> PhotoImage:
        img = PhotoImage(file=str(icon_dir / name))
        scale = max(1, int(min(img.width(), img.height()) / size))
        if scale > 1:
            img = img.subsample(scale, scale)
        return img

    root.icon_arrow_up = _load_icon("arrow_up.png")
    root.icon_arrow_down = _load_icon("arrow_down.png")
    root.icon_refresh = _load_icon("refresh.png")
    root.icon_play = _load_icon("play.png")
    root.icon_export = _load_icon("export.png")

    # ------------------------------------------------------------------
    # MIDI port detection
    # ------------------------------------------------------------------
    pygame.midi.init()
    port_map: Dict[str, int] = {}
    try:
        for i in range(pygame.midi.get_count()):
            info = pygame.midi.get_device_info(i)
            if info[3]:  # output
                name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
                port_map[name] = i
    finally:
        pygame.midi.quit()

    midi_port_var = StringVar(value="Desconectado")

    def _ensure_port():
        """Return a global MIDI output instance for the selected port."""
        global MIDI_PORT
        if MIDI_PORT is None:
            choice = midi_port_var.get()
            if choice == "Desconectado":
                return None
            try:
                pygame.midi.init()
                if choice == "Todas las salidas MIDI":
                    ports = [pygame.midi.Output(i) for i in port_map.values()]
                    MIDI_PORT = _MultiPort(ports)
                else:
                    idx = port_map.get(choice)
                    if idx is None:
                        return None
                    MIDI_PORT = pygame.midi.Output(idx)
            except Exception:
                pygame.midi.quit()
                MIDI_PORT = None
        return MIDI_PORT

    # Global fonts using centralized configuration
    general_font = get_general_font(root)
    entry_font = get_entry_font(root)
    measure_font = get_measure_font(root)
    if fonts_cfg:
        gf = fonts_cfg.get('general')
        ef = fonts_cfg.get('entry')
        mf = fonts_cfg.get('measure')
        if gf:
            general_font.configure(family=gf.get('family', general_font.cget('family')),
                                   size=gf.get('size', general_font.cget('size')))
        if ef:
            entry_font.configure(family=ef.get('family', entry_font.cget('family')),
                                 size=ef.get('size', entry_font.cget('size')))
        if mf:
            measure_font.configure(family=mf.get('family', measure_font.cget('family')),
                                  size=mf.get('size', measure_font.cget('size')))
    root.option_add("*Font", general_font)
    Label(
        root,
        text="©Jaime Jaramillo Arias - 2025",
        font=general_font,
        text_color=COLORS['fg'],
    ).place(relx=1.0, y=0, anchor="ne")

    clave_var = StringVar(value="Clave 2-3")
    variacion_var = StringVar(value=VARIACIONES[0])  # Default pattern
    inversion_var = StringVar(value=INVERSIONES[0][1])
    bpm_var = StringVar(value="120")
    midi_var = StringVar()
    status_var = StringVar()
    current_seed: Optional[int] = None

    pm_preview = None
    play_thread: Optional[threading.Thread] = None
    play_stop = threading.Event()

    def actualizar_midi() -> None:
        """Update the reference MIDI label according to the selected mode."""
        cfg = CLAVES[clave_var.get()]
        variacion = variacion_var.get()
        modo = MODOS_INV.get(modo_combo.get(), modo_combo.get())
        if modo == "Salsa":
            sufijos = ['A', 'B', 'C', 'D']
            import random
            idx = random.randint(0, 3)  # O usa tu lógica de seed si la tienes
            sufijo = sufijos[idx]
            clave_tag = cfg["midi_prefix"].split("_")[1]
            inversion = limpiar_inversion(inversion_var.get())
            path = str(BASE_DIR / "reference_midi_loops" / f"salsa_{clave_tag}_{inversion}_{sufijo}.mid")
        else:
            path = str(BASE_DIR / "reference_midi_loops" / f"{cfg['midi_prefix']}_{variacion}.mid")

        midi_var.set(path)
        actualizar_visualizacion(force_new_seed=True)
        _update_text_from_selections()

    # Ensure any change to these variables triggers a refresh.  Using
    # ``trace_add`` covers programmatic updates as well as user interactions.
    clave_var.trace_add("write", lambda *a: (_update_text_from_selections(), actualizar_midi()))
    variacion_var.trace_add("write", lambda *a: (_update_text_from_selections(), actualizar_midi()))
    inversion_var.trace_add("write", lambda *a: (_update_text_from_selections(), actualizar_midi()))

    def _calc_default_inversions(asig):
        return calc_default_inversions(
            asig,
            inversion_var.get,
            salsa.get_bass_pitch,
            salsa._ajustar_rango_flexible,
            salsa.seleccionar_inversion,
        )


    def _normalise_bars(text: str) -> str:
        return normalise_bars(text)

    def _update_text_from_selections(event=None):
        global _updating
        if _updating:
            return
        _updating = True


        prev_text = texto.text.get("1.0", "end-1c")
        acordes = [c for c in CHORD_RE.findall(prev_text)]

        # Ajusta las listas internas al número exacto de acordes
        while len(chord_styles) < len(acordes):
            chord_styles.append(get_modo())
        while len(chord_styles) > len(acordes):
            chord_styles.pop()
        while len(chord_armos) < len(acordes):
            chord_armos.append(get_armon())
        while len(chord_armos) > len(acordes):
            chord_armos.pop()
        while len(current_inversions) < len(acordes):
            current_inversions.append("root")
        while len(current_inversions) > len(acordes):
            current_inversions.pop()


        texto.text.edit_modified(False)


        def _clear():
            globals()["_updating"] = False
        root.after_idle(_clear)


    def _on_text_modified(event=None) -> None:
        """Trigger a visual refresh after any modification of the text widget."""
        # ``edit_modified(False)`` must be called to reset the internal flag,
        # otherwise this event would keep firing continuously.  Tkinter does not
        # reliably emit ``<KeyRelease>`` for every kind of edit (for example
        # pasting with the mouse), so we use ``<<Modified>>`` as a fallback.  If
        # the automatic update ever fails, the user can press the "Actualizar
        # Vista" button to force a refresh.
        global _updating
        if _updating:
            texto.text.edit_modified(False)
            return
        nonlocal last_text
        new_text = texto.text.get("1.0", "end-1c")
        if new_text != last_text:
            _push_text_change(last_text)
            last_text = new_text
        texto.text.edit_modified(False)
        root.after_idle(lambda: actualizar_visualizacion())

    def _on_cursor_move(event=None) -> None:
        idx = _cursor_chord_index()
        if idx is not None:
            actualizar_visualizacion(idx)
            root.after_idle(lambda i=idx: _scroll_to_chord(i))

    menu_widgets: List = []

    def _register_widget(w):
        """Register widgets created in ``_draw_piano_roll``.

        Every interactive control added to the piano roll must be passed
        through this helper so it gets stored in ``menu_widgets`` for
        proper destruction on the next redraw.
        """
        menu_widgets.append(w)
        return w
    note_items: List[int] = []
    note_info: List[Dict] = []
    chord_rects: List[Tuple[float, float]] = []
    manual_edits: List[Dict] = []
    dragging_idx: Optional[int] = None
    selected_notes: Set[int] = set()
    select_rect: Optional[int] = None
    select_start: Optional[Tuple[float, float]] = None
    last_text = ""

    def _record_modify(start: float, end: float, pitch: int) -> None:
        for ed in manual_edits:
            if ed.get("type") == "modify" and abs(ed["start"] - start) < 1e-6 and abs(ed["end"] - end) < 1e-6:
                ed["pitch"] = pitch
                return
        manual_edits.append({"type": "modify", "start": start, "end": end, "pitch": pitch})

    def _record_add(start: float, end: float, pitch: int) -> None:
        manual_edits.append({"type": "add", "start": start, "end": end, "pitch": pitch})

    def _record_delete(start: float, end: float, pitch: int) -> None:
        manual_edits.append({"type": "delete", "start": start, "end": end, "pitch": pitch})

    class _Tooltip:
        def __init__(self):
            self.tip = None

        def show(self, text, x, y):
            try:
                self.hide()
                self.tip = Toplevel(root)
                self.tip.wm_overrideredirect(True)
                Label(
                    self.tip,
                    text=text,
                    fg_color="#ffffe0",
                    text_color="black",
                    corner_radius=5,
                ).pack(padx=2, pady=2)
                self.tip.geometry(f"+{int(x)}+{int(y)}")
            except Exception as e:
                status_var.set(f"Error tooltip: {e}")

        def hide(self):
            if self.tip is not None:
                self.tip.destroy()
                self.tip = None

    tooltip = _Tooltip()

    def _clean_tokens(txt: str) -> str:
        return clean_tokens(txt)

    def _cursor_chord_index() -> Optional[int]:
        text = texto.text.get("1.0", "end-1c")
        caret = len(texto.text.get("1.0", "insert"))
        cleaned = _clean_tokens(text)
        idx = 0
        for m in CHORD_CURSOR_RE.finditer(cleaned):
            if m.start(1) <= caret <= m.end(1):
                return idx
            idx += 1
        return None

    def _scroll_to_chord(idx: int) -> None:
        if idx < 0 or idx >= len(chord_rects):
            return
        x1, x2 = chord_rects[idx]
        view_w = canvas.winfo_width()
        mid = (x1 + x2) / 2
        target = max(0, mid - view_w / 2)
        width = canvas.scrollregion[2]
        canvas.xview_moveto(target / max(1, width))

    def _capture_state() -> dict:
        return {
            "text": texto.text.get("1.0", "end-1c"),
            "modo": modo_combo.get(),
            "armon": armon_combo.get(),
            "clave": clave_var.get(),
            "variacion": variacion_var.get(),
            "inversion": limpiar_inversion(inversion_var.get()),
            "bpm": bpm_var.get(),
            "manual_edits": [dict(ed) for ed in manual_edits],
        }

    def _push_state() -> None:
        if _suppress_undo:
            return
        UNDO_STACK.append(_capture_state())
        REDO_STACK.clear()

    def _restore_state(st: dict) -> None:
        nonlocal last_text
        global _suppress_undo, _updating
        _suppress_undo = True
        _updating = True
        texto.text.delete("1.0", "end")
        texto.text.insert("1.0", st["text"])
        modo_combo.set(st["modo"])
        armon_combo.set(st["armon"])
        clave_var.set(st["clave"])
        variacion_var.set(st["variacion"])
        inversion_var.set(limpiar_inversion(st["inversion"]))
        bpm_var.set(st.get("bpm", bpm_var.get()))
        manual_edits[:] = [dict(ed) for ed in st["manual_edits"]]
        last_text = st["text"]
        _suppress_undo = False
        def _clear():
            globals()["_updating"] = False
        root.after_idle(_clear)
        actualizar_visualizacion()

    def _push_text_change(prev: str) -> None:
        if _suppress_undo:
            return
        st = _capture_state()
        st["text"] = prev
        UNDO_STACK.append(st)

    def undo(event=None) -> None:
        if not UNDO_STACK:
            return
        current = _capture_state()
        st = UNDO_STACK.pop()
        REDO_STACK.append(current)
        _restore_state(st)

    def redo(event=None) -> None:
        if not REDO_STACK:
            return
        current = _capture_state()
        st = REDO_STACK.pop()
        UNDO_STACK.append(current)
        _restore_state(st)

    def _shift_all_inversions(delta: int) -> None:
        """Shift every chord inversion by ``delta`` steps circularly."""
        if not current_inversions:
            return
        _push_state()
        for i, inv in enumerate(current_inversions):
            if inv in INV_ORDER:
                idx = INV_ORDER.index(inv)
                current_inversions[i] = INV_ORDER[(idx + delta) % len(INV_ORDER)]
        if current_inversions:
            inversion_var.set(current_inversions[0])
        _update_text_from_selections()
        actualizar_visualizacion()

    def _draw_piano_roll(pm, asignaciones, highlight_idx=None):
        Y_OFFSET = 32  # Puedes aumentar este número para más espacio arriba

        for w in menu_widgets:
            try:
                w.destroy()
            except Exception:
                pass
        menu_widgets.clear()
        # Remove any previous canvas windows tagged as menus just in case a
        # widget was not registered correctly in ``menu_widgets``.
        canvas.delete("menu")
        # Ensure all previous menu windows were removed.  Some widgets may not
        # get properly registered under the "menu" tag when created on certain
        # platforms, so gracefully clean up any leftovers without raising an
        # exception that would prevent the piano roll from updating.
        for item in canvas.find_withtag("menu"):
            try:
                canvas.delete(item)
            except Exception:
                pass
        # Asegura que no queden widgets huérfanos en el canvas en caso de que
        # alguno no se hubiese añadido a ``menu_widgets`` correctamente.
        for child in canvas.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass
        canvas.delete("all")
        note_items.clear()
        note_info.clear()
        chord_rects.clear()
        selected_notes.clear()
        def _update_selection_highlight() -> None:
            for idx, item in enumerate(note_items):
                color = COLORS['note_selected'] if idx in selected_notes else COLORS['note']
                canvas.itemconfigure(item, fill=color)
        drag_start_y = 0.0
        drag_start_x = 0.0
        orig_pitches: Dict[int, int] = {}
        orig_times: Dict[int, Tuple[float, float]] = {}
        notes = [n for n in pm.instruments[0].notes if n.pitch > 0]
        if not notes:
            return
        tmax = max(n.end for n in notes)
        pmin = min(n.pitch for n in notes)
        pmax = max(n.pitch for n in notes)
        step = 5
        note_height = (pmax - pmin + 1) * step + 20
        global PLAYHEAD_HEIGHT
        PLAYHEAD_HEIGHT = note_height + Y_OFFSET - 10
        bpm = float(bpm_var.get() or 120)
        grid = 60.0 / bpm / 2
        num_eighths = int(round(tmax / grid)) + 1
        width = num_eighths * CELL_WIDTH + 20
        height = note_height + 90 + Y_OFFSET - 10
        canvas.configure(scrollregion=(0, 0, width, height), height=height)

        # Draw playhead line at the current position
        global PLAYHEAD_LINE
        x_ph = PLAYHEAD_POS * CELL_WIDTH
        if PLAYHEAD_LINE is not None:
            try:
                canvas.delete(PLAYHEAD_LINE)
            except Exception:
                pass
        PLAYHEAD_LINE = canvas.create_line(
            x_ph,
            Y_OFFSET - 12,
            x_ph,
            PLAYHEAD_HEIGHT,
            fill="red",
            width=2,
            tags="playhead",
        )

        # Piano key background
        black_keys = {1, 3, 6, 8, 10}
        for p in range(pmax, pmin - 1, -1):
            y1 = (pmax - p) * step + Y_OFFSET
            y2 = (pmax - p + 1) * step + Y_OFFSET
            color = "#333" if p % 12 in black_keys else "#555"
            rect = canvas.create_rectangle(0, y1, width, y2, fill=color, outline="")
            if p % 12 == 0:
                canvas.create_line(0, y2, width, y2, fill="#aaa", width=2)
            note_name = pretty_midi.note_number_to_name(p)
            def show_key(evt, n=note_name):
                tooltip.show(n, evt.x_root + 10, evt.y_root + 10)
            canvas.tag_bind(rect, "<Enter>", show_key)
            canvas.tag_bind(rect, "<Leave>", lambda e: tooltip.hide())

        for i in range(num_eighths + 1):
            x = i * CELL_WIDTH
            canvas.create_line(x, Y_OFFSET - 12, x, note_height + Y_OFFSET - 10, fill="#666")
        compases = (num_eighths + 7) // 8
        for i in range(compases):
            x = i * 8 * CELL_WIDTH
            canvas.create_line(x, Y_OFFSET - 12, x, note_height + Y_OFFSET - 10, fill="#aaa", width=2)
            canvas.create_text(
                x + 2,
                Y_OFFSET - 10,
                text=str(i + 1),
                anchor="nw",
                fill=COLORS['measure'],
                font=(measure_font.cget("family"), measure_font.cget("size")),
            )

        # Draw notes
        def _round_rect_points(x1, y1, x2, y2, r=4):
            return [
                x1 + r, y1,
                x2 - r, y1,
                x2, y1,
                x2, y1 + r,
                x2, y2 - r,
                x2, y2,
                x2 - r, y2,
                x1 + r, y2,
                x1, y2,
                x1, y2 - r,
                x1, y1 + r,
                x1, y1,
            ]

        def _round_rect(x1, y1, x2, y2, r=4, **kwargs):
            return canvas.create_polygon(
                _round_rect_points(x1, y1, x2, y2, r),
                smooth=True,
                splinesteps=36,
                **kwargs,
            )
        def _play_note(pitch: int, dur: float = 0.2) -> None:
            port = _ensure_port()
            if port is None:
                return
            try:
                port.note_on(pitch, 100)
                root.after(int(dur * 1000), lambda p=pitch: port.note_off(p, 100))
            except Exception:
                pass

        def _bind_note(item: int, note_obj: pretty_midi.Note, idx_n: int) -> None:
            def start_drag(event):
                nonlocal dragging_idx, select_start, drag_start_y, drag_start_x
                nonlocal orig_pitches, orig_times
                _push_state()
                dragging_idx = idx_n
                if idx_n not in selected_notes:
                    selected_notes.clear()
                    selected_notes.add(idx_n)
                    _update_selection_highlight()
                orig_pitches = {i: note_info[i]["note"].pitch for i in selected_notes}
                orig_times = {i: (note_info[i]["note"].start, note_info[i]["note"].end) for i in selected_notes}
                drag_start_y = canvas.canvasy(event.y)
                drag_start_x = canvas.canvasx(event.x)
                _play_note(note_obj.pitch)
                canvas.focus_set()
                return "break"

            def drag(event):
                if dragging_idx != idx_n:
                    return
                y = canvas.canvasy(event.y)
                x = canvas.canvasx(event.x)
                delta_p = int((drag_start_y - y) // 5)
                delta_x = int(round((x - drag_start_x) / CELL_WIDTH))
                for i, p0 in orig_pitches.items():
                    n = note_info[i]["note"]
                    new_pitch = max(pmin, min(pmax, p0 + delta_p))
                    start0, end0 = orig_times[i]
                    new_start = max(0.0, start0 + delta_x * grid)
                    new_end = new_start + (end0 - start0)
                    changed = False
                    if new_pitch != n.pitch:
                        n.pitch = new_pitch
                        changed = True
                    if abs(new_start - n.start) > 1e-6 or abs(new_end - n.end) > 1e-6:
                        n.start = new_start
                        n.end = new_end
                        changed = True
                    if changed:
                        x1 = n.start / grid * CELL_WIDTH
                        x2 = n.end / grid * CELL_WIDTH
                        y1 = (pmax - n.pitch) * step + Y_OFFSET
                        y2 = (pmax - n.pitch + 1) * step + Y_OFFSET
                        canvas.coords(
                            note_items[i],
                            *_round_rect_points(x1, y1, x2, y2, r=4),
                        )
                _update_selection_highlight()
                return "break"

            def end_drag(event):
                nonlocal dragging_idx
                if dragging_idx != idx_n:
                    return
                dragging_idx = None
                for i in selected_notes:
                    n = note_info[i]["note"]
                    start0, end0 = orig_times[i]
                    pitch0 = orig_pitches[i]
                    if (
                        abs(start0 - n.start) > 1e-6
                        or abs(end0 - n.end) > 1e-6
                        or pitch0 != n.pitch
                    ):
                        _record_delete(start0, end0, pitch0)
                        _record_add(n.start, n.end, n.pitch)
                _play_note(note_obj.pitch)
                return "break"

            canvas.tag_bind(item, "<ButtonPress-1>", start_drag)
            canvas.tag_bind(item, "<B1-Motion>", drag)
            canvas.tag_bind(item, "<ButtonRelease-1>", end_drag)
            def del_note(evt):
                _push_state()
                for i in sorted(selected_notes or {idx_n}, reverse=True):
                    n = note_info[i]["note"]
                    _record_delete(n.start, n.end, n.pitch)
                    try:
                        pm.instruments[0].notes.remove(n)
                    except ValueError:
                        pass
                actualizar_visualizacion()
                return "break"
            canvas.tag_bind(item, "<Delete>", del_note)
            canvas.tag_bind(item, "<BackSpace>", del_note)
            canvas.tag_bind(item, "<Button-3>", del_note)
            canvas.bind("<Delete>", del_note, add=True)
            canvas.bind("<BackSpace>", del_note, add=True)
            def on_enter(evt):
                nombre = pretty_midi.note_number_to_name(note_obj.pitch)
                dur = note_obj.end - note_obj.start
                tooltip.show(
                    f"{nombre}\n{dur:.2f}s",
                    evt.x_root + 10,
                    evt.y_root + 10,
                )

            def on_leave(evt):
                tooltip.hide()

            canvas.tag_bind(item, "<Enter>", on_enter)
            canvas.tag_bind(item, "<Leave>", on_leave)

        for idx_n, n in enumerate(notes):
            x1 = n.start / grid * CELL_WIDTH
            x2 = n.end / grid * CELL_WIDTH
            y1 = (pmax - n.pitch) * step + Y_OFFSET
            y2 = (pmax - n.pitch + 1) * step + Y_OFFSET
            item = _round_rect(x1, y1, x2, y2, r=4, fill=COLORS['note'], outline="#e0e0e0")
            note_items.append(item)
            note_info.append({"note": n})
            _bind_note(item, n, idx_n)

        _update_selection_highlight()

        def start_select(evt):
            cur = canvas.find_withtag("current")
            if cur and cur[0] in note_items:
                return
            nonlocal select_rect, select_start
            select_start = (canvas.canvasx(evt.x), canvas.canvasy(evt.y))
            if select_rect:
                canvas.delete(select_rect)
            select_rect = canvas.create_rectangle(
                *select_start, *select_start, outline=COLORS['accent'], dash=(2, 2)
            )
            selected_notes.clear()
            _update_selection_highlight()

        def drag_select(evt):
            if select_start is None:
                return
            x = canvas.canvasx(evt.x)
            y = canvas.canvasy(event.y)
            canvas.coords(select_rect, select_start[0], select_start[1], x, y)

        def end_select(evt):
            nonlocal select_rect, select_start
            if select_start is None:
                return
            x0, y0 = select_start
            x1 = canvas.canvasx(evt.x)
            y1 = canvas.canvasy(evt.y)
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            canvas.delete(select_rect)
            select_rect = None
            select_start = None
            selected_notes.clear()
            for idx, item_i in enumerate(note_items):
                ix1, iy1, ix2, iy2 = canvas.coords(item_i)
                if not (ix2 < x0 or ix1 > x1 or iy2 < y0 or iy1 > y1):
                    selected_notes.add(idx)
            _update_selection_highlight()

        canvas.bind("<ButtonPress-1>", start_select)
        canvas.bind("<B1-Motion>", drag_select)
        canvas.bind("<ButtonRelease-1>", end_select)

        def add_note(event):
            try:
                _push_state()
                x = canvas.canvasx(event.x)
                y = canvas.canvasy(event.y)
                pitch = pmax - int((y - Y_OFFSET) // step)
                start = (x // CELL_WIDTH) * grid
                end = start + grid
                pitch = max(pmin, min(pmax, pitch))
                new_note = pretty_midi.Note(
                    velocity=100, pitch=pitch, start=start, end=end
                )
                pm.instruments[0].notes.append(new_note)
                _record_add(start, end, pitch)
                actualizar_visualizacion()
                _play_note(new_note.pitch)
            except Exception as e:
                status_var.set(f"Error al crear nota: {e}")

        canvas.bind("<Double-Button-1>", add_note)

        def set_playhead(event):
            global PLAYHEAD_POS
            PLAYHEAD_POS = canvas.canvasx(event.x) / CELL_WIDTH
            x = PLAYHEAD_POS * CELL_WIDTH
            if PLAYHEAD_LINE is not None:
                canvas.coords(PLAYHEAD_LINE, x, Y_OFFSET - 12, x, PLAYHEAD_HEIGHT)

        canvas.bind("<Button-3>", set_playhead)

        label_map = {v: "" for _, v in INVERSIONES}
        rev_map = {t: v for t, v in INVERSIONES}
        arm_map = {v: ARMONIZACION_LABELS[v] for v in ARMONIZACIONES}
        arm_rev = {v: k for k, v in arm_map.items()}
        base_y = note_height + Y_OFFSET + 10
        sep = 25
        for idx, data in enumerate(asignaciones):
            cif, idxs, arm = data[:3]
            start_idx = idxs[0]
            end_idx = idxs[-1] + 1
            x1 = start_idx * CELL_WIDTH
            x2 = end_idx * CELL_WIDTH
            chord_rects.append((x1, x2))
            xm = x1
            canvas.create_text(
                xm + 2,
                18,
                text=cif,
                anchor="sw",
                fill="#ffcc33",
                font=(general_font.cget("family"), general_font.cget("size"), "bold"),
            )
            var_inv = StringVar(value=label_map[current_inversions[idx]])

            def _shift_inv(delta: int, i: int = idx) -> None:
                _push_state()
                cur = current_inversions[i]
                if cur in INV_ORDER:
                    idx_inv = INV_ORDER.index(cur)
                    current_inversions[i] = INV_ORDER[(idx_inv + delta) % len(INV_ORDER)]
                    var_inv.set(label_map[current_inversions[i]])
                actualizar_visualizacion()

            # All widgets go through _register_widget to ensure proper cleanup
            frm_inv = _register_widget(Frame(canvas))
            lbl_inv = Label(frm_inv, textvariable=var_inv, font=general_font, text_color=COLORS['fg'])
            btn_up = Button(frm_inv, image=root.icon_arrow_up, text="", width=20, command=lambda i=idx: _shift_inv(1, i))
            btn_down = Button(frm_inv, image=root.icon_arrow_down, text="", width=20, command=lambda i=idx: _shift_inv(-1, i))
            lbl_inv.pack(side="left")
            btn_up.pack(side="left", padx=2)
            btn_down.pack(side="left")
            
            canvas.create_window(xm, base_y, window=frm_inv, anchor="nw", tags="menu")
            var_style = StringVar(value=MODOS_LABELS[chord_styles[idx]])
            def cb_style(choice, i=idx):
                _push_state()
                nuevo_modo = MODOS_INV[choice]
                n = len(chord_styles)
                # Buscar el siguiente acorde donde el usuario haya hecho un cambio manual (es decir, un valor distinto al actual antes de cambiar)
                next_change = n
                for j in range(i + 1, n):
                    if chord_styles[j] != chord_styles[i]:
                        next_change = j
                        break
                # Propagar el modo desde el acorde i hasta el siguiente punto de cambio (o hasta el final)
                for j in range(i, next_change):
                    chord_styles[j] = nuevo_modo
                    # Puedes adaptar esto si quieres diferentes armonizaciones por modo
                    if nuevo_modo == "Salsa":
                        chord_armos[j] = "Octavas"
                    elif nuevo_modo == "Tradicional":
                        chord_armos[j] = "Octavas"
                actualizar_visualizacion()


            # Same here: register the option menu so it is cleaned up later
            om_style = _register_widget(
                OptionMenu(
                    canvas,
                    values=list(MODOS_LABELS.values()),
                    variable=var_style,
                    command=cb_style,
                    width=80,
                )
            )
            canvas.create_window(
                xm, base_y + sep, window=om_style, anchor="nw", tags="menu"
            )



            if chord_styles[idx] == "Tradicional":
                var_arm = StringVar(value=arm_map[chord_armos[idx]])
                def cb_arm(choice, i=idx):
                    _push_state()
                    nueva_arm = arm_rev[choice]
                    if chord_styles[i] == "Tradicional":
                        chord_armos[i] = nueva_arm
                    actualizar_visualizacion()

                # Arm menus are dynamic; ensure they are registered as well
                om_arm = _register_widget(
                    OptionMenu(
                        canvas,
                        values=list(arm_map.values()),
                        variable=var_arm,
                        command=cb_arm,
                        width=80,
                    )
                )
                canvas.create_window(xm, base_y + 2*sep, window=om_arm, anchor="nw", tags="menu")
                canvas.tag_raise(PLAYHEAD_LINE)


    def actualizar_visualizacion(highlight_idx=None, *, force_new_seed=False) -> None:
        nonlocal pm_preview, current_seed
        import random
        if force_new_seed or current_seed is None:
            current_seed = random.randint(0, 2**32 - 1)
        prog = texto.text.get("1.0", "end").strip()
        if not prog:
            return
        try:
            print("\n[DEBUG] Texto original que va al parser:", repr(prog))
            asign, _ = salsa.procesar_progresion_salsa(prog)
        except Exception as e:
            status_var.set(str(e))
            return

        num_chords = len(asign)
        while len(chord_styles) < num_chords:
            chord_styles.append(get_modo())
        while len(chord_styles) > num_chords:
            chord_styles.pop()

        while len(chord_armos) < num_chords:
            chord_armos.append(get_armon())
        while len(chord_armos) > num_chords:
            chord_armos.pop()

        default_inv = _calc_default_inversions(asign)
        while len(current_inversions) < num_chords:
            current_inversions.append(default_inv[len(current_inversions)])
        while len(current_inversions) > num_chords:
            current_inversions.pop()

        prog_mod = _normalise_bars(prog)
        print("[DEBUG] Texto que va a generar/override_text:", repr(prog_mod))

        pm_preview = generar(
            status_var,
            clave_var,
            variacion_var,
            inversion_var,
            texto.text,
            modo_combo,
            armon_combo,
            inversiones_custom=current_inversions,
            armonias_custom=chord_armos,
            return_pm=True,
            output_path=None,
            override_text=prog_mod,
            manual_edits=manual_edits,
            seed=current_seed,
            bpm=float(bpm_var.get() or 120),
        )
        if pm_preview is not None:
            _draw_piano_roll(pm_preview, asign, highlight_idx)
            status_var.set("Vista actualizada")

    def reproducir_preview() -> None:
        import time
        import mido
        from tempfile import TemporaryDirectory

        # Asegúrate de que pm_preview esté generado
        if pm_preview is None:
            actualizar_visualizacion()
        if pm_preview is None:
            return

        if not hasattr(root, "midi_preview_data"):
            root.midi_preview_data = {}
        midi_state = root.midi_preview_data
        midi_state.clear()
        midi_state["stop"] = False

        def play_preview(pm):
            midi_bpm = 120.0
            preview_bpm = float(bpm_var.get() or 120)
            scale = 1.0
            grid = 60.0 / preview_bpm / 2
            start_time = PLAYHEAD_POS * grid

            with TemporaryDirectory() as td:
                path = Path(td) / "prev.mid"
                pm.write(str(path))
                mid = mido.MidiFile(str(path))

                messages = []
                cur = 0.0
                for msg in mid:
                    cur += msg.time * scale
                    messages.append((cur, msg))
                midi_state["messages"] = messages
                midi_state["current_idx"] = 0
                midi_state["start_wall"] = time.time()
                midi_state["start_time"] = start_time
                midi_state["grid"] = grid

                port = _ensure_port()
                if port is None:
                    status_var.set("No hay dispositivo MIDI disponible")
                    return

                def process_next():
                    if midi_state.get("stop"):
                        try:
                            port.write_short(0xB0, 0x7B, 0)
                        except Exception:
                            pass
                        status_var.set("Playback detenido.")
                        return

                    idx = midi_state["current_idx"]
                    if idx >= len(messages):
                        try:
                            port.write_short(0xB0, 0x7B, 0)
                        except Exception:
                            pass
                        status_var.set("Playback finalizado.")
                        return

                    cur, msg = messages[idx]
                    elapsed = time.time() - midi_state["start_wall"]
                    wait = (cur - start_time) - elapsed
                    if wait > 0:
                        root.after(int(wait * 1000), process_next)
                        return

                    if not msg.is_meta:
                        port.write_short(*msg.bytes())
                    pos = (cur - start_time) / grid
                    x = pos * CELL_WIDTH
                    root.after(0, lambda px=x: canvas.coords(PLAYHEAD_LINE, px, 20, px, PLAYHEAD_HEIGHT))
                    midi_state["current_idx"] += 1
                    root.after(1, process_next)

                midi_state["stop"] = False
                midi_state["current_idx"] = 0
                midi_state["start_wall"] = time.time()
                process_next()
                status_var.set("Preview iniciado.")

        play_preview(pm_preview)

    def detener_preview():
        if hasattr(root, "midi_preview_data"):
            root.midi_preview_data["stop"] = True

    def mover_playhead(delta: int) -> None:
        global PLAYHEAD_POS
        PLAYHEAD_POS = max(0.0, PLAYHEAD_POS + delta)
        x = PLAYHEAD_POS * CELL_WIDTH
        canvas.coords(PLAYHEAD_LINE, x, 20, x, PLAYHEAD_HEIGHT)


    Label(root, text="Clave:", font=general_font, text_color=COLORS['fg']).pack(pady=(5, 0))
    frame_clave = Frame(root)
    frame_clave.pack()
    for nombre in CLAVES:
        Radiobutton(
            frame_clave,
            text=nombre,
            variable=clave_var,
            value=nombre,
            font=general_font,
            text_color=COLORS['fg'],
            command=lambda n=nombre: (_push_state(), clave_var.set(n), actualizar_midi()),
        ).pack(side="left", padx=5)

    bpm_frame = Frame(root)
    bpm_frame.pack(pady=(5, 0))
    Label(bpm_frame, text="BPM:", font=general_font, text_color=COLORS['fg']).pack(side="left", padx=(0, 5))
    bpm_entry = ctk.CTkEntry(bpm_frame, textvariable=bpm_var, width=60)
    bpm_entry.pack(side="left")

    def _bpm_start_drag(event):
        bpm_entry._drag_start_y = event.y_root
        try:
            bpm_entry._drag_start_v = float(bpm_var.get())
        except Exception:
            bpm_entry._drag_start_v = 120.0
        bpm_entry.configure(cursor="sb_v_double_arrow")

    def _bpm_drag(event):
        if not hasattr(bpm_entry, "_drag_start_y"):
            return
        delta = bpm_entry._drag_start_y - event.y_root
        val = max(20, bpm_entry._drag_start_v + delta)
        bpm_var.set(str(int(val)))

    def _bpm_end_drag(event):
        bpm_entry.configure(cursor="")
        actualizar_visualizacion()

    bpm_entry.bind("<ButtonPress-1>", _bpm_start_drag)
    bpm_entry.bind("<B1-Motion>", _bpm_drag)
    bpm_entry.bind("<ButtonRelease-1>", _bpm_end_drag)
    bpm_var.trace_add("write", lambda *a: actualizar_visualizacion())

    prog_label = Label(root, text="Progresión de acordes:", font=general_font, text_color=COLORS['fg'])
    prog_label.pack(anchor="w")

    # Select a large, handwriting-style font for the chord input.  The
    # ``MuseJazz.ttf`` file is bundled with the project so the font is
    # available even if not installed system-wide.  If it cannot be loaded,
    # fall back to another handwriting font.
    prog_label.configure(font=general_font)

    def _apply_widget_styles(w):
        if isinstance(w, Frame):
            w.configure(fg_color=COLORS['bg'])
        if isinstance(w, Canvas):
            w.configure(bg=COLORS['pianoroll_bg'])
        if isinstance(w, Button):
            w.configure(
                fg_color=COLORS['button'],
                hover_color=COLORS['accent'],
                text_color=COLORS['fg'],
                font=general_font,
            )
        elif isinstance(w, Radiobutton):
            w.configure(text_color=COLORS['fg'], fg_color=COLORS['button'], font=general_font)
        elif isinstance(w, (Label, OptionMenu, ComboBox)):
            opts = {
                'text_color': COLORS['fg'],
                'font': general_font,
            }
            if isinstance(w, (OptionMenu, ComboBox)):
                w.configure(fg_color=COLORS['button'], button_color=COLORS['button'], dropdown_fg_color=COLORS['bg'], dropdown_text_color=COLORS['fg'], dropdown_font=general_font, **opts)
            else:
                w.configure(**opts)
        for child in getattr(w, 'winfo_children', lambda: [])():
            _apply_widget_styles(child)

    def apply_global_styles():
        root.configure(fg_color=COLORS['bg'])
        _apply_widget_styles(root)

    def open_style_editor():
        win = Toplevel(root)
        win.title("Personalizar")
        color_vars = {}
        row = 0
        for key, label_txt in [
            ("bg", "Fondo"),
            ("fg", "Texto"),
            ("button", "Botones"),
            ("accent", "Resaltado"),
            ("measure", "Color compases"),
            ("pianoroll_bg", "Fondo piano roll"),
            ("note", "Nota"),
            ("note_selected", "Nota seleccionada"),
        ]:
            Label(win, text=label_txt).grid(row=row, column=0, padx=5, pady=5, sticky="e")
            var = StringVar(value=COLORS[key])

            def _choose(k=key, v=var):
                c = colorchooser.askcolor(color=v.get())[1]
                if c:
                    v.set(c)

            btn = Button(win, text="", width=40, command=_choose,
                         fg_color=var.get(), hover_color=var.get())
            btn.grid(row=row, column=1, padx=5, pady=5)
            var.trace_add("write", lambda *a, b=btn, v=var: b.configure(fg_color=v.get(), hover_color=v.get()))
            color_vars[key] = var
            row += 1
        fonts = sorted(tkfont.families(root))
        Label(win, text="Fuente general").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        font_general_var = StringVar(value=general_font.cget("family"))
        font_general_combo = ComboBox(win, variable=font_general_var, values=fonts)
        font_general_combo.grid(row=row, column=1, padx=5, pady=5)
        row += 1
        Label(win, text="Tamaño general").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        size_general_entry = ctk.CTkEntry(win)
        size_general_entry.insert(0, str(general_font.cget("size")))
        size_general_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1
        Label(win, text="Fuente acordes").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        font_entry_var = StringVar(value=entry_font.cget("family"))
        font_entry_combo = ComboBox(win, variable=font_entry_var, values=fonts)
        font_entry_combo.grid(row=row, column=1, padx=5, pady=5)
        row += 1
        Label(win, text="Tamaño acordes").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        size_entry_entry = ctk.CTkEntry(win)
        size_entry_entry.insert(0, str(entry_font.cget("size")))
        size_entry_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1
        Label(win, text="Fuente compases").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        font_measure_var = StringVar(value=measure_font.cget("family"))
        font_measure_combo = ComboBox(win, variable=font_measure_var, values=fonts)
        font_measure_combo.grid(row=row, column=1, padx=5, pady=5)
        row += 1
        Label(win, text="Tamaño compases").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        size_measure_entry = ctk.CTkEntry(win)
        size_measure_entry.insert(0, str(measure_font.cget("size")))
        size_measure_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        def apply_changes():
            for k, v in color_vars.items():
                COLORS[k] = v.get().strip() or COLORS[k]
            general_font.configure(
                family=font_general_var.get().strip() or general_font.cget("family"),
                size=int(size_general_entry.get().strip() or general_font.cget("size")),
            )
            entry_font.configure(
                family=font_entry_var.get().strip() or entry_font.cget("family"),
                size=int(size_entry_entry.get().strip() or entry_font.cget("size")),
            )
            measure_font.configure(
                family=font_measure_var.get().strip() or measure_font.cget("family"),
                size=int(size_measure_entry.get().strip() or measure_font.cget("size")),
            )
            apply_global_styles()
            actualizar_visualizacion()

        def save_and_close():
            apply_changes()
            save_preferences(
                {
                    'general': {
                        'family': general_font.cget('family'),
                        'size': general_font.cget('size'),
                    },
                    'entry': {
                        'family': entry_font.cget('family'),
                        'size': entry_font.cget('size'),
                    },
                    'measure': {
                        'family': measure_font.cget('family'),
                        'size': measure_font.cget('size'),
                    },
                }
            )
            win.destroy()

        Button(win, text="Aplicar", command=apply_changes).grid(row=row, column=0, pady=10)
        Button(win, text="Guardar", command=save_and_close).grid(row=row, column=1, pady=10)

    texto = NumberedChordInput(root, width=40, font=entry_font, undo=True)
    texto.pack(fill="x", padx=5)
    texto.text.bind("<KeyRelease>", lambda e: _on_cursor_move(), add=True)
    texto.text.bind("<ButtonRelease-1>", lambda e: _on_cursor_move(), add=True)
    # ``<<Modified>>`` captures changes like paste operations that do not emit a
    # key event.  ``edit_modified(False)`` clears the flag after handling.
    texto.text.bind("<<Modified>>", _on_text_modified)
    texto.text.edit_modified(False)
    last_text = texto.text.get("1.0", "end-1c")

    # Variation options removed from the main interface

    inversion_label = Label(root, text="Inversión:", font=general_font, text_color=COLORS['fg'])
    frame_inversion = Frame(root)
    for txt, val in INVERSIONES:
        Radiobutton(
            frame_inversion,
            text=txt,
            variable=inversion_var,
            value=val,
            font=general_font,
            text_color=COLORS['fg'],
            command=lambda v=val: (_push_state(), inversion_var.set(v), actualizar_midi()),
        ).pack(anchor="w")


    def actualizar_armonizacion() -> None:
        """Refresh preview when global settings change."""
        actualizar_midi()
        actualizar_visualizacion()
        _update_text_from_selections()

    # Default mode combobox kept hidden for internal use
    modo_var = StringVar(value=MODOS_LABELS["Tradicional"])
    modo_combo = ComboBox(
        root,
        variable=modo_var,
        values=list(MODOS_LABELS.values()),
        command=lambda *_: (_push_state(), actualizar_armonizacion()),
    )
    modo_var.trace_add("write", lambda *a: (_push_state(), actualizar_armonizacion()))

    armon_var = StringVar(value=ARMONIZACION_LABELS[ARMONIZACIONES[0]])
    armon_combo = ComboBox(
        root,
        variable=armon_var,
        values=list(ARMONIZACION_LABELS.values()),
        command=lambda *_: (_push_state(), actualizar_armonizacion()),
    )
    armon_var.trace_add("write", lambda *a: (_push_state(), actualizar_armonizacion()))

    def get_modo() -> str:
        return MODOS_INV.get(modo_combo.get(), modo_combo.get())

    def get_armon() -> str:
        return ARMONIZACION_INV.get(armon_combo.get(), armon_combo.get())

    globals()["get_modo"] = get_modo
    globals()["get_armon"] = get_armon
    import autocomplete
    autocomplete.get_modo = get_modo

    actualizar_armonizacion()

    scroll_fr = Frame(root)
    scroll_fr.pack(fill="both", expand=True, padx=5, pady=5)
    scroll_fr.rowconfigure(0, weight=1)
    scroll_fr.columnconfigure(0, weight=1)

    canvas = Canvas(scroll_fr, height=300, width=800, bg=COLORS['pianoroll_bg'])
    canvas.grid(row=0, column=0, sticky="nsew")
    y_scroll = Scrollbar(scroll_fr, orientation="vertical", command=canvas.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll = Scrollbar(scroll_fr, orientation="horizontal", command=canvas.xview)
    x_scroll.grid(row=1, column=0, columnspan=2, sticky="ew")
    canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
    canvas.bind(
        "<MouseWheel>",
        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        add=True,
    )
    canvas.bind(
        "<Shift-MouseWheel>",
        lambda e: canvas.xview_scroll(int(-1 * (e.delta / 120)), "units"),
        add=True,
    )

    btn_frame = Frame(root)
    btn_frame.pack(pady=5)

    Button(
        btn_frame,
        image=root.icon_arrow_up,
        text="",
        width=40,
        fg_color=COLORS['button'],
        hover_color=COLORS['accent'],
        command=lambda: _shift_all_inversions(1),
    ).pack(side="left", padx=5)
    Button(
        btn_frame,
        image=root.icon_arrow_down,
        text="",
        width=40,
        fg_color=COLORS['button'],
        hover_color=COLORS['accent'],
        command=lambda: _shift_all_inversions(-1),
    ).pack(side="left", padx=5)

    # Button to force a refresh in case automatic updates fail for any reason.
    Button(
        btn_frame,
        text="Generar Variación",
        image=root.icon_refresh,
        compound="left",
        font=general_font,
        fg_color=COLORS['button'],
        hover_color=COLORS['accent'],
        command=lambda: actualizar_visualizacion(force_new_seed=True),
    ).pack(side="left", padx=5)
    Button(
        btn_frame,
        text="Escuchar Preview",
        image=root.icon_play,
        compound="left",
        font=general_font,
        fg_color=COLORS['button'],
        hover_color=COLORS['accent'],
        command=reproducir_preview,
    ).pack(side="left", padx=5)
    Button(
        btn_frame,
        text="Detener",
        compound="left",
        font=general_font,
        fg_color=COLORS['button'],
        hover_color=COLORS['accent'],
        command=detener_preview,
    ).pack(side="left", padx=5)
    Button(
        btn_frame,
        text="<<",
        width=40,
        fg_color=COLORS['button'],
        hover_color=COLORS['accent'],
        command=lambda: mover_playhead(-8),
    ).pack(side="left", padx=5)
    Button(
        btn_frame,
        text=">>",
        width=40,
        fg_color=COLORS['button'],
        hover_color=COLORS['accent'],
        command=lambda: mover_playhead(8),
    ).pack(side="left", padx=5)
    Button(
        btn_frame,
        text="Exportar MIDI",
        image=root.icon_export,
        compound="left",
        font=general_font,
        fg_color=COLORS['button'],
        hover_color=COLORS['accent'],
        command=lambda: generar(
            status_var,
            clave_var,
            variacion_var,
            inversion_var,
            texto.text,
            modo_combo,
            armon_combo,
            inversiones_custom=current_inversions,
            armonias_custom=chord_armos,
            override_text=_normalise_bars(texto.text.get("1.0", "end")),
            manual_edits=manual_edits,
            seed=current_seed,
            bpm=None,
        ),
    ).pack(side="left", padx=5)
    Button(
        btn_frame,
        text="Personalizar",
        font=general_font,
        fg_color=COLORS['accent'],
        hover_color=COLORS['button'],
        command=open_style_editor,
    ).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # MIDI connection menu
    # ------------------------------------------------------------------
    def _select_port(choice: str) -> None:
        global MIDI_PORT
        if MIDI_PORT is not None:
            try:
                MIDI_PORT.close()
            except Exception:
                pass
            MIDI_PORT = None
        if choice == "Desconectado" or choice == "Todas las salidas MIDI":
            status_var.set("MIDI desconectado")
            # Aquí puedes poner alguna lógica extra si quieres deshabilitar playback/exportación, etc.
        else:
            status_var.set(f"Puerto MIDI: {choice}")
            # Aquí tu lógica normal de conexión (si aplica)

    frame_midi = Frame(root)
    frame_midi.pack(pady=(5, 0))
    Label(frame_midi, text="Puerto MIDI:", font=general_font, text_color=COLORS['fg']).pack(side="left", padx=(0, 5))
    OptionMenu(
        frame_midi,
        variable=midi_port_var,
        values=["Desconectado", "Todas las salidas MIDI"] + list(port_map.keys()),
        command=_select_port,
        width=160,
    ).pack(side="left")
    Label(root, textvariable=status_var, font=general_font, text_color=COLORS['fg']).pack(pady=(5, 0))

    root.bind_all("<Control-z>", undo)
    root.bind_all("<Control-Z>", undo)
    root.bind_all("<Command-z>", undo)
    root.bind_all("<Command-Z>", undo)
    root.bind_all("<Control-y>", redo)
    root.bind_all("<Control-Y>", redo)
    root.bind_all("<Command-Shift-Z>", redo)
    root.bind_all("<Command-Shift-z>", redo)

    apply_global_styles()

    try:
        root.mainloop()
    finally:
        save_preferences(
            {
                'general': {
                    'family': general_font.cget('family'),
                    'size': general_font.cget('size'),
                },
                'entry': {
                    'family': entry_font.cget('family'),
                    'size': entry_font.cget('size'),
                },
                'measure': {
                    'family': measure_font.cget('family'),
                    'size': measure_font.cget('size'),
                },
            }
        )
        if MIDI_PORT is not None:
            try:
                MIDI_PORT.close()
            finally:
                pygame.midi.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generador de Montunos")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug or os.getenv("MONTUNOS_DEBUG") else logging.INFO
    logging.basicConfig(level=log_level)

    main()
