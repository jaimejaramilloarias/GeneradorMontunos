import customtkinter as ctk
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, Any

CONFIG_FILE = Path.home() / ".cajonea2_prefs.json"

COLORS = {
    'bg': '#222831',
    'fg': '#f5f5f5',
    'button': '#2196f3',
    'accent': '#ff9800',
    'pianoroll_bg': '#1e1e1e',
    'note': '#2196f3',
    'note_selected': '#ff9800',
    'measure': '#cccccc',
}

PREFERENCES: Dict[str, Any] = {}


def _load_font_from_file(root, name: str) -> Optional[str]:
    try:
        import tkinter.font as tkfont
        font_path = Path(__file__).with_name(name)
        f = tkfont.Font(root=root, file=font_path)
        return f.actual('family')
    except Exception:
        return None


def get_entry_font(root) -> ctk.CTkFont:
    """Font used for the chord entry widget."""
    return ctk.CTkFont(family="Impact", size=48)


def get_general_font(root) -> ctk.CTkFont:
    """Return the default font for most widgets."""
    return ctk.CTkFont(family="Monaco", size=14, weight="bold")


def get_measure_font(root) -> ctk.CTkFont:
    """Return the font used for measure numbers on the piano roll."""
    return ctk.CTkFont(family="Monaco", size=14, weight="bold")


def load_preferences() -> Tuple[Dict, Dict]:
    """Load saved appearance preferences from disk.

    Returns a tuple ``(fonts, prefs)`` with any stored font settings and
    miscellaneous preferences."""
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open() as fh:
                data = json.load(fh)
            COLORS.update(data.get('colors', {}))
            PREFERENCES.update(data.get('prefs', {}))
            return data.get('fonts', {}), PREFERENCES
        except Exception:
            return {}, {}
    return {}, {}


def save_preferences(fonts: Optional[Dict] = None, prefs: Optional[Dict] = None) -> None:
    """Persist current colors, fonts and extra preferences to disk."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {'colors': COLORS}
    if fonts:
        data['fonts'] = fonts
    if prefs is not None:
        data['prefs'] = prefs
    with CONFIG_FILE.open('w') as fh:
        json.dump(data, fh, indent=2)
