# -*- coding: utf-8 -*-
"""CustomTkinter Text widget with chord autocompletion."""

import customtkinter as ctk
from tkinter import Listbox, END, ACTIVE
from typing import List, Optional
import re

from voicings import INTERVALOS_TRADICIONALES, NOTAS, parsear_nombre_acorde
from armonia_extendida import DICCIONARIO_EXTENDIDA

# Placeholder replaced by :func:`main.get_modo` at runtime
def get_modo() -> str:
    return "Tradicional"


class ChordAutocomplete(ctk.CTkTextbox):
    """A ``CTkTextbox`` that shows chord suggestions as the user types."""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._popup: Optional[ctk.CTkToplevel] = None
        self._listbox: Optional[Listbox] = None
        self._suggestions: List[str] = []
        self._popup_type: Optional[str] = None

        self._roots = sorted(NOTAS.keys(), key=lambda x: (len(x), x))

        # Precompile regular expressions for syntax highlighting
        suf_regex = "|".join(
            sorted((re.escape(s) for s in INTERVALOS_TRADICIONALES.keys()), key=len, reverse=True)
        )
        self._chord_regex = re.compile(rf"(?P<root>[A-G](?:b|#)?)(?P<suffix>{suf_regex})?")

        # Configure syntax highlighting tags.  The root and suffix use the
        # same yellow tone as the small chord labels shown on the piano roll.
        self.tag_config("root", foreground="#ffcc33")
        self.tag_config("suffix", foreground="#ffcc33")
        # Errors are shown in red.  This tag is applied only when a token
        # cannot be parsed as a valid chord or special marker.
        self.tag_config("error", foreground="red")

        self.bind("<KeyRelease>", self._on_keyrelease)
        self.bind("<Tab>", self._on_select)
        self.bind("<Return>", self._on_select)
        self.bind("<Down>", self._on_down)
        self.bind("<Up>", self._on_up)
        self.bind("<Button-1>", lambda e: self._hide_popup())

    # ------------------------------------------------------------------
    # Suggestion logic
    # ------------------------------------------------------------------
    @property
    def _suffixes(self) -> List[str]:
        """Return the list of chord suffixes supported by the app."""
        if get_modo() == "Armonía extendida":
            return list(DICCIONARIO_EXTENDIDA.keys())
        return list(INTERVALOS_TRADICIONALES.keys())

    def _current_word(self) -> str:
        """Return the current chord fragment before the cursor."""
        prefix = self.get("1.0", "insert")
        last_space = prefix.rfind(" ")
        last_bar = prefix.rfind("|")
        last_nl = prefix.rfind("\n")
        start = max(last_space, last_bar, last_nl) + 1
        return prefix[start:]

    def _previous_word(self) -> str:
        """Return the previous chord token before the cursor."""
        prefix = self.get("1.0", "insert")
        prefix = prefix.rstrip()
        last_space = prefix.rfind(" ")
        last_bar = prefix.rfind("|")
        last_nl = prefix.rfind("\n")
        start = max(last_space, last_bar, last_nl) + 1
        return prefix[start:]

    def _es_cifrado_valido(self, frag: str) -> bool:
        """Return ``True`` if ``frag`` is a valid root or chord name."""
        if frag in NOTAS:
            return True
        try:
            parsear_nombre_acorde(frag)
            return True
        except Exception:
            return False

    def _get_suggestions(self, fragment: str) -> List[str]:
        fragment = fragment.strip()
        if not fragment:
            return []

        # longest root prefix
        root_match = None
        for r in self._roots:
            if fragment.lower().startswith(r.lower()):
                if root_match is None or len(r) > len(root_match):
                    root_match = r
        if root_match:
            suf_part = fragment[len(root_match):]
            return [root_match + suf for suf in self._suffixes if suf.startswith(suf_part)]

        # not matching any root yet -> propose roots
        return [r for r in self._roots if r.lower().startswith(fragment.lower())]

    # ------------------------------------------------------------------
    # Popup management
    # ------------------------------------------------------------------
    def _show_popup(self, suggestions: List[str], popup_type: str = "chord") -> None:
        if not suggestions:
            self._hide_popup()
            return

        if self._popup is None:
            self._popup = ctk.CTkToplevel(self)
            self._popup.wm_overrideredirect(True)
            self._listbox = Listbox(self._popup, height=6)
            self._listbox.pack()
            self._listbox.bind("<ButtonRelease-1>", self._on_select)
        else:
            self._listbox.delete(0, END)

        self._listbox.delete(0, END)
        for s in suggestions:
            self._listbox.insert(END, s)
        self._listbox.select_set(0)
        self._suggestions = suggestions
        self._popup_type = popup_type

        bbox = self.bbox("insert")
        if bbox:
            x = bbox[0]
            y = bbox[1] + bbox[3]
            self._popup.geometry(f"+{self.winfo_rootx()+x}+{self.winfo_rooty()+y}")
        self._popup.deiconify()

    def _hide_popup(self) -> None:
        if self._popup is not None:
            self._popup.withdraw()
        self._popup_type = None

    def _highlight(self) -> None:
        """Apply basic syntax highlighting for the progression text."""
        text = self.get("1.0", "end-1c")
        for tag in ("root", "suffix", "error"):
            self.tag_remove(tag, "1.0", "end")

        for m in self._chord_regex.finditer(text):
            start_r = f"1.0+{m.start('root')}c"
            end_r = f"1.0+{m.end('root')}c"
            self.tag_add("root", start_r, end_r)
            if m.group('suffix'):
                start_s = f"1.0+{m.start('suffix')}c"
                end_s = f"1.0+{m.end('suffix')}c"
                self.tag_add("suffix", start_s, end_s)

        # Mark tokens that are not recognised as valid chords or special
        # markers.  This mirrors the behaviour of the classic UI where
        # syntax errors are highlighted in red.
        import re

        def _token_ok(tok: str) -> bool:
            if tok == "%":
                return True
            if re.fullmatch(r"[|:]+", tok):
                return True
            try:
                parsear_nombre_acorde(tok)
                return True
            except Exception:
                return False

        for m in re.finditer(r"\S+", text):
            token = m.group(0)
            if not _token_ok(token):
                self.tag_add("error", f"1.0+{m.start()}c", f"1.0+{m.end()}c")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_keyrelease(self, event) -> None:
        if event.keysym in {"Up", "Down", "Return", "Tab"}:
            return

        if self._popup_type:
            if event.char in {" ", "\n", "|"}:
                token = self._previous_word()
                if token and self._es_cifrado_valido(token):
                    self._hide_popup()
                    self._highlight()
                    return
            fragment = self._current_word()
            suggestions = self._get_suggestions(fragment)
            self._show_popup(suggestions, "chord")
            self._highlight()
            return

        fragment = self._current_word()
        suggestions = self._get_suggestions(fragment)
        self._show_popup(suggestions, "chord")
        self._highlight()

    def _on_select(self, event=None):
        if not self._popup or not self._suggestions:
            return "break"
        idx = self._listbox.curselection()
        if not idx:
            idx = (0,)
        choice = self._suggestions[int(idx[0])]
        if self._popup_type == "chord":
            frag = self._current_word()
            self.delete(f"insert-{len(frag)}c", "insert")
            self.insert("insert", choice)
        self._hide_popup()
        self.focus_set()
        self._highlight()
        return "break"

    def _on_down(self, event=None):
        if self._popup and self._suggestions:
            cur = self._listbox.curselection()
            next_idx = 0 if not cur else (int(cur[0]) + 1) % len(self._suggestions)
            self._listbox.select_clear(0, END)
            self._listbox.select_set(next_idx)
            return "break"
        return None

    def _on_up(self, event=None):
        if self._popup and self._suggestions:
            cur = self._listbox.curselection()
            next_idx = len(self._suggestions) - 1 if not cur else (int(cur[0]) - 1) % len(self._suggestions)
            self._listbox.select_clear(0, END)
            self._listbox.select_set(next_idx)
            return "break"
        return None

