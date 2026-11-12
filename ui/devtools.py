"""
ui/devtools.py
-----------------
Reference codes: tiny numbered chips -- [01], [02], [03] -- attached to the UI elements worth
pointing at. Hover a chip for the element's full structured info (page/section/button/field, and for
fields the CURRENT live value, recomputed every time); click it to copy that same text (with the
reference number on top) to the clipboard. Reporting a change is then just "reference 27 needs to
change" -- the number says exactly which page/section/element you mean.

Two kinds of chip:

* Reference codes (the default, `overlay=True`): drawn as faint overlays PLACED on top of the UI --
  they take no layout space, so switching them on or off never moves anything. Visible while the
  REF CODES switch at the top right of the window is on (MainWindow.set_reference_codes()); that
  switch turns itself off after REFERENCE_CODES_SECONDS.
* Diagnostic readouts (`overlay=False`): the live-value chips in ANALYSIS' debug strip. They are
  part of Developer Mode (SETTINGS), which also reveals the larger diagnostic panels, and they do
  occupy layout space like the panels they sit in.

Every chip is created once, at page-build time, and registered here so a single toggle can show/hide
them all immediately -- pages are never rebuilt to turn anything on or off.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional, Union

import customtkinter as ctk

REFERENCE_CODES_SECONDS = 2 * 60 * 60   # the REF CODES switch turns itself off after this long

# Shaded, not deep: clearly legible up close, quiet at a glance (the UI's secondary text is #9C9CA3).
REF_COLOUR = "#6D6D78"          # on the dark panels; chips on other fills are shaded to suit (see shade_for)
REF_COLOUR_HOVER = "#B4B4BC"
_REF_FONT = ("Consolas", 8)
_TOOLTIP_FONT = ("Consolas", 9)
_COPIED_REVERT_MS = 900
_STACK_STEP_PX = 13   # vertical spacing when several corner chips share one parent

_registry = []  # every DevRef ever built, so refresh_all() can walk them
_ref_counter = 0  # global, sequential -- matches page-build order, so refs read low-to-high top to bottom


def is_enabled(app) -> bool:
    """Developer Mode (SETTINGS): the diagnostic panels and live-value readouts."""
    return bool(getattr(app.config, "developer_mode", False))


def codes_visible(app) -> bool:
    """The REF CODES switch: the faint overlay reference codes."""
    return bool(getattr(app, "ref_codes_on", False))


def refresh_all(app) -> None:
    """Call right after either switch changes."""
    for ref in list(_registry):
        try:
            alive = ref.winfo_exists()
        except tk.TclError:
            alive = False  # its whole window/interpreter is gone
        if alive:
            ref.set_visible(codes_visible(app) if ref.overlay else is_enabled(app))
        else:
            _registry.remove(ref)


def _hex_rgb(colour) -> Optional[tuple]:
    if isinstance(colour, str) and len(colour) == 7 and colour.startswith("#"):
        try:
            return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            return None
    return None


def _blend(rgb: tuple, target: int, amount: float) -> str:
    return "#" + "".join(f"{round(c + (target - c) * amount):02X}" for c in rgb)


def shade_for(backdrop, strength: float = 1.0) -> str:
    """A muted text colour for a chip sitting on `backdrop`: the backdrop pulled part-way towards white
    (dark fills) or black (bright fills), so it reads as a shade of what's behind it instead of a
    colour of its own. strength > 1 is the brighter hover/confirm tint."""
    rgb = _hex_rgb(backdrop)
    if rgb is None:
        return REF_COLOUR if strength <= 1.0 else REF_COLOUR_HOVER
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    if luma > 110:
        return _blend(rgb, 0, min(0.95, 0.50 * strength))
    return _blend(rgb, 255, min(0.95, 0.38 * strength))


def _next_ref_number() -> int:
    global _ref_counter
    _ref_counter += 1
    return _ref_counter


# ------------------------------------------------------------------
# Structured info builders -- DevRef prepends "REF: NN" ahead of whatever these produce.
# ------------------------------------------------------------------
def page_text(identifier: str) -> str:
    return f"PAGE: {identifier}"


def section_text(identifier: str, page: str) -> str:
    return f"SECTION: {identifier}\nPAGE: {page}"


def tab_text(identifier: str) -> str:
    return f"TAB: {identifier}\nPAGE: {identifier}"


def button_text(identifier: str, page: str, section: str) -> str:
    return f"BUTTON: {identifier}\nPAGE: {page}\nSECTION: {section}"


def field_text(identifier: str, value) -> str:
    return f"FIELD: {identifier}\nVALUE: {value}"


def error_text(identifier: str, page: str, **extra) -> str:
    lines = [f"ERROR: {identifier}", f"PAGE: {page}"]
    for key, value in extra.items():
        lines.append(f"{key.upper()}: {value}")
    return "\n".join(lines)


# ------------------------------------------------------------------
class DevRef(ctk.CTkLabel):
    """A tiny, shaded, clickable/hoverable reference chip -- [07], or [E12] for an error ref.
    `info_text` may be a plain string or a zero-arg callable, so a chip attached to a live value
    always shows/copies the up-to-date text, computed fresh on every hover and every click.

    Overlay chips (the default) are positioned with Tk's `place`, relative to `anchor`, and never
    take layout space. `placement` is the place() options for where they sit; diagnostic chips
    (`overlay=False`) are packed like any other widget."""

    def __init__(
        self, parent, colors: dict, kind: str, info_text: Union[str, Callable[[], str]],
        overlay: bool = True, placement: Optional[dict] = None, backdrop=None,
    ):
        self.ref_number = _next_ref_number()
        self.overlay = overlay
        self.kind = kind
        self.anchor_widget = None      # what the code is attached to (None = a corner of its parent)
        self.backdrop_anchor = None    # the widget whose fill it is drawn over, if any
        self._is_error = kind == "error"
        self._normal_text = f"[E{self.ref_number}]" if self._is_error else f"[{self.ref_number:02d}]"
        self._rest_colour = (
            colors.get("danger") if self._is_error
            else (shade_for(backdrop) if overlay else colors["text_secondary"])
        )
        self._hover_colour = shade_for(backdrop, 1.7) if overlay else self._rest_colour
        super().__init__(
            parent, text=self._normal_text, text_color=self._rest_colour, font=_REF_FONT, cursor="hand2", anchor="w",
            **({"height": 11, "corner_radius": 0} if overlay else {}), **({"fg_color": backdrop, "bg_color": backdrop} if backdrop else {}),
        )
        self._info_text = info_text
        self._placement = placement or {}
        self._pack_opts: dict = {}
        self._visible = False
        self._ever_packed = False
        self._tooltip: Optional[tk.Toplevel] = None
        self._revert_job = None
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        _registry.append(self)

    def set_visible(self, visible: bool, pack_opts: Optional[dict] = None) -> None:
        if self.overlay:
            self._set_overlay_visible(visible)
            return
        if pack_opts is not None:
            self._pack_opts = pack_opts
        if not self._ever_packed:
            # First call ever, regardless of whether the switch is on yet -- establishes this chip's
            # position among its real siblings right now, before any later-created widgets get
            # packed after it (the pack-order bug this project has hit before).
            self.pack(**self._pack_opts)
            self._ever_packed = True
            self._visible = True
            if not visible:
                self.pack_forget()
                self._visible = False
            return
        if visible and not self._visible:
            self.pack(**self._pack_opts)
            self._visible = True
        elif not visible and self._visible:
            self.pack_forget()
            self._visible = False

    def _set_overlay_visible(self, visible: bool) -> None:
        if visible and not self._visible:
            self.place(**self._placement)
            self.tk.call("raise", self._w)  # above whatever it sits on
            self._visible = True
        elif visible and self._visible:
            self.tk.call("raise", self._w)
        elif not visible and self._visible:
            self._hide_tooltip()
            self.place_forget()
            self._visible = False

    def destroy(self) -> None:
        """Tidy exit: a rebuilt panel destroys its chips, so drop this one from the registry (it would
        otherwise grow with every rebuild), close any tooltip and cancel the pending '✓' revert."""
        if self in _registry:
            _registry.remove(self)
        if self._revert_job is not None:
            try:
                self.after_cancel(self._revert_job)
            except tk.TclError:
                pass
            self._revert_job = None
        self._hide_tooltip()
        super().destroy()

    def retint(self) -> None:
        """Re-match the chip's background to what it sits on (a nav tab's fill changes when selected)."""
        colour = backdrop_colour(self.backdrop_anchor, self.master)
        if colour:
            if not self._is_error:
                self._rest_colour = shade_for(colour)
                self._hover_colour = shade_for(colour, 1.7)
            try:
                self.configure(fg_color=colour, bg_color=colour, text_color=self._rest_colour)
            except tk.TclError:
                pass

    def _resolve_info_text(self) -> str:
        body = self._info_text() if callable(self._info_text) else self._info_text
        ref_line = f"REF: E{self.ref_number}" if self._is_error else f"REF: {self.ref_number:02d}"
        return f"{ref_line}\n{body}"

    def _on_click(self, event=None) -> None:
        text = self._resolve_info_text()
        try:
            top = self.winfo_toplevel()
            top.clipboard_clear()
            top.clipboard_append(text)
        except tk.TclError:
            pass
        if self._revert_job is not None:
            self.after_cancel(self._revert_job)
        self.configure(text="✓", text_color=self._hover_colour)
        self._revert_job = self.after(_COPIED_REVERT_MS, self._revert_text)
        self._hide_tooltip()

    def _revert_text(self) -> None:
        self._revert_job = None
        if self.winfo_exists():
            self.configure(text=self._normal_text, text_color=self._rest_colour)

    def _on_enter(self, event=None) -> None:
        if self.overlay and self._revert_job is None:
            self.configure(text_color=self._hover_colour)
        if self._tooltip is not None:
            return
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 2
        tip = tk.Toplevel(self)
        tip.overrideredirect(True)
        tip.attributes("-topmost", True)
        tk.Label(
            tip, text=self._resolve_info_text(), bg="#101012", fg="#F2F2F0", font=_TOOLTIP_FONT,
            padx=6, pady=4, borderwidth=1, relief="solid", justify="left", anchor="w",
        ).pack()
        tip.geometry(f"+{x}+{y}")
        self._tooltip = tip

    def _on_leave(self, event=None) -> None:
        if self.overlay and self._revert_job is None:
            self.configure(text_color=self._rest_colour)
        self._hide_tooltip()

    def _hide_tooltip(self) -> None:
        if self._tooltip is not None:
            try:
                self._tooltip.destroy()
            except tk.TclError:
                pass  # already gone along with its window
            self._tooltip = None


# ------------------------------------------------------------------
# Convenience constructors -- used at each page's _build() time. Call signatures are unchanged
# (identifier/page/section/value_getter/pack_opts), so existing call sites keep working; for overlay
# chips `pack_opts` is ignored and `anchor` says which widget the code belongs to.
# ------------------------------------------------------------------
def _corner_placement(parent) -> dict:
    """Top-right inside `parent`; several chips on one parent stack downwards rather than overlapping."""
    # Counted from the chips that currently exist (a parent whose children get rebuilt, like a progress
    # panel or the video drop zone, must not drift downward on every rebuild).
    slot = sum(
        1 for child in parent.winfo_children()
        if isinstance(child, DevRef) and child.overlay and child.anchor_widget is None
    )
    return dict(in_=parent, relx=1.0, x=-6, y=2 + _STACK_STEP_PX * slot, anchor="ne")


def _anchored_placement(anchor, side: str) -> dict:
    """Where a chip sits relative to the widget it labels. 'right'/'left' = beside it (vertically
    centred), 'inside' = its top-right corner (the safe choice for a control with no room beside it),
    'inside_bottom' = its bottom-right corner, 'bottom' = along its bottom edge (nav tabs)."""
    if side == "left":
        return dict(in_=anchor, relx=0.0, rely=0.5, x=-4, anchor="e")
    if side == "inside":
        return dict(in_=anchor, relx=1.0, rely=0.0, x=-2, y=0, anchor="ne")
    if side == "inside_bottom":
        return dict(in_=anchor, relx=1.0, rely=1.0, x=-2, y=-1, anchor="se")
    if side == "bottom":
        return dict(in_=anchor, relx=0.5, rely=1.0, y=-2, anchor="s")
    return dict(in_=anchor, relx=1.0, rely=0.5, x=5, anchor="w")  # "right"


def _make(parent, app, colors, kind, info_text, pack_opts, overlay=True, anchor=None, side=None) -> DevRef:
    placement = None
    sits_on_anchor = None   # the widget whose fill the chip lies over (only when it is drawn on top of it)
    if overlay:
        if anchor is None:
            placement = _corner_placement(parent)
        else:
            chosen = side or {"button": "inside", "tab": "bottom"}.get(kind, "right")
            placement = _anchored_placement(anchor, chosen)
            if chosen in ("inside", "inside_bottom", "bottom"):
                sits_on_anchor = anchor
    ref = DevRef(
        parent, colors, kind, info_text, overlay=overlay, placement=placement,
        backdrop=backdrop_colour(sits_on_anchor, parent) if overlay else None,
    )
    ref.anchor_widget = anchor
    ref.backdrop_anchor = sits_on_anchor
    ref.set_visible(codes_visible(app) if overlay else is_enabled(app), pack_opts=pack_opts)
    return ref


def backdrop_colour(anchor, parent):
    """The solid colour behind a chip: Tk has no transparent child widgets, so the chip's own
    background has to match whatever it sits on -- the anchor's fill if it has one, otherwise the
    nearest filled ancestor."""
    for widget in (anchor, parent):
        while widget is not None:
            try:
                colour = widget.cget("fg_color")
            except Exception:
                colour = None
            if colour and colour != "transparent":
                return colour
            widget = getattr(widget, "master", None)
    return None


def page_label(parent, app, colors, identifier: str, pack_opts: Optional[dict] = None, anchor=None) -> DevRef:
    return _make(parent, app, colors, "page", lambda: page_text(identifier), pack_opts or {"anchor": "w"}, anchor=anchor)


def section_label(
    parent, app, colors, identifier: str, page: str, pack_opts: Optional[dict] = None, anchor=None,
    side: Optional[str] = None,
) -> DevRef:
    """With `anchor` (a heading, or a whole panel with side="inside") the code sits on/after it; without
    one it takes the next free slot in the top-right corner of `parent`."""
    return _make(
        parent, app, colors, "section", lambda: section_text(identifier, page), pack_opts or {"anchor": "w"},
        anchor=anchor, side=side,
    )


def field_label(
    parent, app, colors, identifier: str, value_getter, pack_opts: Optional[dict] = None,
    anchor=None, overlay: bool = True, side: Optional[str] = None,
) -> DevRef:
    """`side` ('right' default, 'left', 'inside') says where the chip sits beside `anchor`; with no
    anchor it takes the next free slot in the top-right corner of `parent`. overlay=False is the
    Developer Mode readout strip: a packed chip that does take layout space."""
    def info() -> str:
        value = value_getter() if callable(value_getter) else value_getter
        return field_text(identifier, value)

    return _make(
        parent, app, colors, "field", info, pack_opts or {"anchor": "w"}, overlay=overlay, anchor=anchor, side=side,
    )


def readout_label(parent, app, colors, identifier: str, value_getter, pack_opts: Optional[dict] = None) -> DevRef:
    """A Developer Mode (SETTINGS) live-value chip in a dedicated readout strip: packed, so it takes
    layout space -- but only while Developer Mode is on, in a strip made for it."""
    return field_label(parent, app, colors, identifier, value_getter, pack_opts=pack_opts, overlay=False)


def error_label(parent, app, colors, identifier: str, page: str, pack_opts: Optional[dict] = None, anchor=None, **extra) -> DevRef:
    return _make(
        parent, app, colors, "error", lambda: error_text(identifier, page, **extra), pack_opts or {"anchor": "w"},
        anchor=anchor,
    )


def dev_button(
    parent, app, colors, identifier: str, page: str, section: str,
    pack_opts: Optional[dict] = None, button_pack_opts: Optional[dict] = None, **button_kwargs
) -> ctk.CTkButton:
    """Builds the real CTkButton (returned, so callers keep using it exactly as before -- .configure(),
    etc.) inside a small container; `pack_opts` places that container (so side="left" rows / fill="x"
    stacks behave exactly as the bare button used to); pass `button_pack_opts` too if the button itself
    also needs e.g. fill="x" to stretch inside that container. Its reference code is an overlay on the
    button's top-right corner, so the container is exactly the button's size with codes on or off."""
    container = ctk.CTkFrame(parent, fg_color="transparent")
    container.pack(**(pack_opts if pack_opts is not None else {"side": "left"}))
    button = ctk.CTkButton(container, **button_kwargs)
    button.pack(side="left", **(button_pack_opts or {}))
    _make(container, app, colors, "button", lambda: button_text(identifier, page, section), None, anchor=button)
    return button


def tab_label(parent, app, colors, identifier: str, anchor, pack_opts: Optional[dict] = None) -> DevRef:
    return _make(parent, app, colors, "tab", lambda: tab_text(identifier), pack_opts, anchor=anchor)
