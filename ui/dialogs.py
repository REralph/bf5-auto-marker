"""
ui/dialogs.py
---------------
Reusable dark-themed modal dialogs, replacing the native (light, OS-themed)
tkinter.messagebox for user-facing messages -- keeps the "consistent dark
theme across ... error messages" requirement holding as the app grows.
Errors use a structured What happened / Why it may have happened / What to
do layout (spec item 19) instead of a bare message.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import customtkinter as ctk


def _base_dialog(parent, colors: dict, title: str, width: int = 440) -> ctk.CTkToplevel:
    dialog = ctk.CTkToplevel(parent)
    dialog.title(title)
    dialog.configure(fg_color=colors["bg"])
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.minsize(width, 1)  # width only -- height auto-fits whatever gets packed into it
    return dialog


def show_info(parent, colors: dict, title: str, message: str) -> None:
    dialog = _base_dialog(parent, colors, title)
    ctk.CTkLabel(
        dialog, text=title, font=("Segoe UI", 15, "bold"), text_color=colors["text_primary"]
    ).pack(padx=20, pady=(20, 6), anchor="w")
    ctk.CTkLabel(
        dialog, text=message, font=("Segoe UI", 12), text_color=colors["text_secondary"],
        wraplength=380, justify="left",
    ).pack(padx=20, pady=(0, 16), anchor="w")
    ctk.CTkButton(
        dialog, text="OK", fg_color=colors["accent"], hover_color=colors["accent_hover"],
        command=dialog.destroy, width=100,
    ).pack(pady=(0, 20))
    dialog.wait_window()


def ask_yes_no(parent, colors: dict, title: str, message: str) -> bool:
    dialog = _base_dialog(parent, colors, title)
    result = {"value": False}

    ctk.CTkLabel(
        dialog, text=title, font=("Segoe UI", 15, "bold"), text_color=colors["text_primary"]
    ).pack(padx=20, pady=(20, 6), anchor="w")
    ctk.CTkLabel(
        dialog, text=message, font=("Segoe UI", 12), text_color=colors["text_secondary"],
        wraplength=380, justify="left",
    ).pack(padx=20, pady=(0, 16), anchor="w")

    def confirm() -> None:
        result["value"] = True
        dialog.destroy()

    row = ctk.CTkFrame(dialog, fg_color="transparent")
    row.pack(pady=(0, 20))
    ctk.CTkButton(
        row, text="Cancel", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
        text_color=colors["text_primary"], command=dialog.destroy, width=100,
    ).pack(side="left", padx=6)
    ctk.CTkButton(
        row, text="Confirm", fg_color=colors["accent"], hover_color=colors["accent_hover"],
        command=confirm, width=100,
    ).pack(side="left", padx=6)

    dialog.wait_window()
    return result["value"]


def show_error(
    parent,
    colors: dict,
    title: str,
    what_happened: str,
    why: Optional[Sequence[str]] = None,
    what_to_do: Optional[str] = None,
    action: Optional[Tuple[str, Callable[[], None]]] = None,
) -> None:
    """
    Structured error dialog (spec item 19): what happened, why it may have
    happened (as bullet points), and what the user can do -- with an
    optional single action button (label, callback) for the fix itself.
    """
    dialog = _base_dialog(parent, colors, title, width=460)

    ctk.CTkLabel(
        dialog, text=title, font=("Segoe UI", 16, "bold"), text_color=colors["danger"]
    ).pack(padx=20, pady=(20, 10), anchor="w")

    ctk.CTkLabel(
        dialog, text=what_happened, font=("Segoe UI", 13), text_color=colors["text_primary"],
        wraplength=400, justify="left",
    ).pack(padx=20, pady=(0, 10), anchor="w")

    if why:
        ctk.CTkLabel(
            dialog, text="Possible causes:", font=("Segoe UI", 12, "bold"), text_color=colors["text_secondary"]
        ).pack(padx=20, anchor="w")
        bullet_text = "\n".join(f"•  {reason}" for reason in why)
        ctk.CTkLabel(
            dialog, text=bullet_text, font=("Segoe UI", 12), text_color=colors["text_secondary"],
            wraplength=400, justify="left",
        ).pack(padx=20, pady=(2, 10), anchor="w")

    if what_to_do:
        ctk.CTkLabel(
            dialog, text=what_to_do, font=("Segoe UI", 12), text_color=colors["text_secondary"],
            wraplength=400, justify="left",
        ).pack(padx=20, pady=(0, 14), anchor="w")

    row = ctk.CTkFrame(dialog, fg_color="transparent")
    row.pack(pady=(0, 20))
    if action:
        label, callback = action

        def run_action() -> None:
            dialog.destroy()
            callback()

        ctk.CTkButton(
            row, text=label, fg_color=colors["accent"], hover_color=colors["accent_hover"], command=run_action
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            row, text="Close", fg_color=colors["panel_alt"], hover_color=colors["accent_dim"],
            text_color=colors["text_primary"], command=dialog.destroy,
        ).pack(side="left", padx=6)
    else:
        ctk.CTkButton(
            row, text="OK", fg_color=colors["accent"], hover_color=colors["accent_hover"],
            command=dialog.destroy, width=100,
        ).pack()

    dialog.wait_window()


class InfoIcon(ctk.CTkLabel):
    """
    A small clickable ⓘ, dropped beside any setting whose meaning isn't
    obvious (spec item 16). Click shows a short explanation popup.
    """

    def __init__(self, parent, colors: dict, title: str, text: str, **kwargs):
        super().__init__(
            parent, text="ⓘ", font=("Segoe UI", 13), text_color=colors["text_secondary"], cursor="hand2", **kwargs
        )
        self.bind("<Button-1>", lambda e: show_info(self.winfo_toplevel(), colors, title, text))
