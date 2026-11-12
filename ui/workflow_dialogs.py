"""
ui/workflow_dialogs.py
-------------------------
The Resolve "why am I not connected" help popup. (The analysis-complete summary modal, the pre-send
marker-preview confirmation and the new-colour/new-symbol review prompts were all removed: REVIEW
MARKERS / SEND TO RESOLVE live on the WORKFLOW page's ANALYSIS stage, SEND TO RESOLVE sends
immediately, and the reconnect diagnostic lives on-page in ui/pages/resolve_page.py.)
"""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk


def show_resolve_help(parent, colors: dict, on_reconnect: Callable[[], None]) -> None:
    dialog = ctk.CTkToplevel(parent)
    dialog.title("Resolve Not Connected")
    dialog.configure(fg_color=colors["bg"])
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    ctk.CTkLabel(
        dialog, text='What does "Resolve Not Connected" mean?', font=("Segoe UI", 15, "bold"),
        text_color=colors["text_primary"], wraplength=380, justify="left",
    ).pack(padx=24, pady=(24, 4), anchor="w")
    ctk.CTkLabel(
        dialog, text="BF5 Auto Marker cannot currently communicate with DaVinci Resolve.",
        font=("Segoe UI", 12), text_color=colors["text_secondary"], wraplength=380, justify="left",
    ).pack(padx=24, pady=(0, 14), anchor="w")

    ctk.CTkLabel(
        dialog, text="How do I reconnect?", font=("Segoe UI", 14, "bold"), text_color=colors["text_primary"]
    ).pack(padx=24, anchor="w")
    steps = [
        "Open DaVinci Resolve.",
        "Open the project containing your gameplay.",
        "Open or select the timeline you want to edit.",
        "Make sure Resolve's scripting/API connection is available.",
        "Return to BF5 Auto Marker.",
        "Press RECONNECT.",
    ]
    for i, step in enumerate(steps, start=1):
        ctk.CTkLabel(
            dialog, text=f"{i}. {step}", font=("Segoe UI", 12), text_color=colors["text_secondary"],
            wraplength=380, justify="left",
        ).pack(padx=24, pady=2, anchor="w")

    def reconnect() -> None:
        dialog.destroy()
        on_reconnect()

    ctk.CTkButton(
        dialog, text="RECONNECT TO RESOLVE", fg_color=colors["accent"], hover_color=colors["accent_hover"],
        command=reconnect, height=36,
    ).pack(pady=(18, 24))
