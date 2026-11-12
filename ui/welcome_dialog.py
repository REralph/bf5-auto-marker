"""
ui/welcome_dialog.py
-----------------------
First-use setup guide (spec item 17). Shown once on first launch unless
dismissed with "Don't show this again", which persists to config.json.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk


def show_welcome_dialog(parent, colors: dict, on_dont_show_again: Callable[[], None]) -> None:
    dialog = ctk.CTkToplevel(parent)
    dialog.title("Welcome to BF5 Auto Marker")
    dialog.configure(fg_color=colors["bg"])
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    ctk.CTkLabel(
        dialog, text="WELCOME TO BF5 AUTO MARKER", font=("Segoe UI", 18, "bold"),
        text_color=colors["text_primary"],
    ).pack(padx=28, pady=(28, 6), anchor="w")

    ctk.CTkLabel(
        dialog, text="The application works in four steps:", font=("Segoe UI", 13),
        text_color=colors["text_secondary"],
    ).pack(padx=28, pady=(0, 12), anchor="w")

    steps = [
        "Select your gameplay video.",
        "Calibrate where the skull appears.",
        "Analyse the video.",
        "Send the detected markers to DaVinci Resolve.",
    ]
    for i, step in enumerate(steps, start=1):
        row = ctk.CTkFrame(dialog, fg_color="transparent")
        row.pack(fill="x", padx=28, pady=4, anchor="w")
        ctk.CTkLabel(
            row, text=str(i), width=24, height=24, corner_radius=12, fg_color=colors["accent"],
            text_color="#FFFFFF", font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        ctk.CTkLabel(
            row, text=step, font=("Segoe UI", 13), text_color=colors["text_primary"]
        ).pack(side="left", padx=10)

    dont_show_var = tk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        dialog, text="Don't show this again", variable=dont_show_var, text_color=colors["text_secondary"],
    ).pack(padx=28, pady=(18, 6), anchor="w")

    def start_setup() -> None:
        if dont_show_var.get():
            on_dont_show_again()
        dialog.destroy()

    ctk.CTkButton(
        dialog, text="START SETUP", fg_color=colors["accent"], hover_color=colors["accent_hover"],
        height=38, width=180, command=start_setup,
    ).pack(pady=(6, 28))
