"""The REF CODES switch (top right): faint, click-to-copy reference codes drawn over the UI.

They must (a) appear only while the switch is on, (b) never move or resize anything, (c) copy their
reference + context on click, (d) look shaded rather than strong, and (e) switch themselves off after two
hours -- all verified against the real MainWindow, with the clock faked where time has to pass.
"""

import dataclasses
import unittest

from tests.ui_base import UiTestCase


def brightness(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


class ReferenceCodesTests(UiTestCase):
    def setUp(self):
        super().setUp()
        import ui.devtools as devtools
        self.devtools = devtools
        self.now = [1_000_000.0]
        self.app._clock = lambda: self.now[0]
        self.addCleanup(self.app.set_reference_codes, False)   # leave no countdown timer behind

    # ---- helpers ----
    def chips(self, overlay=True):
        return [w for w in self.widgets(self.root) if isinstance(w, self.devtools.DevRef) and w.overlay == overlay]

    def mapped_chips(self, overlay=True):
        """Chips actually on screen. (winfo_viewable, not winfo_ismapped: a chip on a hidden page keeps Tk's
        'mapped' flag but is not viewable because its page is unmapped.)"""
        return [c for c in self.chips(overlay) if c.winfo_viewable()]

    def geometry_snapshot(self):
        """Position + size of every real (non-chip) widget that is on screen, plus the window itself."""
        shot = {"<root>": (self.root.winfo_width(), self.root.winfo_height())}

        def walk(widget):
            if isinstance(widget, self.devtools.DevRef):
                return  # the chips themselves (and their insides) are what's being switched
            try:
                if widget.winfo_viewable():
                    shot[str(widget)] = (
                        widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_width(), widget.winfo_height(),
                    )
                children = [] if widget is self.app.ref_codes_time_label else widget.winfo_children()
            except Exception:  # noqa: BLE001 - a widget destroyed mid-walk
                return
            for child in children:
                walk(child)

        walk(self.root)
        return shot

    def switch_on(self):
        self.app.set_reference_codes(True)
        self.pump(0.25)

    # ---- off by default ----
    def test_the_codes_start_switched_off_and_nothing_is_drawn(self):
        self.assertFalse(self.app.ref_codes_on)
        self.assertFalse(self.app.ref_codes_var.get())
        self.assertGreater(len(self.chips()), 30, "the pages register their codes up front")
        self.assertEqual(self.mapped_chips(), [])
        self.assertEqual(self.app.ref_codes_time_label.cget("text"), "")

    def test_the_switch_sits_in_the_top_right_corner_of_the_header(self):
        self.pump(0.2)
        switch = self.app.ref_codes_switch
        right_edge = switch.winfo_rootx() + switch.winfo_width()
        window_right = self.root.winfo_rootx() + self.root.winfo_width()
        self.assertGreater(switch.winfo_rootx(), self.root.winfo_rootx() + self.root.winfo_width() * 0.7)
        self.assertLess(window_right - right_edge, 60, "hard against the right edge")
        self.assertLess(switch.winfo_rooty() - self.root.winfo_rooty(), 70, "inside the top bar")

    # ---- switching on ----
    def test_switching_on_shows_the_codes_of_the_visible_page_and_the_tabs(self):
        self.switch_on()
        shown = self.mapped_chips()
        self.assertGreater(len(shown), 8)
        kinds = {c.kind for c in shown}
        self.assertLessEqual({"tab", "button"}, kinds)
        tab_chips = [c for c in shown if c.kind == "tab"]
        self.assertEqual(len(tab_chips), 3, "one on each nav tab")
        for chip in shown:
            self.assertGreater(chip.winfo_width(), 8)
            self.assertGreater(chip.winfo_height(), 8)
            self.assertRegex(chip.cget("text"), r"^\[\d{2,}\]$")

    def test_hidden_pages_show_their_codes_only_when_you_go_there(self):
        self.switch_on()
        settings_chips = [c for c in self.chips() if self.app.pages["SETTINGS"] in self.parents(c)]
        self.assertTrue(settings_chips)
        self.assertEqual([c for c in settings_chips if c.winfo_viewable()], [], "SETTINGS isn't the visible page")
        self.app.show_page("SETTINGS")
        self.pump(0.3)
        self.assertTrue([c for c in settings_chips if c.winfo_viewable()])

    def parents(self, widget):
        out = []
        while widget is not None:
            out.append(widget)
            widget = getattr(widget, "master", None)
        return out

    def test_reference_numbers_are_unique(self):
        numbers = [c.ref_number for c in self.chips(True) + self.chips(False)]
        self.assertEqual(len(numbers), len(set(numbers)))

    # ---- never interferes with the layout ----
    def test_switching_the_codes_on_and_off_moves_and_resizes_nothing_on_any_page(self):
        for page in ("WORKFLOW", "CALIBRATE", "SETTINGS"):
            self.app.show_page(page)
            self.pump(0.3)
            before = self.geometry_snapshot()
            self.switch_on()
            during = self.geometry_snapshot()
            self.app.set_reference_codes(False)
            self.pump(0.25)
            after = self.geometry_snapshot()
            self.assertGreater(len(before), 15, f"{page}: nothing measured?")
            self.assertEqual(during, before, f"{page}: layout changed when the codes were switched on")
            self.assertEqual(after, before, f"{page}: layout changed when the codes were switched off")

    def test_the_codes_do_not_add_a_row_under_the_top_bar(self):
        content_top = self.app.content_frame.winfo_rooty()
        self.switch_on()
        self.assertEqual(self.app.content_frame.winfo_rooty(), content_top)

    # ---- look ----
    def test_codes_are_shaded_not_loud(self):
        self.switch_on()
        chips = self.mapped_chips()
        self.assertGreater(len(chips), 10)
        for chip in chips:
            text, backdrop = chip.cget("text_color"), chip.cget("fg_color")
            gap = abs(brightness(text) - brightness(backdrop))
            self.assertGreater(gap, 35, f"{chip.cget('text')} legible on {backdrop}")
            self.assertLess(gap, 130, f"{chip.cget('text')} shaded, not stark, on {backdrop}")
        # ...and on the regular dark panels it is the quiet grey, dimmer than the UI's own secondary text.
        self.assertLess(brightness(self.devtools.REF_COLOUR), brightness(self.devtools_colour("text_secondary")))
        on_panel = [c for c in chips if c.cget("fg_color") in (self.devtools_colour("panel"), self.devtools_colour("bg"))]
        self.assertTrue(on_panel)
        self.assertTrue(all(c.cget("text_color") == self.devtools.shade_for(c.cget("fg_color")) for c in on_panel))

    @staticmethod
    def devtools_colour(name):
        from ui.main_window import COLORS
        return COLORS[name]

    def test_shade_follows_what_the_code_sits_on(self):
        shade = self.devtools.shade_for
        dark = shade("#1B1B1F")
        accent = shade("#FF5C39")          # the orange buttons
        self.assertGreater(brightness(dark), brightness("#1B1B1F") + 50)      # lighter than a dark panel
        self.assertLess(brightness(accent), brightness("#FF5C39") - 40)       # darker than a bright fill
        self.assertGreater(brightness(shade("#1B1B1F", 1.7)), brightness(dark), "hover is a little brighter")
        self.assertEqual(shade(None), self.devtools.REF_COLOUR)
        self.assertEqual(shade("transparent"), self.devtools.REF_COLOUR)

    def test_a_code_sitting_on_a_tab_follows_the_tab_fill(self):
        from ui.main_window import COLORS
        self.switch_on()
        by_tab = {c._resolve_info_text().split("TAB: ")[1].split("\n")[0]: c for c in self.mapped_chips() if c.kind == "tab"}
        self.assertEqual(set(by_tab), {"WORKFLOW", "CALIBRATE", "SETTINGS"})
        self.assertEqual(by_tab["WORKFLOW"].cget("fg_color"), COLORS["accent"], "WORKFLOW is the selected tab")
        self.assertEqual(by_tab["CALIBRATE"].cget("fg_color"), COLORS["panel"], "an unselected tab shows the bar")
        self.app.show_page("CALIBRATE")
        self.pump(0.2)
        self.assertEqual(by_tab["CALIBRATE"].cget("fg_color"), COLORS["accent"], "re-tinted when selected")
        self.assertEqual(by_tab["WORKFLOW"].cget("fg_color"), COLORS["panel"])

    # ---- click to copy ----
    def test_clicking_a_code_copies_its_reference_and_context(self):
        self.switch_on()
        button_chip = [c for c in self.mapped_chips() if c.kind == "button"][0]
        self.root.clipboard_clear()
        # (Tk on Windows drops synthesized mouse events, so call the bound handler; the binding itself is checked.)
        for part in (button_chip._label, button_chip._canvas):
            self.assertIn("<Button-1>", part.bind(), "a click on the chip is wired to the copy handler")
        button_chip._on_click()
        self.pump(0.1)
        copied = self.root.clipboard_get()
        self.assertTrue(copied.startswith(f"REF: {button_chip.ref_number:02d}\n"), copied)
        self.assertIn("BUTTON: ", copied)
        self.assertIn("PAGE: ", copied)
        self.assertIn("SECTION: ", copied)
        self.assertEqual(button_chip.cget("text"), "✓", "confirms the copy")
        self.pump(1.2)
        self.assertEqual(button_chip.cget("text"), f"[{button_chip.ref_number:02d}]")

    def test_a_field_code_copies_the_value_as_it_is_right_now(self):
        self.app.show_page("SETTINGS")
        self.switch_on()
        settings = self.app.pages["SETTINGS"]
        chip = [c for c in self.mapped_chips() if c.kind == "field" and "MARKER_NAME" in c._resolve_info_text()][0]
        settings.marker_name_var.set("Frag")
        chip._on_click()
        self.assertIn("VALUE: Frag", self.root.clipboard_get())
        settings.marker_name_var.set("Headshot")
        chip._on_click()
        self.assertIn("VALUE: Headshot", self.root.clipboard_get())

    def test_hovering_a_code_shows_what_it_points_at_and_leaving_hides_it(self):
        self.switch_on()
        chip = self.mapped_chips()[0]
        chip._on_enter()
        self.assertIsNotNone(chip._tooltip)
        chip._on_leave()
        self.assertIsNone(chip._tooltip)

    def test_switching_off_while_a_tooltip_is_open_closes_it(self):
        self.switch_on()
        chip = self.mapped_chips()[0]
        chip._on_enter()
        self.app.set_reference_codes(False)
        self.assertIsNone(chip._tooltip)

    # ---- two-hour limit ----
    def test_the_codes_last_two_hours(self):
        self.assertEqual(self.devtools.REFERENCE_CODES_SECONDS, 2 * 60 * 60)
        self.switch_on()
        self.assertEqual(self.app._ref_codes_deadline, self.now[0] + 7200)
        self.assertEqual(self.app.ref_codes_time_label.cget("text"), "2h 00m")

    def test_they_switch_themselves_off_after_two_hours(self):
        self.switch_on()
        self.now[0] += 7200 - 1
        self.app._update_ref_codes_countdown()
        self.assertTrue(self.app.ref_codes_on, "one second early: still on")
        self.assertEqual(self.app.ref_codes_time_label.cget("text"), "<1m")
        self.assertTrue(self.mapped_chips())

        self.now[0] += 1
        self.app._update_ref_codes_countdown()
        self.pump(0.2)
        self.assertFalse(self.app.ref_codes_on)
        self.assertFalse(self.app.ref_codes_var.get(), "the switch itself flips back")
        self.assertEqual(self.mapped_chips(), [])
        self.assertEqual(self.app.ref_codes_time_label.cget("text"), "")
        self.assertIsNone(self.app._ref_codes_job, "no countdown left running")

    def test_the_real_timer_switches_them_off_by_itself(self):
        # Real clock and real `after` timer, with the duration/tick shortened so the test is quick.
        import time
        import ui.main_window as main_window
        self.app._clock = time.time
        self.addCleanup(setattr, self.devtools, "REFERENCE_CODES_SECONDS", self.devtools.REFERENCE_CODES_SECONDS)
        self.addCleanup(setattr, main_window, "_REF_CODES_TICK_MS", main_window._REF_CODES_TICK_MS)
        self.devtools.REFERENCE_CODES_SECONDS = 0.4
        main_window._REF_CODES_TICK_MS = 60
        self.app.set_reference_codes(True)
        self.pump(0.15)
        self.assertTrue(self.app.ref_codes_on)
        self.wait_for(lambda: not self.app.ref_codes_on, timeout=5, message="the codes to switch themselves off")
        self.assertEqual(self.mapped_chips(), [])

    def test_switching_off_by_hand_cancels_the_countdown_and_on_again_restarts_the_two_hours(self):
        self.switch_on()
        self.assertIsNotNone(self.app._ref_codes_job)
        self.now[0] += 3000
        self.app.set_reference_codes(False)
        self.assertIsNone(self.app._ref_codes_job)
        self.assertIsNone(self.app._ref_codes_deadline)
        self.now[0] += 500
        self.app.set_reference_codes(True)
        self.assertEqual(self.app._ref_codes_deadline, self.now[0] + 7200, "a fresh two hours, not the remainder")

    def test_the_time_left_counts_down_in_minutes(self):
        from ui.main_window import _format_remaining
        self.assertEqual(_format_remaining(7200), "2h 00m")
        self.assertEqual(_format_remaining(7140), "1h 59m")
        self.assertEqual(_format_remaining(3601), "1h 01m")
        self.assertEqual(_format_remaining(3600), "1h 00m")
        self.assertEqual(_format_remaining(3599), "1h 00m")
        self.assertEqual(_format_remaining(600), "10m")
        self.assertEqual(_format_remaining(61), "2m")
        self.assertEqual(_format_remaining(59), "<1m")
        self.assertEqual(_format_remaining(-5), "<1m")

    # ---- the switch widget ----
    def test_clicking_the_switch_turns_the_codes_on_and_off(self):
        self.app.ref_codes_switch.toggle()
        self.pump(0.2)
        self.assertTrue(self.app.ref_codes_on)
        self.assertTrue(self.mapped_chips())
        self.app.ref_codes_switch.toggle()
        self.pump(0.2)
        self.assertFalse(self.app.ref_codes_on)
        self.assertEqual(self.mapped_chips(), [])

    # ---- not saved, and separate from Developer Mode ----
    def test_the_state_is_not_written_to_the_config(self):
        self.switch_on()
        self.app.pages["SETTINGS"].flush_pending_save(force=True)
        names = {f.name for f in dataclasses.fields(self.app.config)}
        self.assertFalse([n for n in names if "ref" in n.lower() or "code" in n.lower()], names)
        from data.config import load_config
        self.assertFalse(any("ref" in k for k in dataclasses.asdict(load_config())))

    def test_the_codes_and_developer_mode_are_independent(self):
        # Developer Mode alone: its readouts appear, the overlay codes do not.
        self.app.config.developer_mode = True
        self.app.refresh_developer_mode()
        self.pump(0.25)
        self.assertEqual(self.mapped_chips(overlay=True), [])
        readouts = self.chips(overlay=False)
        self.assertTrue(readouts)
        self.assertTrue([c for c in readouts if c.winfo_viewable()], "the ANALYSIS evidence strip shows")

        # Codes on top of it: both; then Developer Mode off: only the codes remain.
        self.switch_on()
        self.assertTrue(self.mapped_chips(overlay=True))
        self.app.config.developer_mode = False
        self.app.refresh_developer_mode()
        self.pump(0.25)
        self.assertTrue(self.mapped_chips(overlay=True))
        self.assertEqual([c for c in self.chips(overlay=False) if c.winfo_viewable()], [])

    def test_there_is_no_leftover_developer_tab_strip_taking_a_row(self):
        self.assertFalse(hasattr(self.app, "dev_tab_strip"))
        self.app.config.developer_mode = True
        self.app.refresh_developer_mode()
        self.pump(0.2)
        header_bottom = self.app.content_frame.winfo_rooty()
        self.assertLessEqual(header_bottom - self.root.winfo_rooty(), 90, "content starts right under the 56px top bar")

    # ---- rebuilt content ----
    def test_codes_in_rebuilt_panels_do_not_drift_or_pile_up(self):
        video = self.app.pages["WORKFLOW"].video_section
        self.switch_on()

        def zone_chips():
            return [c for c in video.drop_zone.winfo_children() if isinstance(c, self.devtools.DevRef)]

        first = zone_chips()
        self.assertTrue(first)
        before = [c.place_info()["y"] for c in first]
        for _ in range(3):
            video._render_loading("clip.mp4")
            video._render_idle()
        self.pump(0.2)
        again = zone_chips()
        self.assertEqual(len(again), len(first))
        self.assertEqual([c.place_info()["y"] for c in again], before)
        self.assertTrue(all(c.winfo_viewable() for c in again), "rebuilt codes follow the switch that is already on")

    def test_destroyed_chips_leave_the_registry_and_cancel_their_timers(self):
        video = self.app.pages["WORKFLOW"].video_section
        self.switch_on()
        baseline = len(self.devtools._registry)
        for _ in range(5):
            video._render_loading("clip.mp4")
            video._render_idle()
        self.pump(0.2)
        self.assertEqual(len(self.devtools._registry), baseline, "rebuilding a panel must not leak chips")

        chip = [c for c in video.drop_zone.winfo_children() if isinstance(c, self.devtools.DevRef)][0]
        chip._on_click()
        self.assertIsNotNone(chip._revert_job)
        chip._on_enter()
        video._render_loading("clip.mp4")      # destroys the chip mid-flash with its tooltip open
        self.pump(1.1)                           # the pending '✓' revert must not fire on the dead chip
        self.assertNotIn(chip, self.devtools._registry)

    def test_a_dialog_opened_while_the_codes_are_on_shows_its_codes(self):
        from ui.main_window import COLORS
        from ui.pages.settings_page import ManageCalibrationDialog
        self.switch_on()
        dialog = ManageCalibrationDialog(self.root, self.app, COLORS)
        self.pump(0.3)
        chips = [w for w in self.widgets(dialog) if isinstance(w, self.devtools.DevRef)]
        self.assertGreaterEqual(len(chips), 2)
        self.assertTrue(all(c.winfo_viewable() for c in chips))
        dialog.destroy()
        self.pump(0.1)
        self.app.set_reference_codes(False)   # must cope with the dead chips left in the registry
        self.pump(0.1)


if __name__ == "__main__":
    unittest.main()
