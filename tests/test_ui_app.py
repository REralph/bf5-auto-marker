"""The real MainWindow end to end: navigation, the worker->UI queue, analysis, exports, sending to
Resolve, settings persistence, and the failure paths (full disk, unreadable video, Resolve gone)."""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import cv2

import core.crash_log as crash_log
import data.config as config_module
from core.session import ReferenceImage
from data.config import AppConfig, load_config
from detectors.reference_profile import build_reference_profile
from detectors.skull_detector import crop_to_content
from tests import helpers
from tests.ui_base import UiTestCase

REGION = (100, 70, 120, 100)
_clip_cache = {}


def synthetic_clip() -> str:
    """The shared 12 s test clip: gold skull 30-89, white skull 150-209 (dropout at 175/176), white flash at 270."""
    if "path" not in _clip_cache:
        directory = tempfile.mkdtemp(prefix="bf5_ui_clip_")
        _clip_cache["dir"] = directory
        frames = []
        for i in range(360):
            frame = helpers.noise_background((240, 320, 3), i)
            if 30 <= i < 90:
                helpers.draw_skull(frame, 160, 120, 60, helpers.GOLD_BGR, background=(5, 5, 5))
            elif 150 <= i < 210 and i not in (175, 176):
                helpers.draw_skull(frame, 160, 120, 60, helpers.WHITE_BGR, background=(5, 5, 5))
            elif i == 270:
                helpers.draw_skull(frame, 160, 120, 60, helpers.WHITE_BGR, background=(5, 5, 5))
            frames.append(frame)
        _clip_cache["path"] = helpers.write_clip(os.path.join(directory, "match.avi"), frames, fps=30.0)
    return _clip_cache["path"]


def tearDownModule():
    if "dir" in _clip_cache:
        shutil.rmtree(_clip_cache["dir"], ignore_errors=True)


class AnalysisReadyCase(UiTestCase):
    """Base for tests that need a loaded video and a finished calibration."""

    def setUp(self):
        super().setUp()
        self.out_dir = os.path.join(self.state_dir, "output")
        os.makedirs(self.out_dir)
        patcher = mock.patch("ui.main_window.get_output_dir", lambda video_path: self.out_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

        session = self.app.session
        session.load_video(synthetic_clip())
        session.set_detection_region(REGION, 320, 240)
        session.reference_images = []
        for label, colour in (("Gold Skull", helpers.GOLD_BGR), ("White Skull", helpers.WHITE_BGR)):
            image = helpers.make_reference_image(colour)
            session.reference_images.append(ReferenceImage(
                label=label, image=crop_to_content(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), scale=1.0,
                profile=build_reference_profile(label, image),
            ))
        self.app.config.confidence_threshold = 0.8
        self.app.config.frame_sampling = 2

    def run_analysis(self):
        self.assertTrue(self.app.start_analysis())
        self.wait_for(lambda: not self.app.analysis_running, 120, "analysis to finish")
        self.pump(0.2)


class AnalysisFlow(AnalysisReadyCase):
    def test_full_analysis_through_the_real_thread_and_queue(self):
        self.run_analysis()
        events = self.app.events
        self.assertEqual([e.colors for e in events], [["Gold"], ["White"]])
        self.assertTrue(self.app.last_analysis_completed)
        self.assertEqual(self.app.analysis_worker_status, "STOPPED")
        for name in ("kill_timestamps.csv", "kill_timestamps.txt", "detections.json", "analysis_log.txt"):
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir, name)), name)
        self.assertEqual(len(json.loads(helpers.read_text(os.path.join(self.out_dir, "detections.json")))), 2)
        self.assertIn("Analysis complete. 2 events", helpers.read_text(os.path.join(self.out_dir, "analysis_log.txt")))
        self.assertIn("2 marker(s) found", " ".join(self.texts(self.app.pages["WORKFLOW"].analyse_section.progress_panel)))

    def test_preview_page_lists_every_event_and_renders_it(self):
        self.run_analysis()
        self.app.show_page("PREVIEW")
        self.pump(0.3)
        preview = self.app.pages["PREVIEW"]
        self.assertEqual(len(preview.rows), 2)
        for index, event in enumerate(self.app.events):
            self.app.select_event(index)
            self.pump(0.1)
            self.assertIn(event.colors_display, preview.preview_panel.detail_label.cget("text"))
            self.assertEqual(preview.preview_panel.timestamp_label.cget("text"), event.timestamp_display)

    def test_sending_to_resolve_places_one_marker_per_event_at_the_start_frame(self):
        self.run_analysis()
        resolve = self.app.pages["WORKFLOW"].resolve_section
        before = len(self.toplevels())
        resolve.send_now()
        self.wait_for(lambda: "✓ MARKERS CREATED" in self.texts(resolve.send_markers_frame), 15, "send result")
        self.assertEqual(len(self.toplevels()), before, "no confirmation pop-up")
        expected = sorted(int(round(e.timestamp_seconds * 60)) for e in self.app.events)  # fake timeline runs at 60 fps
        self.assertEqual(sorted(self.timeline.markers), expected)
        self.assertEqual(self.timeline.add_calls.count(expected[0]), 1)

    def test_a_second_click_while_sending_sends_nothing_extra(self):
        self.run_analysis()
        resolve = self.app.pages["WORKFLOW"].resolve_section
        resolve.send_now()
        resolve.send_now()
        self.assertFalse(self.app.send_markers(), "app-level guard as well")
        self.wait_for(lambda: "✓ MARKERS CREATED" in self.texts(resolve.send_markers_frame), 15, "send result")
        self.assertEqual(len(self.timeline.add_calls), 2)
        # once it has finished, sending again is allowed -- and finds its own markers already there
        resolve.refresh()
        resolve.send_now()
        self.wait_for(lambda: any("already on the timeline" in t for t in self.texts(resolve.send_markers_frame)), 15, "second send result")
        self.assertEqual(len(self.timeline.add_calls), 2, "no duplicate markers were added")

    def test_send_works_on_a_snapshot_of_the_events(self):
        self.run_analysis()
        release = threading.Event()
        original = self.timeline.GetMarkers

        def slow_get_markers():
            release.wait(5)
            return original()

        self.timeline.GetMarkers = slow_get_markers
        self.assertTrue(self.app.send_markers())
        self.app.events.clear()          # the user deletes everything while the send is in flight
        release.set()
        self.wait_for(lambda: len(self.timeline.markers) == 2, 15, "both markers")
        self.assertEqual(len(self.timeline.markers), 2)

    def test_resolve_not_running_gives_a_message_and_retry_works(self):
        self.run_analysis()
        resolve = self.app.pages["WORKFLOW"].resolve_section
        from resolve.resolve_controller import ResolveConnectionError

        def boom():
            raise ResolveConnectionError("DaVinci Resolve does not appear to be running.")

        with mock.patch("ui.main_window.get_resolve", boom):
            resolve.send_now()
            self.wait_for(lambda: "COULD NOT SEND MARKERS" in self.texts(resolve.send_markers_frame), 15, "error state")
        self.assertTrue(any("does not appear to be running" in t for t in self.texts(resolve.send_markers_frame)))
        self.assertFalse(self.app._sending_markers, "can try again")
        retry = [w for w in self.widgets(resolve.send_markers_frame) if getattr(w, "cget", None) and self._text(w) == "RETRY"][0]
        retry.invoke()
        self.wait_for(lambda: "✓ MARKERS CREATED" in self.texts(resolve.send_markers_frame), 15, "retry result")

    @staticmethod
    def _text(widget):
        try:
            return widget.cget("text")
        except Exception:  # noqa: BLE001
            return None

    def test_events_added_or_deleted_by_hand_keep_numbering_consistent(self):
        self.run_analysis()
        self.app.add_manual_event(100.0)
        self.assertEqual([e.event_number for e in self.app.events], [1, 2, 3])
        self.assertEqual(self.app.events[2].timestamp_seconds, 100.0)   # sorted by time: 1.0s, 5.0s, 100.0s
        self.app.select_event(0)
        self.app.delete_selected_event()
        self.assertEqual([e.event_number for e in self.app.events], [1, 2])
        self.assertIsNone(self.app.selected_index)


class AnalysisFailures(AnalysisReadyCase):
    def test_output_folder_that_cannot_be_created_offers_to_carry_on(self):
        self.dialogs.ask_yes_no.return_value = False
        with mock.patch("ui.main_window.get_output_dir", side_effect=OSError("There is not enough space on the disk")):
            self.assertFalse(self.app.start_analysis(), "declining stops it cleanly")
            self.assertFalse(self.app.analysis_running)
            self.assertIn("space", self.dialogs.ask_yes_no.call_args.args[3])

            self.dialogs.ask_yes_no.return_value = True
            self.assertTrue(self.app.start_analysis())
            self.wait_for(lambda: not self.app.analysis_running, 120, "analysis")
        self.pump(0.2)
        self.assertEqual(len(self.app.events), 2, "the markers are found even without result files")
        self.assertTrue(self.app.last_analysis_completed)
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_failing_to_write_result_files_still_completes_the_analysis(self):
        with mock.patch("ui.main_window.write_csv", side_effect=OSError("disk full")):
            self.run_analysis()
        self.assertTrue(self.app.last_analysis_completed)
        self.assertEqual(len(self.app.events), 2)
        self.assertIn("RESULT FILES NOT SAVED", self.error_dialog_titles())
        resolve = self.app.pages["WORKFLOW"].resolve_section
        self.assertTrue(any("Markers ready: 2" in t for t in self.texts(resolve.send_markers_frame)))

    def test_log_write_failures_never_abort_the_scan(self):
        real_open = open

        def failing_open(path, mode="r", *a, **k):
            if str(path).endswith("analysis_log.txt") and "a" in mode:
                raise OSError("disk full")
            return real_open(path, mode, *a, **k)

        with mock.patch("builtins.open", failing_open):
            self.run_analysis()
        self.assertEqual(len(self.app.events), 2)

    def test_a_crash_inside_the_scan_is_reported_and_the_app_stays_usable(self):
        with mock.patch("ui.main_window.scan_video", side_effect=RuntimeError("decoder exploded")):
            self.assertTrue(self.app.start_analysis())
            self.wait_for(lambda: not self.app.analysis_running, 30, "failure to be handled")
        self.assertEqual(self.app.analysis_worker_status, "ERROR")
        self.assertIn("ANALYSIS STOPPED", self.error_dialog_titles())
        self.assertTrue(self.app.start_analysis(), "and a fresh run can start")
        self.app.cancel_analysis()
        self.wait_for(lambda: not self.app.analysis_running, 60, "second run to end")

    def test_cannot_replace_the_video_while_analysing(self):
        self.assertTrue(self.app.start_analysis())
        self.assertFalse(self.app.start_loading_video(synthetic_clip()))
        page = self.app.pages["WORKFLOW"].video_section
        page._handle_selected_path(synthetic_clip())
        self.assertTrue(self.dialogs.show_info.called)
        self.app.cancel_analysis()
        self.wait_for(lambda: not self.app.analysis_running, 60, "analysis to stop")


class QueueAndVideoLoading(UiTestCase):
    def test_every_page_is_reachable_and_only_the_visible_one_is_mapped(self):
        for name in ("CALIBRATE", "SETTINGS", "PREVIEW", "WORKFLOW"):
            self.app.show_page(name)
            self.pump(0.15)
            self.assertEqual([n for n, p in self.app.pages.items() if p.winfo_ismapped()], [name])

    def test_the_video_drop_zone_spans_the_full_width_of_its_stage_but_stays_low(self):
        video = self.app.pages["WORKFLOW"].video_section
        self.pump(0.3)
        body = video.master                      # the stage card's body: the width the zone should fill
        for state in ("idle", "loading", "idle"):
            if state == "loading":
                video._render_loading("clip.mp4")
            else:
                video._render_idle()
            self.pump(0.2)
            zone_left, zone_right = video.drop_zone.winfo_rootx(), video.drop_zone.winfo_rootx() + video.drop_zone.winfo_width()
            body_left, body_right = body.winfo_rootx(), body.winfo_rootx() + body.winfo_width()
            self.assertLessEqual(abs(zone_left - body_left), 2, f"{state}: starts at the stage's left edge")
            self.assertLessEqual(abs(zone_right - body_right), 2, f"{state}: reaches the stage's right edge")
            self.assertLess(video.drop_zone.winfo_height(), 140, f"{state}: still compact in height")
        # ...and it follows the window when it grows.
        before = video.drop_zone.winfo_width()
        self.root.geometry("1500x860")
        self.pump(0.6)
        self.assertGreater(video.drop_zone.winfo_width(), before + 100)

    def test_workflow_page_has_no_title_or_tagline(self):
        texts = self.texts(self.app.pages["WORKFLOW"])
        self.assertNotIn("WORKFLOW", texts)
        self.assertFalse(any("Load a clip" in t for t in texts))

    def test_shutdown_stops_the_windows_own_timers_and_is_safe_to_repeat(self):
        self.pump(0.3)
        self.assertIsNotNone(self.app._poll_job, "queue polling is running")
        self.app.set_reference_codes(True)
        self.assertIsNotNone(self.app._ref_codes_job)
        self.app.shutdown()
        self.assertIsNone(self.app._poll_job)
        self.assertIsNone(self.app._ref_codes_job)
        ran = []
        self.app.post_to_main(lambda: ran.append("processed"))
        self.pump(0.4)
        self.assertEqual(ran, [], "polling really stopped")
        self.app.shutdown()   # a second call does nothing and raises nothing

    def test_queue_polling_survives_a_handler_that_raises(self):
        ran = []

        def explode():
            raise ValueError("handler bug")

        self.app.post_to_main(explode)
        self.app.post_to_main(lambda: ran.append("after the failure"))
        self.wait_for(lambda: ran, 5, "the message after the failing one")
        self.app.post_to_main(lambda: ran.append("later still"))
        self.wait_for(lambda: len(ran) == 2, 5, "polling to continue")
        self.assertEqual(self.dialogs.show_error.call_count, 1)
        self.assertIn("handler bug", helpers.read_text(crash_log.CRASH_LOG_PATH))

    def test_the_same_kind_of_failure_only_pops_one_dialog(self):
        for _ in range(3):
            self.app.post_to_main(lambda: 1 / 0)
        self.pump(0.6)
        self.assertEqual(self.dialogs.show_error.call_count, 1)
        self.assertEqual(helpers.read_text(crash_log.CRASH_LOG_PATH).count("ZeroDivisionError"), 3, "but all are logged")

    def test_unreadable_video_clears_the_loading_state_and_says_why(self):
        bad = os.path.join(self.state_dir, "broken.mkv")
        helpers.write_text(bad, "not a video")
        self.assertTrue(self.app.start_loading_video(bad))
        self.wait_for(lambda: not self.app._video_loading, 10, "loading to end")
        self.assertIn("VIDEO COULD NOT BE OPENED", self.error_dialog_titles())
        self.assertFalse(self.app.session.is_loaded)
        self.assertTrue(self.app.start_loading_video(synthetic_clip()), "and another video can be loaded afterwards")
        self.wait_for(lambda: not self.app._video_loading, 10, "second load")
        self.assertTrue(self.app.session.is_loaded)

    def test_an_unexpected_error_while_opening_also_clears_the_loading_state(self):
        with mock.patch.object(type(self.app.session), "load_video", side_effect=RuntimeError("codec library crashed")):
            self.assertTrue(self.app.start_loading_video(synthetic_clip()))
            self.wait_for(lambda: not self.app._video_loading, 10, "loading to end")
        self.assertIn("VIDEO COULD NOT BE OPENED", self.error_dialog_titles())
        self.assertTrue(self.app.start_loading_video(synthetic_clip()), "the app is not stuck on 'Loading...'")
        self.wait_for(lambda: not self.app._video_loading, 10, "second load")

    def test_loading_a_video_resets_events_but_stays_on_the_workflow_page(self):
        self.app.events.append(object())
        self.app.start_loading_video(synthetic_clip())
        self.wait_for(lambda: self.app.session.is_loaded and not self.app._video_loading, 10, "load")
        self.pump(0.2)
        self.assertEqual(self.app.events, [])
        self.assertEqual(self.app.current_page_name, "WORKFLOW")

    def test_not_ready_analysis_says_exactly_what_is_missing(self):
        readiness = self.app.check_analysis_readiness()
        self.assertEqual((readiness.ready, readiness.state), (False, "VIDEO_LOAD_ERROR"))
        self.app.session.load_video(synthetic_clip())
        self.assertEqual(self.app.check_analysis_readiness().state, "CALIBRATION_ERROR")

    def test_closing_the_window_saves_a_setting_changed_a_moment_ago(self):
        settings = self.app.pages["SETTINGS"]
        settings.multi_skull_var.set(False)
        settings._on_multi_skull_changed()        # a debounced save is now pending
        self.assertTrue(self.app.config.enable_neighbor_validation)
        self.assertFalse(os.path.exists(config_module.CONFIG_PATH))
        self.app._on_close()
        self.assertFalse(load_config().multiple_skull_detection)

    def test_closing_with_nothing_changed_does_not_rewrite_the_settings_file(self):
        self.assertFalse(os.path.exists(config_module.CONFIG_PATH))
        self.app._on_close()
        self.assertFalse(os.path.exists(config_module.CONFIG_PATH))


class SettingsBehaviour(UiTestCase):
    def test_changes_apply_at_once_and_are_written_after_a_pause(self):
        settings = self.app.pages["SETTINGS"]
        settings._on_confidence_changed(0.9)
        settings._on_sampling_changed(7)
        self.assertEqual((self.app.config.confidence_threshold, self.app.config.frame_sampling), (0.9, 7))
        self.assertFalse(os.path.exists(config_module.CONFIG_PATH), "not written on every slider step")
        self.pump(0.7)
        saved = load_config()
        self.assertEqual((saved.confidence_threshold, saved.frame_sampling), (0.9, 7))

    def test_a_slider_drag_writes_the_file_once(self):
        settings = self.app.pages["SETTINGS"]
        with mock.patch("ui.pages.settings_page.save_config", wraps=config_module.save_config) as save:
            for value in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
                settings._on_confidence_changed(value)
            self.pump(0.7)
        self.assertEqual(save.call_count, 1)

    def test_marker_name_and_colour_apply_without_pressing_save(self):
        settings = self.app.pages["SETTINGS"]
        settings.marker_name_var.set("Frag")
        settings.marker_color_var.set("Cyan")
        self.assertEqual((self.app.config.marker_name, self.app.config.marker_color), ("Frag", "Cyan"))
        settings.marker_name_var.set("   ")
        self.assertEqual(self.app.config.marker_name, "Kill", "a blank name never reaches Resolve")
        self.pump(0.7)
        self.assertEqual(load_config().marker_color, "Cyan")

    def test_save_settings_button_writes_immediately_and_confirms(self):
        settings = self.app.pages["SETTINGS"]
        settings.marker_name_var.set("Clip")
        settings._save()
        self.assertEqual(load_config().marker_name, "Clip")
        self.assertTrue(self.dialogs.show_info.called)

    def test_a_failing_save_is_reported_once_and_the_settings_still_apply(self):
        settings = self.app.pages["SETTINGS"]
        with mock.patch("os.replace", side_effect=OSError("There is not enough space on the disk")):
            for value in (0.6, 0.7):
                settings._on_confidence_changed(value)
                settings.flush_pending_save()
        self.assertEqual(self.app.config.confidence_threshold, 0.7)
        self.assertEqual(self.error_dialog_titles().count("SETTINGS COULD NOT BE SAVED"), 1)
        settings._on_confidence_changed(0.8)       # space freed: the next change saves normally
        self.assertTrue(settings.flush_pending_save())
        self.assertEqual(load_config().confidence_threshold, 0.8)

    def test_the_dead_cooldown_control_is_gone(self):
        settings = self.app.pages["SETTINGS"]
        self.assertFalse(hasattr(settings, "cooldown_value_label"))
        self.assertFalse(any("Kill cooldown" in t for t in self.texts(settings)))

    def test_developer_mode_toggle_persists_immediately(self):
        settings = self.app.pages["SETTINGS"]
        settings.developer_mode_var.set(True)
        settings._on_developer_mode_changed()
        self.assertTrue(load_config().developer_mode)

    def test_a_hand_damaged_config_file_does_not_stop_the_app_starting(self):
        helpers.write_text(config_module.CONFIG_PATH, '{"confidence_threshold": "lots", "frame_sampling": 9000}')
        self.assertEqual(load_config(), AppConfig(frame_sampling=10))


class ManualMarkerDialog(UiTestCase):
    def open_dialog(self, max_seconds=60.0):
        from ui.add_marker_dialog import AddMarkerDialog
        got = []
        dialog = AddMarkerDialog(self.root, self.app_colors(), got.append, max_seconds=max_seconds)
        self.pump(0.1)
        return dialog, got

    def app_colors(self):
        from ui.main_window import COLORS
        return COLORS

    def test_valid_time_is_accepted(self):
        dialog, got = self.open_dialog()
        dialog.entry_var.set("0:42.5")
        dialog._confirm()
        self.assertEqual(got, [42.5])

    def test_nan_infinity_negative_and_text_are_rejected_with_a_message(self):
        dialog, got = self.open_dialog()
        for bad in ("nan", "inf", "-3", "abc", "", "1:2:3:4"):
            dialog.entry_var.set(bad)
            dialog._confirm()
        self.assertEqual(got, [])
        self.assertEqual(self.dialogs.show_error.call_count, 6)
        dialog.destroy()

    def test_a_time_after_the_end_of_the_video_is_rejected(self):
        dialog, got = self.open_dialog(max_seconds=60.0)
        dialog.entry_var.set("1:30")
        dialog._confirm()
        self.assertEqual(got, [])
        self.assertIn("Timestamp Beyond The Video", self.error_dialog_titles())
        dialog.destroy()


class ResolvePageThreading(UiTestCase):
    def test_status_checks_report_back_on_the_main_thread_only(self):
        resolve = self.app.pages["WORKFLOW"].resolve_section
        main_thread = threading.get_ident()
        seen = []
        original = type(resolve)._apply_status

        def recording_apply(self_, status):
            seen.append(threading.get_ident())
            return original(self_, status)

        worker_threads = []
        from resolve.resolve_controller import ResolveStatus

        def slow_check():
            worker_threads.append(threading.get_ident())
            time.sleep(0.2)
            return ResolveStatus(True, "Connected", "P", "T")

        with mock.patch.object(type(resolve), "_apply_status", recording_apply), \
                mock.patch("ui.pages.resolve_page.check_connection_status", slow_check):
            resolve._refresh_status_async()
            resolve._refresh_status_async()   # while one is in flight: ignored
            self.wait_for(lambda: seen, 5, "status result")
        self.assertEqual(seen, [main_thread])
        self.assertEqual(len(worker_threads), 1, "no pile-up of status threads")
        self.assertNotEqual(worker_threads[0], main_thread)

    def test_a_status_check_that_raises_becomes_a_disconnected_status(self):
        resolve = self.app.pages["WORKFLOW"].resolve_section
        with mock.patch("ui.pages.resolve_page.check_connection_status", side_effect=OSError("COM error")):
            resolve._refresh_status_async()
            self.wait_for(lambda: not resolve._status_check_running, 5, "check to finish")
        self.assertEqual(resolve.status_label.cget("text"), "Resolve Not Connected")
        self.assertIn("COM error", resolve.connection_detail_label.cget("text"))

    def test_reconnect_that_raises_still_ends_the_connecting_state(self):
        resolve = self.app.pages["WORKFLOW"].resolve_section
        with mock.patch("ui.pages.resolve_page.run_full_diagnostic", side_effect=RuntimeError("boom")):
            resolve._on_reconnect()
            self.wait_for(lambda: resolve.reconnect_btn.cget("text") == "RECONNECT", 5, "reconnect to end")
        self.assertTrue(any("boom" in t for t in self.texts(resolve.diag_results_frame)))


if __name__ == "__main__":
    unittest.main()
