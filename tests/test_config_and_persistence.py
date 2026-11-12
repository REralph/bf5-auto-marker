"""Settings (config.json) and the safe-write helpers every state file goes through."""

import json
import os
import unittest
from unittest import mock

from tests import helpers  # noqa: F401  (also puts the project root on sys.path)
from data import safe_io
from data.config import MARKER_COLORS, AppConfig, load_config, save_config
from core import crash_log


class AppConfigValidation(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = AppConfig.from_dict({})
        self.assertEqual(config, AppConfig())
        self.assertEqual(config.marker_color, "Red")

    def test_unknown_and_retired_keys_are_ignored(self):
        # the real config.json of an older version carried these
        config = AppConfig.from_dict({
            "confidence_threshold": 0.69, "frame_sampling": 10, "kill_cooldown_seconds": 2.0,
            "timeline_start_offset_seconds": -4.0, "enable_neighbor_validation": True, "marker_color": "Red",
        })
        self.assertEqual(config.confidence_threshold, 0.69)
        self.assertEqual(config.frame_sampling, 10)
        self.assertFalse(hasattr(config, "timeline_start_offset_seconds"))
        self.assertFalse(hasattr(config, "kill_cooldown_seconds"))

    def test_numbers_are_clamped_to_the_slider_ranges(self):
        config = AppConfig.from_dict({"confidence_threshold": 7, "frame_sampling": 500})
        self.assertEqual(config.confidence_threshold, 0.99)
        self.assertEqual(config.frame_sampling, 10)
        config = AppConfig.from_dict({"confidence_threshold": -3, "frame_sampling": 0})
        self.assertEqual(config.confidence_threshold, 0.50)
        self.assertEqual(config.frame_sampling, 1)

    def test_garbage_types_fall_back_to_defaults(self):
        config = AppConfig.from_dict({
            "confidence_threshold": "high", "frame_sampling": None, "enable_neighbor_validation": "yes",
            "developer_mode": 1, "marker_name": 42, "marker_color": "Chartreuse",
        })
        defaults = AppConfig()
        self.assertEqual(config.confidence_threshold, defaults.confidence_threshold)
        self.assertEqual(config.frame_sampling, defaults.frame_sampling)
        self.assertEqual(config.enable_neighbor_validation, defaults.enable_neighbor_validation)
        self.assertEqual(config.developer_mode, defaults.developer_mode)
        self.assertEqual(config.marker_name, "Kill")
        self.assertEqual(config.marker_color, "Red")

    def test_nan_and_infinity_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            config = AppConfig.from_dict({"confidence_threshold": bad})
            self.assertEqual(config.confidence_threshold, AppConfig().confidence_threshold)

    def test_marker_name_is_trimmed_and_capped(self):
        self.assertEqual(AppConfig.from_dict({"marker_name": "  Clip  "}).marker_name, "Clip")
        self.assertEqual(AppConfig.from_dict({"marker_name": "   "}).marker_name, "Kill")
        self.assertEqual(len(AppConfig.from_dict({"marker_name": "x" * 500}).marker_name), 60)

    def test_non_dict_input(self):
        self.assertEqual(AppConfig.from_dict(None), AppConfig())
        self.assertEqual(AppConfig.from_dict([1, 2]), AppConfig())

    def test_every_marker_colour_is_accepted(self):
        for colour in MARKER_COLORS:
            self.assertEqual(AppConfig.from_dict({"marker_color": colour}).marker_color, colour)


class ConfigFile(unittest.TestCase):
    def setUp(self):
        self.dir = helpers.make_temp_dir(self)
        self.path = os.path.join(self.dir, "config.json")

    def test_save_and_load_round_trip(self):
        config = AppConfig(confidence_threshold=0.83, frame_sampling=4, marker_name="Frag", marker_color="Cyan",
                           developer_mode=True)
        save_config(config, self.path)
        self.assertEqual(load_config(self.path), config)

    def test_missing_corrupt_or_wrong_shaped_file_gives_defaults(self):
        self.assertEqual(load_config(self.path), AppConfig())
        for content in ("{not json", "[1, 2, 3]", "", "null", '"text"'):
            helpers.write_text(self.path, content)
            self.assertEqual(load_config(self.path), AppConfig(), content)

    def test_failed_save_leaves_the_previous_file_untouched(self):
        save_config(AppConfig(marker_name="Original"), self.path)
        before = helpers.read_text(self.path)
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_config(AppConfig(marker_name="New"), self.path)
        self.assertEqual(helpers.read_text(self.path), before)
        self.assertEqual(load_config(self.path).marker_name, "Original")
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [], "temp file must be cleaned up")

    def test_failure_while_writing_also_keeps_the_old_file(self):
        save_config(AppConfig(marker_name="Original"), self.path)
        with mock.patch("json.dump", side_effect=OSError("no space left on device")):
            with self.assertRaises(OSError):
                save_config(AppConfig(marker_name="New"), self.path)
        self.assertEqual(load_config(self.path).marker_name, "Original")


class SafeIo(unittest.TestCase):
    def setUp(self):
        self.dir = helpers.make_temp_dir(self)

    def test_atomic_write_creates_parent_folders(self):
        path = os.path.join(self.dir, "a", "b", "state.json")
        safe_io.atomic_write_json(path, {"x": 1})
        self.assertEqual(json.loads(helpers.read_text(path)), {"x": 1})

    def test_read_json_dict_only_returns_objects(self):
        path = os.path.join(self.dir, "s.json")
        for content, expected in (('{"a": 1}', {"a": 1}), ("[1]", {}), ("nope", {}), ("", {})):
            helpers.write_text(path, content)
            self.assertEqual(safe_io.read_json_dict(path), expected)
        self.assertEqual(safe_io.read_json_dict(os.path.join(self.dir, "missing.json")), {})

    def test_as_finite_float(self):
        f = safe_io.as_finite_float
        self.assertEqual(f(0.5, 9, 0, 1), 0.5)
        self.assertEqual(f(7, 9, 0, 1), 1)
        self.assertEqual(f(-7, 9, 0, 1), 0)
        for bad in ("1", None, True, float("nan"), float("inf"), [1]):
            self.assertEqual(f(bad, 9, 0, 1), 9, repr(bad))


class CrashLog(unittest.TestCase):
    def test_log_is_capped_by_rotating_to_a_previous_file(self):
        path = os.path.join(helpers.make_temp_dir(self), "crash_log.txt")
        with mock.patch.object(crash_log, "MAX_BYTES", 2000):
            for i in range(60):
                self.assertTrue(crash_log.log_exception_text(f"error number {i}\n" + "x" * 100, path))
        previous = os.path.join(os.path.dirname(path), "crash_log.previous.txt")
        self.assertTrue(os.path.isfile(previous), "an over-size log is rotated, not grown forever")
        self.assertLess(os.path.getsize(path), 2000 + 300)
        self.assertIn("error number 59", helpers.read_text(path), "the newest report is always kept")

    def test_logging_never_raises(self):
        self.assertFalse(crash_log.log_exception_text("x", os.path.join(os.devnull, "nope", "crash.txt")))


if __name__ == "__main__":
    unittest.main()
