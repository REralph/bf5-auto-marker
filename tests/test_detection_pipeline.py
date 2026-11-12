"""Reference profiles, the colour matcher, the combined detector, and the whole scan loop on synthetic
footage (small generated clips -- no real recordings, nothing written beside real files)."""

import os
import unittest

import cv2
import numpy as np

from data.config import AppConfig
from detectors.colour_matcher import ColourReferenceSet, ColourSilhouetteMatcher, learn_footage_appearance
from detectors.combined_detector import build_detector
from detectors.reference_profile import build_reference_profile
from detectors.skull_detector import crop_to_content
from tests import helpers
from video.video_probe import VideoOpenError, probe_video
from video.video_scanner import scan

BAND = (0.75, 1.35, 0.9, 1.1)


def gray_template(color):
    return crop_to_content(cv2.cvtColor(helpers.make_reference_image(color), cv2.COLOR_BGR2GRAY))


def frame_with_skull(color, seed=0, size=60, w=120, h=100, cx=None, cy=None):
    frame = helpers.noise_background((h, w, 3), seed)
    helpers.draw_skull(frame, cx or w // 2, cy or h // 2, size, color, background=(5, 5, 5))
    return frame


class ReferenceProfileTests(unittest.TestCase):
    def test_gold_reference_keeps_its_colour_and_loses_only_the_background(self):
        profile = build_reference_profile("Gold", helpers.make_reference_image(helpers.GOLD_BGR))
        sig = profile.signature
        self.assertEqual(sig.description, "Gold / Yellow")
        self.assertTrue(sig.chromatic)
        self.assertGreater(sig.saturation, 200)
        self.assertTrue(profile.has_source_colour)
        self.assertTrue(profile.background_removed)
        self.assertEqual(int(profile.bgr[profile.mask == 0].max()), 0, "background pixels are blank")
        skull_pixels = profile.bgr[profile.mask > 0]
        self.assertGreater(float(np.mean(skull_pixels[:, 2])), 200, "the skull itself keeps its yellow (high red)")
        self.assertEqual(profile.size, (63, 70))

    def test_white_reference_is_achromatic(self):
        profile = build_reference_profile("White", helpers.make_reference_image(helpers.WHITE_BGR))
        self.assertEqual(profile.signature.description, "White")
        self.assertFalse(profile.signature.chromatic)

    def test_grayscale_source_is_reported_honestly(self):
        gray = cv2.cvtColor(helpers.make_reference_image(helpers.WHITE_BGR), cv2.COLOR_BGR2GRAY)
        self.assertFalse(build_reference_profile("W", gray).has_source_colour)

    def test_all_representations_coexist(self):
        profile = build_reference_profile("Gold", helpers.make_reference_image(helpers.GOLD_BGR))
        h, w = profile.mask.shape
        for representation in (profile.gray(), profile.edges(), profile.contrast_normalised(), profile.binary(), profile.inverted()):
            self.assertEqual(representation.shape, (h, w))
        self.assertGreater(float(profile.edges().max()), 0)

    def test_real_transparency_defines_the_silhouette(self):
        base = helpers.make_reference_image(helpers.GOLD_BGR)
        silhouette = (base.sum(axis=2) > 0).astype(np.uint8) * 255
        rgba = np.dstack([np.where(silhouette[..., None] > 0, base, 255), silhouette])  # hidden white background
        profile = build_reference_profile("Gold", rgba)
        self.assertEqual(profile.signature.description, "Gold / Yellow")
        self.assertTrue(profile.background_removed)

    def test_an_image_with_nothing_to_segment_falls_back_instead_of_crashing(self):
        profile = build_reference_profile("flat", np.full((40, 40, 3), 128, np.uint8))
        self.assertFalse(profile.background_removed)


class ColourMatcherTests(unittest.TestCase):
    def setUp(self):
        self.gold = build_reference_profile("Gold Skull", helpers.make_reference_image(helpers.GOLD_BGR))
        self.matcher = ColourSilhouetteMatcher([self.gold], {"Gold Skull": BAND})

    def best(self, frame):
        candidates = self.matcher.find_candidates(frame)
        return max(candidates, key=lambda c: c.confidence) if candidates else None

    def test_finds_a_gold_skull_with_every_signal_reported(self):
        hit = self.best(frame_with_skull(helpers.GOLD_BGR))
        self.assertGreater(hit.confidence, 0.85)
        self.assertEqual(hit.profile, "Gold Skull")
        for signal in ("structure", "colour", "edge", "luminance", "size", "aspect", "position"):
            self.assertIn(signal, hit.signals)
        self.assertEqual(hit.reason, "")
        self.assertIn("Gold Skull: conf", hit.as_line())

    def test_a_white_skull_of_the_same_shape_is_not_gold(self):
        self.assertIsNone(self.best(frame_with_skull(helpers.WHITE_BGR)))

    def test_nothing_in_an_empty_frame(self):
        self.assertEqual(self.matcher.find_candidates(helpers.noise_background((100, 120, 3), 1)), [])

    def test_a_much_smaller_skull_is_scored_down_by_the_size_signal(self):
        small = self.best(frame_with_skull(helpers.GOLD_BGR, size=34))
        normal = self.best(frame_with_skull(helpers.GOLD_BGR))
        self.assertTrue(small is None or small.confidence < normal.confidence)

    def test_survives_brightness_and_compression_changes(self):
        frame = frame_with_skull(helpers.GOLD_BGR)
        darker = (frame.astype(np.float32) * 0.8).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertGreater(self.best(darker).confidence, 0.8)
        self.assertGreater(self.best(compressed).confidence, 0.8)

    def test_footage_learning_measures_size_and_colour(self):
        observation = learn_footage_appearance(self.gold, [frame_with_skull(helpers.GOLD_BGR, size=66)])
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(observation.scale, 1.1, delta=0.1)
        self.assertGreater(observation.lab_b, 180)  # strongly yellow
        self.assertIsNone(learn_footage_appearance(self.gold, [helpers.noise_background((100, 120, 3), 2)]))


class CombinedDetectorTests(unittest.TestCase):
    def setUp(self):
        self.gold = build_reference_profile("Gold Skull", helpers.make_reference_image(helpers.GOLD_BGR))
        self.white = build_reference_profile("White Skull", helpers.make_reference_image(helpers.WHITE_BGR))
        colour = ColourReferenceSet(matching=[self.gold], scale_bands={"Gold Skull": BAND}, labelling=[self.gold, self.white])
        templates = [("Gold Skull", gray_template(helpers.GOLD_BGR)), ("White Skull", gray_template(helpers.WHITE_BGR))]
        self.detector = build_detector(templates, colour)

    def test_each_skull_is_named_by_its_own_colour(self):
        gold = self.detector.find_matches(frame_with_skull(helpers.GOLD_BGR), 0.7)
        white = self.detector.find_matches(frame_with_skull(helpers.WHITE_BGR), 0.7)
        self.assertEqual([m.symbol_name for m in gold], ["Gold Skull"])
        self.assertEqual([m.symbol_name for m in white], ["White Skull"])

    def test_no_detection_without_a_skull(self):
        self.assertEqual(self.detector.find_matches(helpers.noise_background((100, 120, 3), 5), 0.5), [])

    def test_nearly_caught_candidates_are_recorded_for_developer_mode(self):
        # a gold skull the threshold is set just above: not a detection, but the evidence explains why
        match = self.detector.find_matches(frame_with_skull(helpers.GOLD_BGR), 1.01)
        self.assertEqual(match, [])
        self.assertIsNotNone(self.detector.last_rejected)
        self.assertIn("confidence below threshold", self.detector.last_rejected.reason)

    def test_without_colour_information_it_still_detects_like_the_original(self):
        plain = build_detector([("White Skull", gray_template(helpers.WHITE_BGR))], None)
        self.assertEqual(len(plain.find_matches(frame_with_skull(helpers.WHITE_BGR), 0.7)), 1)


class ScanLoopTests(unittest.TestCase):
    """One synthetic clip: a gold skull (frames 30-89), a white skull with a 2-frame dropout
    (150-209), and a single-frame white flash (270). Expect exactly two events."""

    @classmethod
    def setUpClass(cls):
        import shutil
        import tempfile
        cls.dir = tempfile.mkdtemp(prefix="bf5_scan_")
        cls.addClassCleanup(shutil.rmtree, cls.dir, True)
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
        cls.path = helpers.write_clip(os.path.join(cls.dir, "clip.avi"), frames, fps=30.0)
        cls.metadata = probe_video(cls.path)

        gold = build_reference_profile("Gold Skull", helpers.make_reference_image(helpers.GOLD_BGR))
        white = build_reference_profile("White Skull", helpers.make_reference_image(helpers.WHITE_BGR))
        cls.templates = [("Gold Skull", gray_template(helpers.GOLD_BGR)), ("White Skull", gray_template(helpers.WHITE_BGR))]
        cls.colour = ColourReferenceSet(matching=[gold], scale_bands={"Gold Skull": BAND}, labelling=[gold, white])
        cls.region = (100, 70, 120, 100)   # around the skull; its origin is (100, 70) in the full frame

    def run_scan(self, **overrides):
        settings = dict(confidence_threshold=0.8, frame_sampling=2, enable_neighbor_validation=True)
        settings.update({k: v for k, v in overrides.items() if k in AppConfig.__dataclass_fields__})
        extra = {k: v for k, v in overrides.items() if k not in AppConfig.__dataclass_fields__}
        return scan(self.path, self.metadata, self.region, self.templates, AppConfig(**settings),
                    colour_reference=self.colour, **extra)

    def test_probe_reads_the_clip(self):
        self.assertEqual((self.metadata.width, self.metadata.height, self.metadata.frame_count), (320, 240, 360))
        self.assertAlmostEqual(self.metadata.fps, 30.0, places=1)

    def test_one_event_per_appearance_and_the_flash_is_rejected(self):
        events = self.run_scan()
        self.assertEqual(len(events), 2, [(e.timestamp_seconds, e.symbols) for e in events])
        gold, white = events
        self.assertEqual((gold.symbols, gold.colors), (["Gold Skull"], ["Gold"]))
        self.assertEqual((white.symbols, white.colors), (["White Skull"], ["White"]))
        self.assertEqual([e.event_number for e in events], [1, 2])

    def test_dropped_frames_inside_an_appearance_do_not_split_it(self):
        white = self.run_scan()[1]
        self.assertLessEqual(white.frame_number, 152)
        self.assertGreaterEqual(white.event_end_frame, 206)

    def test_bookmark_is_the_start_of_the_appearance_not_its_confirmation(self):
        for event, true_start in zip(self.run_scan(), (30, 150)):
            self.assertAlmostEqual(event.frame_number, true_start, delta=3, msg="bookmark = when the skull appeared")
            self.assertAlmostEqual(event.timestamp_seconds, event.frame_number / 30.0)
            self.assertLessEqual(event.frame_number, event.best_frame)
            self.assertLessEqual(event.best_frame, event.event_end_frame)
            self.assertLess(event.event_end_frame, event.confirmation_frame)
            self.assertEqual(event.preview_frame, event.best_frame)

    def test_sighting_boxes_are_full_frame_coordinates(self):
        for event in self.run_scan():
            x, y, w, h = event.skulls[0].bbox
            centre = (x + w / 2, y + h / 2)
            self.assertAlmostEqual(centre[0], 160, delta=15)
            self.assertAlmostEqual(centre[1], 120, delta=20)

    def test_neighbour_validation_is_what_rejects_the_single_frame_flash(self):
        self.assertEqual(len(self.run_scan(enable_neighbor_validation=False)), 3)

    def test_a_threshold_above_any_possible_score_finds_nothing(self):
        self.assertEqual(self.run_scan(confidence_threshold=1.01), [])

    def test_progress_and_event_callbacks(self):
        reports, seen = [], []
        events = self.run_scan(progress_callback=lambda *args: reports.append(args), on_event=seen.append)
        self.assertEqual(len(seen), len(events))
        self.assertTrue(reports)
        frames = [r[1] for r in reports]
        self.assertEqual(frames, sorted(frames), "progress never goes backwards")
        self.assertEqual(reports[-1][0], 100.0)
        self.assertEqual(reports[-1][2], len(events))

    def test_cancelling_stops_early_without_error(self):
        calls = {"n": 0}

        def cancel_soon():
            calls["n"] += 1
            return calls["n"] > 100

        events = self.run_scan(should_cancel=cancel_soon)
        self.assertEqual([e.symbols for e in events], [["Gold Skull"]], "stopped before the white appearance, gold kept")

    def test_a_skull_still_visible_when_the_video_ends_is_not_dropped(self):
        frames = [helpers.noise_background((240, 320, 3), i) for i in range(60)]
        for i in range(30, 60):
            helpers.draw_skull(frames[i], 160, 120, 60, helpers.WHITE_BGR, background=(5, 5, 5))
        path = helpers.write_clip(os.path.join(self.dir, "ends_visible.avi"), frames)
        events = scan(path, probe_video(path), self.region, self.templates,
                      AppConfig(confidence_threshold=0.8, frame_sampling=2), colour_reference=self.colour)
        self.assertEqual(len(events), 1)

    def test_unopenable_video_raises_a_clear_error(self):
        with self.assertRaises(VideoOpenError):
            scan(os.path.join(self.dir, "missing.avi"), self.metadata, self.region, self.templates, AppConfig())
        with self.assertRaises(VideoOpenError):
            probe_video(os.path.join(self.dir, "missing.avi"))

    def test_a_corrupt_file_is_refused_not_crashed_on(self):
        bad = os.path.join(self.dir, "bad.avi")
        helpers.write_text(bad, "this is not a video")
        with self.assertRaises(VideoOpenError):
            probe_video(bad)


if __name__ == "__main__":
    unittest.main()
