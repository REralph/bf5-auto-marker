"""Naming a detected skull's colour from its own pixels (the Colours column / marker note)."""

import unittest

import cv2
import numpy as np

from data.colour_label import sample_hsv, suggest_name_for_sample
from tests import helpers


def scene(skull_colour, background, size=40, w=90, h=80, seed=0):
    frame = np.empty((h, w, 3), np.uint8)
    frame[:] = background
    frame = np.clip(frame.astype(int) + helpers.noise_background((h, w, 3), seed, 14).astype(int) - 7, 0, 255).astype(np.uint8)
    helpers.draw_skull(frame, w // 2, h // 2, size, skull_colour, background=background)
    r = size // 2
    bbox = (w // 2 - r, h // 2 - r - r // 4, size, size + r // 4)
    return frame, bbox


def soft_scene(skull_bgr, bg_bgr, blur, opacity, size=40, w=90, h=80, seed=0):
    """A skull with soft, partly transparent edges -- like the kill-feed icon fading in/out over the
    scenery -- composited onto a coloured background."""
    background = np.empty((h, w, 3), np.float32)
    background[:] = bg_bgr
    background += helpers.noise_background((h, w, 3), seed, 14).astype(np.float32) - 7
    mask = np.zeros((h, w), np.uint8)
    helpers.draw_skull(mask, w // 2, h // 2, size, 255, background=0)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), blur) * opacity
    out = alpha[..., None] * np.array(skull_bgr, np.float32) + (1 - alpha[..., None]) * background
    r = size // 2
    return np.clip(out, 0, 255).astype(np.uint8), (w // 2 - r, h // 2 - r - r // 4, size, size + r // 4)


def label(skull_colour, background):
    frame, bbox = scene(skull_colour, background)
    return suggest_name_for_sample(sample_hsv(frame, bbox))


class ColourLabelTests(unittest.TestCase):
    def test_a_white_skull_is_white_whatever_is_behind_it(self):
        backgrounds = {
            "dark": (20, 20, 20), "black": (0, 0, 0), "grey": (110, 110, 110),
            "orange": (40, 110, 200), "brown": (30, 70, 120), "tan": (120, 160, 190), "green": (50, 120, 60),
        }
        for name, bgr in backgrounds.items():
            self.assertEqual(label(helpers.WHITE_BGR, bgr), "White", f"white skull on {name}")

    def test_a_gold_skull_is_gold_whatever_is_behind_it(self):
        for name, bgr in {"dark": (20, 20, 20), "black": (0, 0, 0), "brown": (30, 60, 100), "grey": (100, 100, 100)}.items():
            self.assertEqual(label(helpers.GOLD_BGR, bgr), "Gold", f"gold skull on {name}")

    def test_a_soft_semi_transparent_white_skull_over_warm_scenery_is_still_white(self):
        # Regression: these were all named "Orange"/"Red" because scenery blended into the skull's edges was
        # averaged into its colour (found on real footage: four of eighteen white-skull kills were labelled Orange).
        orange, brown, tan, red = (40, 110, 200), (30, 70, 120), (120, 160, 190), (40, 50, 190)
        cases = [(2.2, 0.70, tan), (2.2, 0.85, brown), (3.0, 0.85, brown), (3.0, 1.0, orange), (3.0, 1.0, red), (3.0, 0.70, tan)]
        for blur, opacity, background in cases:
            frame, bbox = soft_scene(helpers.WHITE_BGR, background, blur, opacity)
            name = suggest_name_for_sample(sample_hsv(frame, bbox))
            self.assertEqual(name, "White", f"blur {blur}, opacity {opacity}, background {background}")

    def test_other_colours_are_still_named(self):
        self.assertEqual(label((255, 80, 0), (15, 15, 15)), "Blue")
        self.assertEqual(label((0, 0, 255), (15, 15, 15)), "Red")
        self.assertEqual(label((0, 255, 0), (15, 15, 15)), "Green")

    def test_boxes_that_miss_or_overhang_the_frame_are_handled(self):
        frame = np.zeros((50, 60, 3), np.uint8)
        self.assertIsNone(sample_hsv(frame, (100, 100, 20, 20)))      # completely outside
        self.assertIsNone(sample_hsv(frame, (10, 10, 0, 0)))          # empty
        self.assertIsNotNone(sample_hsv(frame, (50, 40, 30, 30)))     # partly outside: clipped to the frame
        self.assertIsNotNone(sample_hsv(frame, (-5, -5, 20, 20)))

    def test_a_flat_region_still_gets_a_label(self):
        frame = np.full((40, 40, 3), 200, np.uint8)
        self.assertEqual(suggest_name_for_sample(sample_hsv(frame, (5, 5, 20, 20))), "White")

    def test_label_thresholds(self):
        self.assertEqual(suggest_name_for_sample((0.0, 5.0, 200.0)), "White")
        self.assertEqual(suggest_name_for_sample((0.0, 5.0, 80.0)), "Grey")
        self.assertEqual(suggest_name_for_sample((25.0, 240.0, 240.0)), "Gold")
        self.assertEqual(suggest_name_for_sample((38.0, 240.0, 240.0)), "Yellow")


if __name__ == "__main__":
    unittest.main()
