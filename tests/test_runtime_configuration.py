import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from detection.main import MarkerConfigWatcher, apply_marker_appearances


class MarkerReloadTests(unittest.TestCase):
    def test_updates_heights_and_removes_old_appearance_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.json"
            path.write_text(json.dumps({"4": {"text": "custom", "height_m": 0.2},
                                        "8": {"text": "another"}}))
            watcher = MarkerConfigWatcher(path)
            self.assertTrue(watcher.poll(0))
            defaults = {4: {"text": "default", "color": (1, 2, 3)}}
            plotter = SimpleNamespace(marker_overrides={}, marker_images={"old": object()})
            apply_marker_appearances(plotter, defaults, watcher.current[0])
            self.assertEqual(plotter.marker_overrides[4]["text"], "custom")
            self.assertEqual(watcher.current[1], {4: 0.2})
            self.assertFalse(plotter.marker_images)
            path.write_text("{}")
            self.assertFalse(watcher.poll(0.1))
            self.assertTrue(watcher.poll(0.6))
            apply_marker_appearances(plotter, defaults, watcher.current[0])
            self.assertEqual(plotter.marker_overrides, defaults)
            self.assertEqual(watcher.current[1], {})
            self.assertEqual(defaults[4]["text"], "default")

    def test_partial_saves_and_missing_files_retain_last_good_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.json"
            watcher = MarkerConfigWatcher(path)
            self.assertFalse(watcher.poll(0))
            self.assertEqual(watcher.current, ({}, {}))
            path.write_text('{"2": {"text": "working"}}')
            self.assertTrue(watcher.poll(1))
            path.write_text('{"2":')
            self.assertFalse(watcher.poll(2))
            self.assertEqual(watcher.current[0][2]["text"], "working")
            path.unlink()
            self.assertFalse(watcher.poll(3))
            self.assertEqual(watcher.current[0][2]["text"], "working")
            replacement = Path(directory) / "save.tmp"
            replacement.write_text('{"2": {"text": "recovered", "height_m": 0.3}}')
            replacement.replace(path)
            self.assertTrue(watcher.poll(4))
            self.assertEqual(watcher.current[0][2]["text"], "recovered")
            self.assertEqual(watcher.current[1][2], 0.3)

    def test_new_image_and_scale_replace_cached_sprite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.json"
            for name, value in (("first.png", 40), ("second.png", 180)):
                image = np.full((4, 4, 4), value, np.uint8)
                image[:, :, 3] = 128
                cv2.imwrite(str(path.parent / name), image)
            path.write_text('{"8": {"image": "first.png", "scale": 1}}')
            watcher = MarkerConfigWatcher(path)
            self.assertTrue(watcher.poll(0))
            plotter = SimpleNamespace(marker_overrides={}, marker_images={})
            apply_marker_appearances(plotter, {}, watcher.current[0])
            old = plotter.marker_overrides[8]["image"]
            plotter.marker_images[("image", id(old))] = old
            path.write_text('{"8": {"image": "second.png", "scale": 2}}')
            self.assertTrue(watcher.poll(1))
            apply_marker_appearances(plotter, {}, watcher.current[0])
            self.assertFalse(plotter.marker_images)
            self.assertEqual(plotter.marker_overrides[8]["scale"], 2)
            self.assertEqual(plotter.marker_overrides[8]["image"][0, 0].tolist(), [180, 180, 180, 128])


if __name__ == "__main__":
    unittest.main()
