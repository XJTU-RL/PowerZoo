"""Shared Space resources must stay deployable and consistent without imports."""

import tempfile
import unittest
from pathlib import Path

from tools.sync_hf_shared_assets import DEMO_SPACES, synchronize


class SharedSpaceAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spaces = self.root / "huggingface_spaces"
        self.main = self.spaces / "powerzoo"
        (self.main / "data").mkdir(parents=True)
        (self.main / "assets").mkdir()
        (self.main / "data" / "sample.json").write_bytes(b'{"value":1}')
        (self.main / "assets" / "architecture.svg").write_bytes(b"<svg/>")
        (self.main / "requirements.txt").write_bytes(b"main-requirements\n")
        demo = self.spaces / "powerzoo-vvc"
        demo.mkdir()
        (demo / "requirements.txt").write_bytes(b"demo-requirements\n")

    def test_check_does_not_write_and_refresh_keeps_independent_copies(self):
        self.assertEqual(len(synchronize(self.root)), 12)
        self.assertFalse((self.spaces / "powerzoo-zh").exists())
        synchronize(self.root, check=False)
        self.assertEqual(synchronize(self.root), [])
        for name in DEMO_SPACES:
            self.assertEqual(
                (self.spaces / name / "requirements.txt").read_bytes(),
                b"demo-requirements\n",
            )
        self.assertEqual(
            (self.spaces / "powerzoo-zh" / "requirements.txt").read_bytes(),
            b"main-requirements\n",
        )
        (self.main / "data" / "sample.json").write_bytes(b'{"value":2}')
        self.assertEqual(len(synchronize(self.root)), 1)
        synchronize(self.root, check=False)
        self.assertEqual(synchronize(self.root), [])

    def test_stale_json_is_detected_and_removed_without_touching_localized_app(self):
        synchronize(self.root, check=False)
        translated = self.spaces / "powerzoo-zh"
        stale = translated / "data" / "obsolete.json"
        stale.write_bytes(b"{}")
        app = translated / "app.py"
        app.write_text("# localized app\n", encoding="utf-8")
        self.assertEqual(len(synchronize(self.root)), 1)
        self.assertTrue(stale.exists())
        synchronize(self.root, check=False)
        self.assertFalse(stale.exists())
        self.assertEqual(app.read_text(), "# localized app\n")

    def test_missing_canonical_file_fails_before_any_write(self):
        (self.main / "assets" / "architecture.svg").unlink()
        with self.assertRaises(FileNotFoundError):
            synchronize(self.root, check=False)
        self.assertFalse((self.spaces / "powerzoo-zh").exists())


if __name__ == "__main__":
    unittest.main()
