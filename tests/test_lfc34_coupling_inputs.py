"""Collector-origin regression tests without DSS or MATLAB."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.check_lfc34_coupling_inputs import verify_checkout

ROOT = Path(__file__).resolve().parents[1]


class CollectorOriginTest(unittest.TestCase):
    def test_own_checkout_and_helpers(self):
        self.assertEqual(verify_checkout(ROOT, ROOT), ROOT)

    def test_wrong_checkout_or_cwd(self):
        for repo, cwd in ((ROOT.parent, ROOT), (ROOT, ROOT.parent)):
            with self.subTest(repo=repo, cwd=cwd), self.assertRaisesRegex(ValueError, "own resolved"):
                verify_checkout(repo, cwd)

    def test_import_origins_are_independently_verified(self):
        for key in ("anchor", "operating"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "origin mismatch"):
                verify_checkout(ROOT, ROOT, **{key: SimpleNamespace(__file__="/tmp/wrong.py")})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "origin mismatch"):
                verify_checkout(ROOT, ROOT, **{key: SimpleNamespace()})
