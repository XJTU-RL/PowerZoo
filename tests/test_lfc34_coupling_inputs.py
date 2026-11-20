"""Collector-origin regression tests without DSS or MATLAB."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.check_lfc34_coupling_inputs import _single_stage_kvar, verify_checkout

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


class SingletonShuntTest(unittest.TestCase):
    def test_native_scalar_and_single_stage_serializations(self):
        for raw in ("75", " [ 75] ", "[7.5e1]"):
            self.assertEqual(_single_stage_kvar(raw, 1, [1], 75.), 75.)

    def test_invalid_or_conflicting_kvar_rejected(self):
        for raw, bank in (("[75, 75]", 75.), ("[75 75]", 75.), ("[[75]]", 75.),
                          ("[NaN]", 75.), ("inf", 75.), ("1e400", 75.),
                          ("[75]", 74.), ("[75]", float("inf"))):
            with self.subTest(raw=raw, bank=bank), self.assertRaises(ValueError):
                _single_stage_kvar(raw, 1, [1], bank)

    def test_missing_multiple_or_disabled_native_stages_rejected(self):
        for num, states in ((2, [1, 1]), (1, [0]), (1, []), (1, [1, 0])):
            with self.subTest(num=num, states=states), self.assertRaisesRegex(ValueError, "enabled native stage"):
                _single_stage_kvar("[75]", num, states, 75.)
