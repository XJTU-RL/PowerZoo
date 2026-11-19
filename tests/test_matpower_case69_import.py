"""Pure Python tests; the licensed case matrix and native solver are not fixtures."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_matpower_case69.py"
SPEC = importlib.util.spec_from_file_location("case69_import", SCRIPT)
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


class TestCase69Import(unittest.TestCase):
    def test_refuses_unpinned_content(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "case69.m"
            file.write_text("mpc.bus = [];", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned"):
                IMPORTER.read_pinned_case(file)

    def test_matrix_parser_rejects_nonfinite_or_bad_width(self):
        for source in ("mpc.bus = [1 2 3];", "mpc.bus = [1 nan];"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                IMPORTER.matrix(source, "bus", 2)

    def test_reference_two_bus_has_expected_voltage_and_loss_units(self):
        source = [1, 3, 0, 0, 0, 0, 1, 1, 0, 10, 1, 1, 1]
        load = [2, 1, 1000, 0, 0, 0, 1, 1, 0, 10, 1, 1.2, 0.8]
        voltage, loss_kw, _ = IMPORTER.solve_reference(
            [source, load], {2: (1, complex(1, 0))}, [1, 2]
        )
        # 1 ohm / 10 ohm base; constant 0.1 p.u. demand. V^2 − V + .01 = 0.
        expected_voltage = (1 + 0.96**0.5) / 2
        self.assertAlmostEqual(voltage["b2"], expected_voltage)
        self.assertAlmostEqual(loss_kw, 1000 * (1 / expected_voltage - 1))

    def test_deck_keeps_nominal_physical_units(self):
        buses = [
            [1, 3, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1, 1],
            [2, 1, 4, 2.5, 0, 0, 1, 1, 0, 12.66, 1, 1.2, 0.8],
        ]
        row = [1, 2, 0.075, 0.05, 0, 0, 0, 0, 0, 0, 1, -360, 360]
        deck = IMPORTER.dss_text(buses, [row])
        self.assertIn("r1=0.075 x1=0.05", deck)
        self.assertIn("kv=12.66 kw=4 kvar=2.5", deck)
        self.assertIn("Set ControlMode=Off", deck)
        self.assertNotIn("New Capacitor", deck)
        self.assertNotIn("New PVSystem", deck)


    def test_native_changing_cwd_does_not_break_relative_output(self):
        bus = [1, 3, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1, 1]

        def native_changes_directory(deck, *_):
            self.assertTrue(deck.is_absolute())
            os.chdir(deck.parent)
            return {"buses": 69}

        caller = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            try:
                os.chdir(base)
                with (
                    patch("sys.argv", ["importer", "unused", "case", "--verify-native"]),
                    patch.object(IMPORTER, "read_pinned_case", return_value=([bus], [], {}, [1], "x")),
                    patch.object(IMPORTER, "verify_native", side_effect=native_changes_directory),
                    redirect_stdout(io.StringIO()),
                ):
                    IMPORTER.main()
                report = json.loads((base / "case" / "verification.json").read_text(encoding="utf-8"))
                self.assertEqual(report["native_check"]["buses"], 69)
            finally:
                os.chdir(caller)


if __name__ == "__main__":
    unittest.main()
