"""Engine-free tests of static intake arithmetic and phase mapping."""
import math
import unittest
from types import SimpleNamespace

from scripts.check_opendss_case_intake import independent_case69, readings


class FakeCircuit:
    def __init__(self):
        # DSS may report noncanonical phase order, so key by node labels.
        self.ActiveBus = SimpleNamespace(
            Nodes=[3, 1, 2], VMagAngle=[120, 120, 100, 0, 110, -120]
        )

    def SetActiveBus(self, name):
        return 0 if name == "bus" else -1


class IntakeArithmeticTests(unittest.TestCase):
    def test_phases_and_line_to_line(self):
        circuit = FakeCircuit()
        for phase, magnitude in readings(circuit, {"bus": "LN"})["bus"].items():
            self.assertAlmostEqual(magnitude, dict(A=100, B=110, C=120)[phase])
        voltage = readings(circuit, {"bus": "LL"})["bus"]
        self.assertAlmostEqual(
            voltage["AB"], math.sqrt(100**2 + 110**2 + 100 * 110)
        )
        self.assertAlmostEqual(
            voltage["BC"], math.sqrt(110**2 + 120**2 + 110 * 120)
        )
        self.assertAlmostEqual(
            voltage["CA"], math.sqrt(120**2 + 100**2 + 120 * 100)
        )

    def test_missing_bus_and_phase(self):
        circuit = FakeCircuit()
        with self.assertRaises(ValueError):
            readings(circuit, {"missing": "LL"})
        circuit.ActiveBus.Nodes = [3, 1, 0]
        with self.assertRaises(ValueError):
            readings(circuit, {"bus": "LL"})

    def test_shunt_can_raise_voltage_and_lower_q_loss(self):
        # Two-bus, 12.66 kV, ideal slack, 0+j300 kvar load.
        buses = [
            [1, 3, 0, 0, 0, 0, 0, 0, 0, 12.66],
            [2, 1, 0, 300, 0, 0, 0, 0, 0, 12.66],
        ]
        children = {2: (1, 0.5 + 1j)}
        baseline, loss = independent_case69(buses, children, [1, 2], 1, None)
        with_cap, loss_with_cap = independent_case69(buses, children, [1, 2], 1, 2)
        self.assertGreater(with_cap["b2"], baseline["b2"])
        self.assertLess(loss_with_cap, loss)
