import unittest

from envs.lfc.droop_123_draft import (
    Arm, Frequency, Hold, assess, demand, droop, load_protocol, read_causal,
)
from envs.lfc.native_feeder_123 import PQ, StorageReading


class TestDraft(unittest.TestCase):
    cfg = load_protocol()
    ids = tuple(f"Storage.batt{i}" for i in range(1, 5))

    def fixture(self, mode, *, flat=False, broken=False, spoof=False):
        c = self.cfg["clock"]
        hz = tuple(Frequency(i / 5, 60 if flat or mode == 0 or i <= 125 else 59.999)
                   for i in range(100, 226))
        anchor, source = PQ(7852, 12943), PQ(3615, 1312)
        rows, soc = [], 0.5
        for t in range(20, 45):
            pause = t + 1 / 30
            step = 100 if mode and 25 <= t < 35 else 0
            sample = read_causal(lambda: (s for s in hz if s.time_s <= pause), pause - 1, pause, c) if mode == 2 and t >= 25 else None
            request = droop(60, sample, self.cfg["controller"]) if sample else (0,) * 4
            p = request[0]
            after = soc - p / (3600 * 200) / .95
            terms = tuple(StorageReading(i, p, p, PQ(p, 0), soc, after) for i in self.ids)
            soc = after
            pcc = PQ(source.kw - 4 * p, source.kvar)
            command = demand(anchor, source, pcc, step)
            bus8 = PQ(command.kw + (4 * p if broken else 0), command.kvar)
            if sample is not None and spoof:
                sample = Frequency(sample.time_s - .2, sample.hz)
            rows.append(Hold(t, pause, pause + 1, step, request, terms, pcc, bus8, command, 0, sample))
        return Arm(tuple(rows), hz, anchor, source)

    def test_native_clock_guard_and_fixed_droop(self):
        c = self.cfg["clock"]
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            read_causal(None, 20, 21, c)
        with self.assertRaisesRegex(RuntimeError, "fresh"):
            read_causal(lambda: (Frequency(20.2, 60), Frequency(21, 59.98)), 20, 21, c)
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            read_causal(lambda: (Frequency(20.9, 60), Frequency(20.9, 60)), 20, 21, c)
        with self.assertRaisesRegex(RuntimeError, "future"):
            read_causal(lambda: (Frequency(20.9, 60), Frequency(21.01, 59.98)), 20, 21, c)
        self.assertEqual(
            read_causal(lambda: (Frequency(20.9, 60), Frequency(21, 59.9)), 20, 21, c).hz, 60
        )
        self.assertEqual(droop(60, Frequency(20.9, 58), self.cfg["controller"]), (50,) * 4)
        self.assertEqual(droop(60, Frequency(20.9, 62), self.cfg["controller"]), (-50,) * 4)

    def test_pair_reports_transport_no_response_or_inconclusive(self):
        for options, expected in (
            ({}, "transport_observed"),
            ({"flat": True}, "no_detectable_controller_response"),
            ({"broken": True}, "inconclusive"),
            ({"spoof": True}, "inconclusive"),
        ):
            with self.subTest(options=options):
                arms = dict(zip(self.cfg["arms"], (
                    self.fixture(0), self.fixture(1), self.fixture(2, **options)
                )))
                result = assess(arms, self.ids, self.cfg)
                self.assertEqual(result["status"], expected)
                if not options:
                    self.assertEqual(result["reference_hz"], 60)
                    self.assertTrue(any(row["supported"] for row in result["evidence"]))
