import unittest

from envs.lfc.clock import AlignedEpisodeGate, CouplingClock
from envs.lfc.lfc_config import LFCConfig


class ClockTest(unittest.TestCase):
    def test_only_explicit_booleans_enable_clock(self):
        for malformed in ("false", "true", 0, 1, None, []):
            with self.subTest(aligned_clock=malformed):
                with self.assertRaisesRegex(TypeError, "aligned_clock must be a boolean"):
                    LFCConfig(aligned_clock=malformed)
        self.assertIs(LFCConfig(aligned_clock=True).aligned_clock, True)

    def test_opt_in_and_target_order(self):
        self.assertFalse(LFCConfig().aligned_clock)
        for h in (1, 6, 30):
            clock = CouplingClock(h)
            self.assertEqual(
                [clock.target()] + [clock.target(i) for i in range(h)],
                list(range(1, h + 2)),
            )
            self.assertGreater(clock.stop_time, clock.target(h - 1))
            with self.assertRaises(ValueError):
                clock.target(h)

    def test_actual_and_logged_time_must_advance(self):
        c = CouplingClock(6)
        self.assertIsNone(c.record(1, 1.033, 1.017)["dt_s"])
        for pause, sample in (
            (1.050, 1.033), (2.033, 1.017), (2.033, 2.050),
            (2.033, float("nan")), (8.033, 8.0), (2.051, 2.0),
        ):
            with self.assertRaises(RuntimeError):
                c.record(2, pause, sample)
        measured = c.record(2, 2.033, 2.017)
        self.assertAlmostEqual(measured["dt_s"], 1.0)
        self.assertEqual(measured["clock_sample_s"], 2.017)
        with self.assertRaises(RuntimeError):
            c.record(2, 2.033, 2.017)

    def test_reset_clears_previous_episode(self):
        c = CouplingClock(1)
        c.record(1, 1.033, 1.017)
        c.record(2, 2.033, 2.017)
        c.reset()
        self.assertIsNone(c.record(1, 1.033, 1.017)["dt_s"])


    def test_monotone_but_stale_trace_and_skipped_target(self):
        c = CouplingClock(6)
        c.record(1, 1.033, 1.0)
        with self.assertRaises(RuntimeError):
            c.record(2, 2.033, 1.79)
        with self.assertRaises(RuntimeError):
            c.record(3, 3.033, 3.0)
        self.assertEqual(c.record(2, 2.033, 1.8)["clock_sample_s"], 1.8)

    def test_episode_requires_good_reset_before_attempt_and_after_terminal(self):
        g = AlignedEpisodeGate()
        with self.assertRaises(RuntimeError):
            g.step(lambda: self.fail("unreset callback ran"))
        self.assertEqual(g.reset(lambda: "state"), "state")
        self.assertFalse(g.step(lambda: (None, 0, False))[2])
        self.assertTrue(g.step(lambda: (None, 0, True))[2])
        with self.assertRaises(RuntimeError):
            g.step(lambda: self.fail("terminal callback ran"))

    def test_mutating_step_failure_is_not_retryable(self):
        for phase in ("clock", "observation", "stop"):
            g = AlignedEpisodeGate()
            g.reset(lambda: None)
            called = []
            def mutate_and_fail():
                called.append(phase)
                raise ValueError(phase)
            with self.assertRaisesRegex(ValueError, phase):
                g.step(mutate_and_fail)
            with self.assertRaises(RuntimeError):
                g.step(mutate_and_fail)
            with self.assertRaises(ValueError):
                g.reset(mutate_and_fail)
            with self.assertRaises(RuntimeError):
                g.step(mutate_and_fail)
            self.assertEqual(called, [phase, phase])
            g.reset(lambda: None)
            self.assertFalse(g.step(lambda: (None, 0, False))[2])


if __name__ == "__main__":
    unittest.main()
