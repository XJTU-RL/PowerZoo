import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_lfc_123_feedback_engineering import PROTOCOL, intended, open_gate


class EngineeringPolicy(unittest.TestCase):
    def test_candidate_ramp_and_release(self):
        cfg = json.loads(PROTOCOL.read_text())
        self.assertEqual(intended(60., 59.99, 0., cfg), 30.)
        self.assertEqual(intended(60., 60.01, 0., cfg), 0.)
        self.assertEqual(intended(60., 60., 10., cfg), 6.)

    def test_partial_batch_cannot_start_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"partial.jsonl"
            path.write_text('{"phase":"paired"}\n')
            with self.assertRaisesRegex(ValueError, "incomplete"):
                open_gate(path)
