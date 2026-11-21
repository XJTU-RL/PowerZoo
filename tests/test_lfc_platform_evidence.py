import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_lfc_platform_evidence import check_loop
from scripts.check_lfc_platform_witness import verify


class EvidenceQualityTest(unittest.TestCase):
    def test_failed_archive_cannot_be_engineering_success(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"fault.jsonl"
            p.write_text(json.dumps({"phase":"failed"})+"\n")
            with self.assertRaisesRegex(ValueError,"complete successful native batch"):
                check_loop(p,"34","unused")
            with self.assertRaisesRegex(ValueError,"incomplete"):
                verify(p)

    def test_nonfinite_and_incomplete_witness_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"bad.jsonl"
            p.write_text('{"phase":"port","pcc":[NaN,0]}\n')
            with self.assertRaisesRegex(ValueError,"nonfinite"):
                verify(p)
            p.write_text('{"phase":"manifest"}\n')
            with self.assertRaisesRegex(ValueError,"incomplete"):
                verify(p)
