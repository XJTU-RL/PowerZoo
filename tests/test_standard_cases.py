"""Metadata checks use actual bundled assets and no DSS runtime."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_standard_cases import ROOT, REGISTRY, _check_dss_redirects, _unique_pairs, _verify_assets, validate


class TestStandardCaseRegistry(unittest.TestCase):
    def load(self):
        return json.loads((ROOT / REGISTRY).read_bytes())

    def test_actual_assets_and_reuse_existing_vvc_config(self):
        data = validate()
        self.assertEqual(set(data), {"ieee34", "ieee123", "ieee4", "ieee37", "baran-wu69"})
        self.assertEqual(data["ieee34"]["tasks"]["vvc"]["system_ref"], "34Bus")
        self.assertEqual(data["ieee123"]["tasks"]["vvc"]["system_ref"], "123Bus")
        for case in ("ieee4", "ieee37", "baran-wu69"):
            self.assertEqual(data[case]["tasks"]["vvc"]["status"], "unverified")

    def test_external_static_evidence_is_not_vvc_or_canonical_parity(self):
        data = validate()
        self.assertEqual(data["ieee4"]["static_evidence"]["level"],
                         "powerzoo_static_and_magnitude_spots_validated")
        self.assertEqual(data["ieee37"]["static_evidence"]["level"],
                         "powerzoo_stock_static_regression_validated")
        self.assertEqual(data["baran-wu69"]["static_evidence"]["level"],
                         "independent_balanced_static_validated")
        for key in ("ieee4", "ieee37", "baran-wu69"):
            self.assertEqual(data[key]["static_evidence"]["local_validation_revision"],
                             "98fe198da405b2c0ce618c0812b8b2179a475fe2")
            self.assertEqual(data[key]["tasks"]["vvc"]["status"], "unverified")

    def test_pinned_path_and_readiness_fail_closed(self):
        for field, value, error in (
            ("entrypoint", "node_systems/../escape", "asset path"),
            ("entrypoint", "node_systems/34Bus_PV/network.dss", "not pinned"),
        ):
            doc = self.load()
            doc["cases"][0][field] = value
            with self.assertRaisesRegex(ValueError, error):
                validate(doc)
        doc = self.load()
        doc["cases"][2]["tasks"]["vvc"]["status"] = "smoke_validated"
        with self.assertRaisesRegex(ValueError, "cannot advertise"):
            validate(doc)
        doc = self.load()
        doc["cases"][0]["assets"][0]["digest"] = "0"*64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate(doc)

    def test_external_local_payload_bytes_are_optional_but_strict(self):
        data = b"synthetic test only"
        hashes = [
            ("sha256", hashlib.sha256(data).hexdigest()),
            ("git_blob_sha1", hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "Exact.dss"
            p.write_bytes(data)
            for kind, digest in hashes:
                asset = {"path": p.name, "hash_kind": kind, "digest": digest}
                _verify_assets([asset], root)
                with self.assertRaisesRegex(ValueError, "differently cased"):
                    _verify_assets([dict(asset, path="exact.dss")], root)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    _verify_assets([dict(asset, digest="0"*len(digest))], root)
            alias = root / "alias.dss"
            alias.symlink_to(p)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _verify_assets([dict(asset, path=alias.name)], root)

    def test_duplicate_metadata_cannot_be_silently_overridden(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            json.loads('{"cases":[],"cases":[]}', object_pairs_hook=_unique_pairs)

    def test_bundled_redirect_closure_rejects_unpinned_or_unused_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master, dependency = root / "master.dss", root / "load.dss"
            master.write_text("Redirect load.dss")
            dependency.write_text("! no references")
            case = {"entrypoint": master.name, "assets": [{"path": master.name}]}
            with self.assertRaisesRegex(ValueError, "unregistered DSS redirect"):
                _check_dss_redirects(case, root)
            case["assets"].append({"path": dependency.name})
            _check_dss_redirects(case, root)
            master.write_text("! no references")
            with self.assertRaisesRegex(ValueError, "active DSS redirect closure"):
                _check_dss_redirects(case, root)
