import cmath
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from envs.lfc.damped_droop_123 import load_protocol
from envs.lfc.droop_123_draft import load_protocol as fixed_protocol
from envs.lfc.native_123_availability import (
    Conductor, OperatingPoint, _device_voltages, _sample_element, envelope,
    prospective, verify_device_envelope,
)
from envs.lfc.native_feeder_123 import PQ, stock_probe_spec


class TestAvailability(unittest.TestCase):
    spec = envelope()
    devices = stock_probe_spec()[0]["storage_devices"]
    ids = tuple(d["id"] for d in devices)

    def point(self, kw, *, at=10, soc=.5, state=("locked",)):
        readings = []
        for d in self.devices:
            wye = d["connection"] == "wye"
            if wye:
                pairs = [(0, 0j), (1, 2400 + 0j)]
            else:
                pairs = [(i + 1, cmath.rect(4160 / 3**.5, -2 * cmath.pi * i / 3))
                         for i in range(3)]
            readings.extend(Conductor(d["id"], 1, n, v, kw / 10, False) for n, v in pairs)
        return OperatingPoint(at, state, (soc,) * 4, (PQ(kw, 0),) * 4,
                              PQ(3600 - kw * 4, 1300), tuple(readings))

    def test_correct_delta_ll_and_wye_ln_and_unsafe_current(self):
        pt = self.point(50)
        for device in self.devices:
            self.assertTrue(all(abs(v - 1) < 1e-9 for v in _device_voltages(pt, device)))
        req = dict.fromkeys(self.ids, 50)
        verify_device_envelope(pt, req, self.spec, load_protocol(), fixed_protocol())
        for changes in (dict(amps=15), dict(volts=0j), dict(opened=True)):
            rows = list(pt.conductors)
            rows[-1] = replace(rows[-1], **changes)
            with self.assertRaises(RuntimeError):
                verify_device_envelope(replace(pt, conductors=tuple(rows)), req,
                                       self.spec, load_protocol(), fixed_protocol())
        with self.assertRaisesRegex(RuntimeError, "reserve"):
            verify_device_envelope(replace(pt, soc=(.1,) * 4), req, self.spec,
                                   load_protocol(), fixed_protocol())

    def test_native_element_terminals_node_identity(self):
        c = dict(enabled=True, nodes=[1, 3, 1, 3], conductors=2, terminals=2,
                 open=[[False, False], [True, False]])
        elem = SimpleNamespace(NodeOrder=c["nodes"], VoltagesMagAng=[100, 0] * 4,
                               CurrentsMagAng=[1, 0, 2, 0, 3, 0, 4, 0])
        session = SimpleNamespace(connection=lambda _: c,
                                  native=SimpleNamespace(ActiveCktElement=elem))
        samples = _sample_element(session, "Line.Sample", (2,))
        self.assertEqual([(x.terminal, x.node, x.amps, x.opened) for x in samples],
                         [(2, 1, 3, True), (2, 3, 4, False)])
        elem.NodeOrder = [1, 3, 3, 1]
        with self.assertRaisesRegex(RuntimeError, "NodeOrder"):
            _sample_element(session, "Line.Sample", (2,))

    def test_complete_vector_in_isolated_context_never_advances_clock_or_soc(self):
        s, live_s = Mock(), Mock()
        live_s.source = s.source = Path("/same/deck")
        live_s.folder, s.folder = Path("/live"), Path("/scratch")
        s.batteries = {name.split(".")[1]: SimpleNamespace(soc=.5) for name in self.ids}
        s.read_property.return_value = "50"
        live = SimpleNamespace(session=live_s, storage_ids=self.ids,
                               calibration=SimpleNamespace(target_t0=PQ(1, 1)))
        scratch = SimpleNamespace(session=s, close=Mock())
        zero, next_ = dict.fromkeys(self.ids, 0), dict.fromkeys(self.ids, 36)
        values = [self.point(0), self.point(0, at=0), self.point(0),
                  self.point(36), self.point(0)]
        with patch("envs.lfc.native_123_availability.collect", side_effect=values):
            p = prospective(live, zero, next_, self.spec,
                            _factory=lambda _: scratch)
        self.assertEqual(p, values[3])
        self.assertEqual(s.apply.call_args.args[0], {d: {"kw": 36} for d in self.ids})
        self.assertEqual(s.solve.call_count, 2)
        s.finish_interval.assert_not_called()
        s.begin_interval.assert_not_called()
        self.assertEqual(live_s.mock_calls, [])
        scratch.close.assert_called_once()
        s.reset_mock()
        values[3] = self.point(36, soc=.49)
        with patch("envs.lfc.native_123_availability.collect", side_effect=values):
            with self.assertRaisesRegex(RuntimeError, "changed clock, native SOC"):
                prospective(live, zero, next_, self.spec,
                            _factory=lambda _: scratch)

    def test_failed_scratch_parity_prevents_applying_candidate(self):
        live_s, s = Mock(), Mock()
        live_s.source = s.source = Path("/same/deck")
        live_s.folder, s.folder = Path("/live"), Path("/scratch")
        s.batteries = {name.split(".")[1]: SimpleNamespace(soc=.5) for name in self.ids}
        s.read_property.return_value = "50"
        live = SimpleNamespace(session=live_s, storage_ids=self.ids,
                               calibration=SimpleNamespace(target_t0=PQ(1, 1)))
        scratch = SimpleNamespace(session=s, close=Mock())
        with patch("envs.lfc.native_123_availability.collect",
                   side_effect=[self.point(0), self.point(0, at=0), self.point(1), self.point(0)]):
            with self.assertRaisesRegex(RuntimeError, "baseline differs"):
                prospective(live, dict.fromkeys(self.ids, 0), dict.fromkeys(self.ids, 45),
                            self.spec, _factory=lambda _: scratch)
        s.apply.assert_called_once()
        s.finish_interval.assert_not_called()

    def test_no_alias_object_context_or_path_is_owned_or_closed(self):
        root = Path("/tmp/live-dss")
        ctx = object()
        live_s = SimpleNamespace(folder=root, dss=ctx)
        live = SimpleNamespace(session=live_s, close=Mock(), storage_ids=self.ids,
                               calibration=SimpleNamespace(target_t0=PQ(1, 1)))
        aliases = (
            live, SimpleNamespace(session=live_s, close=Mock()),
            SimpleNamespace(session=SimpleNamespace(folder=Path("/other"), dss=ctx),
                            close=Mock()),
            SimpleNamespace(session=SimpleNamespace(folder=root / ".." / root.name, dss=object()),
                            close=Mock()),
        )
        zero = dict.fromkeys(self.ids, 0)
        for alias in aliases:
            with patch("envs.lfc.native_123_availability.collect", return_value=self.point(0)) as col:
                with self.assertRaisesRegex(RuntimeError, "independent DSS"):
                    prospective(live, zero, zero, self.spec, _factory=lambda _: alias)
            alias.close.assert_not_called()
            live.close.assert_not_called()
            self.assertEqual([call.args[0] for call in col.call_args_list], [live, live])

    def test_live_mutation_after_scratch_rejection_preserves_original_failure(self):
        live = SimpleNamespace(storage_ids=self.ids,
                               calibration=SimpleNamespace(target_t0=PQ(1, 1)))
        zero = dict.fromkeys(self.ids, 0)
        problem = RuntimeError("candidate already rejected")
        with patch("envs.lfc.native_123_availability.collect",
                   side_effect=[self.point(0), self.point(0, soc=.45)]):
            with self.assertRaisesRegex(RuntimeError, "LIVE feeder parity") as raised:
                prospective(live, zero, zero, self.spec, _factory=Mock(side_effect=problem))
        self.assertIs(raised.exception.__cause__, problem)
