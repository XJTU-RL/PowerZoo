"""独立小电路的原生物理验收，不使用训练reward作为电气正确性证据。"""

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from envs.powergrid.session import DSSSession


@pytest.fixture
def model(tmp_path):
    file = tmp_path / "model.dss"
    file.write_text(
        """New Circuit.test basekv=12.47 pu=1 phases=3 bus1=source
New Line.l bus1=source bus2=bus phases=3 r1=0.1 x1=0.1 r0=0.2 x0=0.2 length=1 units=km
New Load.load bus1=bus phases=3 conn=wye kv=12.47 kw=100 kvar=50 model=1
New Capacitor.cap bus1=bus phases=3 kv=12.47 kvar=10 states=[1]
New XYCurve.eff npts=2 xarray=[0 1] yarray=[0.9 0.9]
New PVSystem.pv bus1=bus phases=3 conn=wye kv=12.47 kva=100 pmpp=100 irradiance=0.4 effcurve=eff %cutin=0 %cutout=0 kvarmax=80 kvarmaxabs=60 WattPriority=yes PFPriority=no
New Storage.bat bus1=bus phases=3 conn=wye kv=12.47 kwrated=100 kwhrated=200 %stored=50 %reserve=10 %effcharge=95 %effdischarge=95 %idlingkw=0 dispmode=external state=idling
New Generator.svg bus1=bus phases=3 conn=wye kv=12.47 kva=100 kw=0 kvar=0 model=1 vminpu=0.01 vmaxpu=2 maxkvar=100 minkvar=-100 status=fixed
Set VoltageBases=[12.47]
""",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("seconds", [60, 300, 900])
@pytest.mark.parametrize("power", [20.0, -20.0])
def test_native_storage_energy_once_and_reset(model, seconds, power):
    original = (model / "model.dss").read_bytes()
    session = DSSSession(model, "model.dss", step_seconds=seconds)
    try:
        session.begin_interval(0)
        session.apply({"Storage.bat": {"kw": power}})
        for _ in range(3):
            session.available_pv_power()
        assert float(session.read_property("Storage.bat", "%stored")) == 50
        assert session.time_s == 0
        measured = -session.powers("Storage.bat")[0]
        session.finish_interval()
        dc = measured / 0.95 if measured > 0 else measured * 0.95
        assert session.batteries["bat"].soc == pytest.approx(
            0.5 - dc * seconds / 3600 / 200, abs=1e-8
        )
        assert session.time_s == pytest.approx(seconds)
        with pytest.raises(RuntimeError):
            session.finish_interval()
        session.reset()
        assert session.batteries["bat"].soc == 0.5 and session.time_s == 0
    finally:
        session.close()
    assert (model / "model.dss").read_bytes() == original


def test_idle_grid_loss_does_not_charge_storage(model):
    file = model / "model.dss"
    file.write_text(file.read_text().replace("%idlingkw=0", "%idlingkw=1"))
    session = DSSSession(model, "model.dss")
    try:
        session.begin_interval(0)
        session.solve()
        assert session.powers("Storage.bat")[0] == pytest.approx(1.0, rel=1e-5)
        session.finish_interval()
        assert session.batteries["bat"].soc == 0.5
    finally:
        session.close()


def test_pv_available_ac_power_and_native_limits(model):
    session = DSSSession(model, "model.dss")
    try:
        available = session.available_pv_power()["pv"]
        assert available == pytest.approx(36, rel=1e-5)
        session.begin_interval(0)
        trace = session.apply({"PVSystem.pv": {"kw": available / 2, "kvar": 1000.0}})
        session.solve()
        session.finish_interval()
        p, q = -session.powers("PVSystem.pv")
        assert p == pytest.approx(18, rel=1e-4)
        assert 0 < q <= 80.01 and np.hypot(p, q) <= 100.01
        assert trace["PVSystem.pv"]["requested"]["kvar"] == 1000
        assert trace["PVSystem.pv"]["applied"]["kvar"] == 80
        assert float(session.read_property("PVSystem.pv", "Pmpp")) == 100
        assert abs(session.read_state().active_balance_error_kw) < 0.02
    finally:
        session.close()


@pytest.mark.parametrize("pu", [1.0, 0.7, 0.4])
@pytest.mark.parametrize("q", [-200.0, 200.0])
def test_svg_direction_capability_and_phase_current(model, pu, q):
    file = model / "model.dss"
    file.write_text(file.read_text().replace("pu=1 phases=3", f"pu={pu} phases=3"))
    session = DSSSession(model, "model.dss", svg_elements=["Generator.svg"])
    try:
        session.begin_interval(0)
        trace = session.apply({"Generator.svg": {"kvar": q}})
        session.solve()
        session.finish_interval()
        p_actual, q_actual = -session.powers("Generator.svg")
        assert abs(p_actual) < 0.01 and q_actual * q > 0
        assert abs(q_actual) <= 100.01
        session.require_element("Generator.svg")
        phase_current = session.native.ActiveCktElement.CurrentsMagAng[::2][:3]
        assert max(phase_current) <= session.svgs["svg"].rated_current_a * (1 + 2e-6)
        assert trace["Generator.svg"]["measured"]["kvar"] == pytest.approx(q_actual)
        if pu < 1:
            assert trace["Generator.svg"]["limit_reason"] == "phase_current"
    finally:
        session.close()


def test_invalid_frame_atomic_and_measurement_errors_are_not_masked(model):
    session = DSSSession(model, "model.dss")
    try:
        session.solve()
        before = float(session.read_property("Load.load", "kW"))
        with pytest.raises(ValueError):
            session.apply(
                {"Load.load": {"kw": 12}, "PVSystem.missing": {"kw": 0, "kvar": 0}}
            )
        assert float(session.read_property("Load.load", "kW")) == before
        with pytest.raises(ValueError):
            session.powers("Load.missing")
        with pytest.raises(ValueError):
            session.bus_voltage("missing")
        session.apply({"Load.load": {"kw": 12}})
        with pytest.raises(RuntimeError):
            session.read_state()
    finally:
        session.close()


def test_solver_failure_never_advances_energy_or_clock(model):
    session = DSSSession(model, "model.dss")
    try:
        native = session.native
        session.begin_interval(0)
        session.apply({"Storage.bat": {"kw": 20}})

        def fail():
            raise RuntimeError("injected native failure")

        session.native = SimpleNamespace(
            Solution=SimpleNamespace(SolveSnap=fail, dblHour=0)
        )
        with pytest.raises(RuntimeError, match="injected"):
            session.solve()
        with pytest.raises(RuntimeError):
            session.finish_interval()
        assert native.Solution.dblHour == 0
        session.native = native
        assert float(session.read_property("Storage.bat", "%stored")) == 50
        session.reset()
        session.solve()
    finally:
        session.close()


def test_paths_contexts_and_model_hash(model):
    a = DSSSession(model, "model.dss")
    b = DSSSession(model, "model.dss")
    try:
        assert a.model_sha256 == b.model_sha256
        a.apply({"Load.load": {"kw": 12}})
        assert float(b.read_property("Load.load", "kW")) == 100
        assert (
            a.manifest()["input_files"]["model.dss"]
            == sha256((model / "model.dss").read_bytes()).hexdigest()
        )
        with pytest.raises(ValueError):
            DSSSession(model, "../other.dss")
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize(
    "soc,kw", [(0.1001, 100), (0.9999, -100), (0.1, 100), (1, -100)]
)
@pytest.mark.parametrize("idle", [0, 1])
def test_native_storage_bounds_include_idle_loss(model, soc, kw, idle):
    file = model / "model.dss"
    file.write_text(
        file.read_text()
        .replace("%stored=50", f"%stored={soc*100}")
        .replace("%idlingkw=0", f"%idlingkw={idle}")
    )
    session = DSSSession(model, "model.dss", step_seconds=900)
    try:
        session.begin_interval(0)
        session.apply({"Storage.bat": {"kw": kw}})
        session.solve()
        session.finish_interval()
        assert 0.1 - 1e-9 <= session.batteries["bat"].soc <= 1 + 1e-9
    finally:
        session.close()


def test_native_finish_failure_invalidates_measurements(model):
    session = DSSSession(model, "model.dss")
    try:
        session.begin_interval(0)
        session.solve()

        def fail():
            raise RuntimeError("native finish failure")

        session.native = SimpleNamespace(
            Solution=SimpleNamespace(dblHour=0, FinishTimeStep=fail)
        )
        with pytest.raises(RuntimeError, match="native finish"):
            session.finish_interval()
        with pytest.raises(RuntimeError):
            session.read_state()
        with pytest.raises(RuntimeError):
            session.begin_interval(0)
    finally:
        session.close()
