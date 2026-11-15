"""独立小电路的原生物理验收，不使用训练reward作为电气正确性证据。"""

from hashlib import sha256
import os
from pathlib import Path
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


def test_temporary_model_root_is_canonical_before_containment_check(model, monkeypatch):
    """临时目录别名解析后仍可加载模型，同时保持越界输入拒绝。"""
    from envs.powergrid import session as session_module

    temporary_directory = session_module.tempfile.TemporaryDirectory

    def aliased_temporary_directory(*args, **kwargs):
        """以相对路径模拟 Windows 短路径解析后的目录别名。"""
        kwargs["dir"] = Path.cwd()
        temporary = temporary_directory(*args, **kwargs)
        return SimpleNamespace(
            name=os.path.relpath(temporary.name, Path.cwd()),
            cleanup=temporary.cleanup,
        )

    monkeypatch.setattr(
        session_module.tempfile, "TemporaryDirectory", aliased_temporary_directory
    )
    session = DSSSession(model, "model.dss")
    try:
        session.solve()
        assert session.folder == session.folder.resolve()
        assert session.native.Solution.Converged
        with pytest.raises(ValueError, match="模型目录内"):
            session._input_path("../outside.dss")
    finally:
        session.close()


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


def test_incomplete_native_controls_cannot_finish_or_publish_state(model):
    session = DSSSession(model, "model.dss")
    try:
        session.begin_interval(0)
        session.native = SimpleNamespace(
            Solution=SimpleNamespace(
                dblHour=0.0,
                Converged=True,
                ControlActionsDone=False,
                SolveSnap=lambda: None,
            )
        )
        with pytest.raises(RuntimeError, match="控制循环"):
            session.solve(native_controls=True)
        with pytest.raises(RuntimeError):
            session.finish_interval()
        with pytest.raises(RuntimeError):
            session.read_state()
        assert session.time_s == 0
    finally:
        session.close()


def _pv_variant(model, replacement):
    """在 fixture 模型的 PVSystem 行上做文本替换，返回写入后的文件。"""
    file = model / "model.dss"
    text = file.read_text(encoding="utf-8")
    old = "kvarmax=80 kvarmaxabs=60"
    assert old in text, "fixture 模型 PV 行缺少替换锚点"
    file.write_text(text.replace(old, replacement), encoding="utf-8")
    return file


def test_probe_restores_model_native_constant_q(model):
    """模型原生 kvar=10 的 PV：探测后降辐照，恒无功必须保持 10。

    复现 4d8af3b5 复审缺陷 2（P1）：探测恢复只认 apply 显式写过的 kvar，
    模型原生 constant-Q 设备被当 fixed-pf 恢复（pf=1），Q 被钉成 0。
    """
    _pv_variant(model, "kvar=10 kvarmax=80 kvarmaxabs=60")
    session = DSSSession(model, "model.dss")
    try:
        session.solve()
        p, q = -session.powers("PVSystem.pv")
        assert p == pytest.approx(36, rel=1e-4)  # 100kW × 0.4 × eff 0.9
        assert q == pytest.approx(10, rel=1e-4)

        session.available_pv_power()  # 全量探测（写 %Pmpp=100 kvar=0 后恢复）

        session.set_pv_conditions(0.2, 25)
        session.solve()
        p, q = -session.powers("PVSystem.pv")
        assert p == pytest.approx(18, rel=1e-4)
        assert q == pytest.approx(10, rel=1e-4), (
            "探测后恒无功模式丢失：constant-Q 被恢复成 pf=1（Q 钉 0）"
        )
    finally:
        session.close()


@pytest.mark.parametrize("settings,constant_q", [
    ("pf=.95 kvar=10", True), ("kvar=10 pf=-.95", False),
])
def test_probe_uses_effective_native_driver_after_ordered_edits(model, settings, constant_q):
    """PF/kvar getter 会残留旧值，恢复须遵循引擎最后生效的属性。"""
    _pv_variant(model, f"{settings} kvarmax=80 kvarmaxabs=60")
    session = DSSSession(model, "model.dss")
    try:
        for runtime in (False, True):
            if runtime:
                session.command(f"Edit PVSystem.pv {settings}")
            session.set_pv_conditions(0.4, 25)
            session.solve()
            session.available_pv_power(["pv"])
            session.set_pv_conditions(0.2, 25)
            session.solve()
            p, q = -session.powers("PVSystem.pv")
            expected = 10 if constant_q else p * np.tan(np.arccos(-0.95))
            assert q == pytest.approx(expected, rel=1e-4)
    finally:
        session.close()


def test_probe_keeps_model_fixed_pf_tracking(model):
    """模型原生 pf=-0.95 的 PV：探测后 Q 仍随辐照按 pf 派生（回归保护）。"""
    _pv_variant(model, "pf=-0.95 kvarmax=80 kvarmaxabs=60")
    session = DSSSession(model, "model.dss")
    try:
        session.solve()
        p, q = -session.powers("PVSystem.pv")
        assert q == pytest.approx(p * np.tan(np.arccos(-0.95)), rel=1e-3)

        session.available_pv_power()

        session.set_pv_conditions(0.2, 25)
        session.solve()
        p_low, q_low = -session.powers("PVSystem.pv")
        assert p_low == pytest.approx(18, rel=1e-4)
        assert q_low == pytest.approx(
            p_low * np.tan(np.arccos(-0.95)), rel=1e-3
        ), "探测恢复把 fixed-pf 模式错还原成恒无功"
    finally:
        session.close()


def test_probe_after_explicit_apply_restores_explicit_kvar(model):
    """RL 经 apply 显式接管 Q 后，探测恢复显式 kvar（既有行为回归）。"""
    session = DSSSession(model, "model.dss")
    try:
        session.begin_interval(0)
        session.apply({"PVSystem.pv": {"kw": 18, "kvar": 5}})
        session.available_pv_power()
        session.solve()
        _, q = -session.powers("PVSystem.pv")
        assert q == pytest.approx(5, rel=1e-3)
        session.finish_interval()
    finally:
        session.close()


def test_probe_after_apply_reset_returns_to_native_mode(model):
    """apply 接管后 reset：驱动注册随模型重编译回到原生定义。"""
    _pv_variant(model, "kvar=10 kvarmax=80 kvarmaxabs=60")
    session = DSSSession(model, "model.dss")
    try:
        session.begin_interval(0)
        session.apply({"PVSystem.pv": {"kw": 18, "kvar": 5}})
        session.available_pv_power()
        session.reset()
        session.solve()
        _, q = -session.powers("PVSystem.pv")
        assert q == pytest.approx(10, rel=1e-4), "reset 后原生恒无功 10 未恢复"
    finally:
        session.close()


def test_subset_probe_leaves_unlisted_native_pv_untouched(model):
    """子集探测只碰接管名单：未列出的原生 fixed-pf PV 驱动不被改写。"""
    file = model / "model.dss"
    text = file.read_text(encoding="utf-8")
    extra = (
        "New PVSystem.pv2 bus1=bus phases=3 conn=wye kv=12.47 kva=100 pmpp=100 "
        "irradiance=0.4 effcurve=eff %cutin=0 %cutout=0 pf=-0.95 "
        "kvarmax=80 kvarmaxabs=60 WattPriority=yes PFPriority=no\n"
    )
    file.write_text(
        text.replace("New Storage.bat", extra + "New Storage.bat"), encoding="utf-8"
    )
    session = DSSSession(model, "model.dss")
    try:
        session.available_pv_power(names=["pv"])

        session.set_pv_conditions(0.2, 25)
        session.solve()
        p2, q2 = -session.powers("PVSystem.pv2")
        assert p2 == pytest.approx(18, rel=1e-4)
        assert q2 == pytest.approx(
            p2 * np.tan(np.arccos(-0.95)), rel=1e-3
        ), "子集探测改写了未列名 PV 的无功驱动模式"
    finally:
        session.close()
