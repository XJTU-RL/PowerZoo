"""SMART-DS 接入测试：合成小模型单测（无网络、无大数据）+ 真实算例可选测试。

合成模型复刻 SMART-DS 结构：LoadShape ``mult/qmult=(file=..)`` pu 曲线、
中心抽头两负荷对象共享曲线、PV 辐照曲线（kW/m²）、kvarMax 非对称、原生
InvControl。真实算例测试标记 requires_external_case，未下载数据明确 skip。
"""

from pathlib import Path
import json
import zipfile

import numpy as np
import pytest

from utils.external_cases import (
    CASES,
    FILE_SET_CASES,
    ExternalCaseError,
    FilePatch,
    FileSetCase,
    CaseEntry,
    external_root,
    verify,
)
from envs.powergrid.case_inventory import dependency_closure

pytestmark = [pytest.mark.requires_opendss]

POINTS = 35040


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _profile(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, fmt="%.12g")


@pytest.fixture
def smartds_like_model(tmp_path):
    """SMART-DS 结构的微型模型：pu 负荷曲线 + 中心抽头 + PV + InvControl。"""
    root = tmp_path / "layout"
    load_pu = np.full(POINTS, 0.4)
    load_pu[100] = 1.0  # 年峰
    load_q = np.full(POINTS, 0.2)
    pv_curve = np.zeros(POINTS)
    pv_curve[48] = 0.8  # kW/m²
    _profile(root / "profiles" / "com_kw_1_pu.csv", load_pu)
    _profile(root / "profiles" / "com_kvar_1_pu.csv", load_q)
    _profile(root / "profiles" / "GSO_1.0_-1.0_25_180.csv", pv_curve)
    _write(
        root / "solar_data" / "GSO_1.0_-1.0_25_180_full.csv",
        "Year,Month,Day,Hour,Minute,DNI,DHI,Wind Speed,Temperature,GHI,"
        "PoA Irradiance (W/m^2),kW Generated (1000 kW Array)\n"
        + "\n".join(
            f"2017,1,1,{i // 4},{(i % 4) * 15},0.0,0.0,1.0,{10.0 + i * 0.001:.3f},"
            f"0.0,{pv_curve[i] * 1000:.1f},{pv_curve[i] * 800:.1f}"
            for i in range(POINTS)
        )
        + "\n",
    )
    _write(
        root / "feeder" / "LoadShapes.dss",
        "New Loadshape.com_kw_1_pu npts=35040 interval=0.25"
        " mult=(file=../profiles/com_kw_1_pu.csv)"
        " qmult=(file=../profiles/com_kvar_1_pu.csv)\n"
        "New Loadshape.gso_1.0_-1.0_25_180 npts=35040 interval=0.25"
        " mult=(file=../profiles/GSO_1.0_-1.0_25_180.csv)\n",
    )
    _write(
        root / "feeder" / "Loads.dss",
        # 中心抽头：两个负荷对象共享曲线、铭牌各半（官方口径）。
        "New Load.load_ct_1 phases=1 bus1=lv.1 kV=0.12 kW=1.5 kvar=0.3"
        " yearly=com_kw_1_pu\n"
        "New Load.load_ct_2 phases=1 bus1=lv.2 kV=0.12 kW=1.5 kvar=0.3"
        " yearly=com_kw_1_pu\n"
        "New Load.load_3ph phases=3 bus1=mv kV=4.0 kW=20 kvar=5"
        " yearly=com_kw_1_pu\n",
    )
    _write(
        root / "feeder" / "PVSystems.dss",
        "New PVSystem.pv1 bus1=lv.1.2 phases=1 kV=0.24 kVA=2.0 Pmpp=2.0"
        " kvarMax=1.1 kvarMaxAbs=0.6 irradiance=1"
        " yearly=GSO_1.0_-1.0_25_180\n"
        "New XYCurve.vv npts=6 Yarray=(1,1,0,0,-1,-1) Xarray=(.5,.92,.98,1.02,1.08,1.5)\n"
        "New InvControl.ic1 mode=VOLTVAR vvc_curve1=vv"
        " RefReactivePower=VARMAX_VARS PVSystemlist=[pv1]\n",
    )
    _write(
        root / "feeder" / "Master.dss",
        "\n".join(
            [
                "clear",
                "new circuit.t",
                "edit vsource.source basekV=4.0 pu=1.03",
                "new line.mv1 bus1=sourcebus bus2=mv phases=3 r1=0.05 x1=0.02",
                "new transformer.t1 phases=3 buses=[mv,lv] conns=[delta,wye]"
                " kVs=[4.0,0.24] kVA=100 XHL=4.0",
                "redirect LoadShapes.dss",
                "redirect Loads.dss",
                "redirect PVSystems.dss",
                "set voltagebases=[4.0,0.24]",
                "calcvoltagebases",
                "solve",
            ]
        )
        + "\n",
    )
    return root


# ---------------------------------------------------------------------------
# 依赖闭包：LoadShape file= 数据依赖
# ---------------------------------------------------------------------------


def test_closure_follows_loadshape_datafiles(smartds_like_model):
    closure = dependency_closure(smartds_like_model, "feeder/Master.dss")
    kinds = {name: info["kind"] for name, info in closure["files"].items()}
    assert kinds["profiles/com_kw_1_pu.csv"] == "datafile"
    assert kinds["feeder/Master.dss"] == "script"
    assert closure["missing"] == []


def test_closure_reports_missing_datafile_with_line(smartds_like_model):
    target = smartds_like_model / "profiles" / "com_kw_1_pu.csv"
    target.unlink()
    closure = dependency_closure(smartds_like_model, "feeder/Master.dss")
    assert any(
        item["kind"] == "datafile" and item["from"].startswith("feeder/LoadShapes.dss:")
        for item in closure["missing"]
    )


# ---------------------------------------------------------------------------
# SmartDSTimeseries：中心抽头、逐 PV 辐照、单位与时间轴
# ---------------------------------------------------------------------------


def _session_and_series(smartds_like_model):
    from envs.powergrid.session import DSSSession
    from envs.powergrid.smartds_timeseries import SmartDSTimeseries

    session = DSSSession(smartds_like_model, "feeder/Master.dss")
    series = SmartDSTimeseries(smartds_like_model, "feeder/Master.dss").bind(session)
    return session, series


def test_timeseries_center_tap_shares_curve_with_half_nameplate(smartds_like_model):
    session, series = _session_and_series(smartds_like_model)
    try:
        assert series.load_shapes["load_ct_1"] == series.load_shapes["load_ct_2"]
        # 铭牌各半：pu 曲线峰=1 时两侧各为总额定一半，且逐对象写入即无重复。
        kw1, _ = series.row(0)["loads"]["load_ct_1"]
        kw2, _ = series.row(0)["loads"]["load_ct_2"]
        assert kw1 == pytest.approx(1.5 * 0.4)
        assert kw2 == pytest.approx(kw1)
        peak_kw, _ = series.row(100)["loads"]["load_ct_1"]
        assert peak_kw == pytest.approx(1.5)  # 年峰=铭牌（pu 峰=1）
    finally:
        session.close()


def test_timeseries_pv_units_and_temperature(smartds_like_model):
    session, series = _session_and_series(smartds_like_model)
    try:
        irradiance, temperature = series.row(48)["pv"]["pv1"]
        assert irradiance == pytest.approx(0.8)  # kW/m² 直读
        assert temperature == pytest.approx(10.0 + 48 * 0.001)
        night_irradiance, _ = series.row(0)["pv"]["pv1"]
        assert night_irradiance == 0.0
        mapping = series.mapping()
        assert mapping["units"]["pv_curve"] == "kW/m2"
        assert mapping["interval_seconds"] == 900
    finally:
        session.close()


def test_timeseries_rejects_broken_peak(smartds_like_model, tmp_path):
    _profile(smartds_like_model / "profiles" / "com_kw_1_pu.csv", np.full(POINTS, 0.4))
    from envs.powergrid.smartds_timeseries import SmartDSTimeseries
    from envs.powergrid.session import DSSSession

    session = DSSSession(smartds_like_model, "feeder/Master.dss")
    try:
        with pytest.raises(ValueError, match="年峰"):
            SmartDSTimeseries(smartds_like_model, "feeder/Master.dss").bind(session)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 会话层：逐 PV 辐照、子集探测恢复、LoadMult 栅栏
# ---------------------------------------------------------------------------


def test_set_pv_conditions_mapping_only_writes_named(smartds_like_model):
    session, series = _session_and_series(smartds_like_model)
    try:
        session.set_pv_conditions(0.5, 20.0, mapping={"pv1": (0.9, 30.0)})
        assert float(session.read_property("PVSystem.pv1", "irradiance")) == pytest.approx(0.9)
        with pytest.raises(ValueError, match="不存在"):
            session.set_pv_conditions(0.0, 0.0, mapping={"nope": (1.0, 1.0)})
    finally:
        session.close()


def test_subset_probe_restores_settings_and_keeps_measurements(smartds_like_model):
    session, series = _session_and_series(smartds_like_model)
    try:
        session.begin_interval(0)
        series.apply(session, 48)
        # 探测前设定非默认值，验证探测前后完全恢复。
        session.command("Edit PVSystem.pv1 %Pmpp=60 kvar=0.3")
        available = session.available_pv_power(["pv1"])
        assert available["pv1"] > 0  # 午间辐照 0.8 kW/m²
        assert session.read_property("PVSystem.pv1", "%pmpp") == "60"
        assert float(session.read_property("PVSystem.pv1", "kvar")) == pytest.approx(0.3)
        state = session.read_state()  # 探测对量测栅栏透明
        # 注入口径"发出为正"：恢复 %Pmpp=60 后仍按恢复设定出力。
        assert state.injections["PVSystem.pv1"][0] > 0
        session.solve()
        session.finish_interval()
    finally:
        session.close()


def test_loadmult_guard_rejects_double_counting(tmp_path):
    root = tmp_path / "m"
    _write(
        root / "model.dss",
        "new circuit.t\nedit vsource.source basekV=1.0\nnew load.l1 bus1=sourcebus"
        " kW=1\nset loadmult=1.5\nsolve\n",
    )
    from envs.powergrid.session import DSSSession

    with pytest.raises(RuntimeError, match="LoadMult"):
        DSSSession(root, "model.dss")


# ---------------------------------------------------------------------------
# PV 控制权与 InvControl 守卫（设备层）
# ---------------------------------------------------------------------------


def test_invcontrol_bindings_strip_class_prefix(smartds_like_model):
    from envs.powergrid.session import DSSSession
    from envs.powergrid.devices import invcontrol_bindings

    session = DSSSession(smartds_like_model, "feeder/Master.dss")
    try:
        assert invcontrol_bindings(session) == {"ic1": ["pv1"]}
    finally:
        session.close()


def test_pv_command_request_modes():
    from envs.powergrid.devices import PVDevice, pv_command_request

    pv = PVDevice(pmpp=2.0, kva=2.0, kvar_max=1.1, kvar_max_abs=0.6, pt_curve="")
    pq = pv_command_request(pv, fraction=0.5, qcommand=1.0, available_kw=1.0)
    assert pq["kw"] == pytest.approx(0.5)
    # 容量圆剩余 = sqrt(4-0.25)，正 Q 命令受 kvar_max=1.1 限制。
    assert pq["kvar"] == pytest.approx(1.1)
    negative = pv_command_request(pv, 0.5, -1.0, 1.0)
    assert negative["kvar"] == pytest.approx(-0.6)  # 非对称吸收上限
    q_only = pv_command_request(pv, 1.0, 0.5, 1.0)
    assert q_only["kw"] == pytest.approx(1.0)  # 满可用功率


# ---------------------------------------------------------------------------
# 文件清单算例：下载补丁链路（file:// 离线）
# ---------------------------------------------------------------------------


def _fileset_case(tmp_path, patches=()):
    source = tmp_path / "upstream"
    _write(source / "feeder" / "Master.dss", "new circuit.t\n")
    _write(source / "feeder" / "PVSystems.dss", "New PVSystem.p kvarlimit=1.1\n")
    manifest = tmp_path / "files.json"
    from hashlib import sha256

    files = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "offline_test",
                "source_base": source.as_uri(),
                "files": files,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return FileSetCase(
        case_id="offline_test",
        dataset_url="https://example.invalid",
        file_manifest=str(manifest),
        layout_root="",
        entries={"main": CaseEntry(folder="feeder", master="Master.dss")},
        patches=patches,
        description="",
        citation="",
        license_note="",
    )


def test_fileset_download_applies_patches(tmp_path, monkeypatch):
    import utils.external_cases as ec

    case = _fileset_case(
        tmp_path,
        patches=(
            FilePatch(
                path="feeder/PVSystems.dss",
                find="kvarlimit=",
                replace="kvarMax=",
                reason="test",
            ),
        ),
    )
    monkeypatch.setitem(ec.FILE_SET_CASES, "offline_test", case)
    monkeypatch.setattr(ec, "ROOT", tmp_path)
    root = tmp_path / "data"
    result = ec.download("offline_test", root)
    assert result["status"] == "downloaded" and result["patched_files"] == 1
    patched = (root / "offline_test" / "feeder" / "PVSystems.dss").read_text()
    assert "kvarMax=1.1" in patched and "kvarlimit" not in patched
    assert ec.verify("offline_test", root)["status"] == "ok"
    entry = ec.locate_entry("offline_test", "main", root)
    assert entry[1] == "Master.dss"


def test_fileset_patch_miss_rejected(tmp_path, monkeypatch):
    import utils.external_cases as ec

    case = _fileset_case(
        tmp_path,
        patches=(
            FilePatch(path="feeder/PVSystems.dss", find="absent", replace="x", reason="t"),
        ),
    )
    monkeypatch.setitem(ec.FILE_SET_CASES, "offline_test", case)
    monkeypatch.setattr(ec, "ROOT", tmp_path)
    with pytest.raises(ExternalCaseError, match="未命中"):
        ec.download("offline_test", tmp_path / "data")


def test_fileset_verify_detects_tampering(tmp_path, monkeypatch):
    import utils.external_cases as ec

    case = _fileset_case(tmp_path)
    monkeypatch.setitem(ec.FILE_SET_CASES, "offline_test", case)
    monkeypatch.setattr(ec, "ROOT", tmp_path)
    root = tmp_path / "data"
    ec.download("offline_test", root)
    (root / "offline_test" / "feeder" / "Master.dss").write_text("tampered")
    report = ec.verify("offline_test", root)
    assert report["status"] == "unavailable"
    assert any("Master.dss" in problem for problem in report["problems"])


# ---------------------------------------------------------------------------
# 真实 SMART-DS 算例（显式可选：未下载数据即 skip，不假装已运行）
# ---------------------------------------------------------------------------

_SMARTDS_REPORT = verify("smart_ds_sfo_p4udt4")
_SMARTDS_READY = _SMARTDS_REPORT is not None and _SMARTDS_REPORT["status"] == "ok"
requires_smartds = pytest.mark.skipif(
    not _SMARTDS_READY,
    reason="外部算例 smart_ds_sfo_p4udt4 未下载：python scripts/download_external_case.py --case smart_ds_sfo_p4udt4",
)


@pytest.mark.slow
@requires_smartds
def test_smartds_p4udt4_inventory_and_timeseries():
    from envs.powergrid.session import DSSSession
    from envs.powergrid.smartds_timeseries import SmartDSTimeseries
    from utils.external_cases import locate_entry

    folder, master = locate_entry("smart_ds_sfo_p4udt4", "solar_high_none")
    layout = folder.parents[4]
    rel = str(folder.relative_to(layout)) + "/" + master
    closure = dependency_closure(layout, rel)
    assert closure["missing"] == []
    assert sum(1 for f in closure["files"].values() if f["kind"] == "datafile") == 45

    session = DSSSession(layout, rel, step_seconds=900)
    try:
        series = SmartDSTimeseries(layout, rel).bind(session)
        assert len(session.loads) == 36
        assert len(session.pvs) == 6
        # 中心抽头：14 对负荷对象共享曲线（28 个），铭牌各半无重复计负荷。
        from collections import defaultdict

        by_shape = defaultdict(list)
        for name, shape in series.load_shapes.items():
            by_shape[shape].append(name)
        assert sum(1 for ns in by_shape.values() if len(ns) == 2) == 14
        # 夜间可用功率为 0（官方口径与引擎一致）。
        session.begin_interval(0)
        series.apply(session, 0)
        assert set(session.available_pv_power(list(session.pvs)).values()) == {0.0}
        session.solve()
        session.finish_interval()
        # 午间：满发探测为正。
        session.begin_interval(900)
        series.apply(session, 48)
        assert all(v > 0 for v in session.available_pv_power(list(session.pvs)).values())
        session.solve()
        session.finish_interval()
    finally:
        session.close()


@pytest.mark.slow
@requires_smartds
def test_smartds_pv_control_modes_end_to_end():
    """rl 全接管停用 InvControl；fixed 不写不探测且奖励/观测有限。"""
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_power_env

    def build(mode):
        values = {
            "env": {
                "data": {"step_seconds": 900},
                "pv_control_mode": mode,
                "max_episode_steps": 4,
                "pv_plan": "none",
            }
        }
        cfg = ConfigLoader().load(
            "happo", "smartgrid", system_ref="SMARTDS_P4UDT4", values=values
        )
        return make_power_env("smartgrid", cfg.env_args, 0, 0)

    env = build("rl")
    try:
        core = env.env
        core.reset()
        actions = [
            s.n // 2 if s.__class__.__name__ == "Discrete"
            else np.zeros(s.shape, dtype=np.float32) + 0.5
            for s in core.action_space
        ]
        result = core.step(actions)
        assert np.isfinite(result[1])
        manifest = core.get_physics_manifest()["pv_control"]
        assert len(manifest["controlled"]) == 6
        assert manifest["disabled_invcontrols"]
        assert core.circuit.read_property(
            "InvControl.invpvctrvv_pv_load_p4ulv40", "enabled"
        ).lower() == "no"
    finally:
        env.close()

    env = build("fixed")
    try:
        core = env.env
        core.reset()
        # fixed：PV 不被写入，保持模型 pf 设定。
        pf = core.circuit.read_property("PVSystem.pv_load_p4ulv25", "pf")
        result = core.step([])
        assert np.isfinite(result[1])
        assert core.get_physics_manifest()["pv_control"]["controlled"] == []
        assert core.circuit.read_property("PVSystem.pv_load_p4ulv25", "pf") == pf
    finally:
        env.close()


_RDT_REPORT = verify("smart_ds_gso_rdt1534")
_RDT_READY = _RDT_REPORT is not None and _RDT_REPORT["status"] == "ok"
requires_rdt = pytest.mark.skipif(
    not _RDT_READY,
    reason="外部算例 smart_ds_gso_rdt1534 未下载：python scripts/download_external_case.py --case smart_ds_gso_rdt1534",
)


@pytest.mark.slow
@requires_rdt
def test_smartds_rdt1534_loads_and_timeseries():
    from envs.powergrid.session import DSSSession
    from envs.powergrid.smartds_timeseries import SmartDSTimeseries
    from utils.external_cases import locate_entry

    folder, master = locate_entry("smart_ds_gso_rdt1534", "solar_high_none")
    layout = folder.parents[4]
    rel = str(folder.relative_to(layout)) + "/" + master
    session = DSSSession(layout, rel)
    try:
        series = SmartDSTimeseries(layout, rel).bind(session)
        assert len(session.pvs) == 472
        assert len(session.loads) > 1000
        row = series.row(48)
        assert all(0 <= irr for irr, _ in row["pv"].values())
    finally:
        session.close()
