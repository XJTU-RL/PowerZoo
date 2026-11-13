"""外部算例接入测试：合成小模型单测（无网络、无大数据）+ UK 算例显式可选测试。

真实 UK 算例测试标记 requires_external_case，未下载数据时明确 skip 并给出
下载命令；合成小模型复刻 UK 结构（主入口 redirect 子网、禁用中压负荷、
原生 RegControl），覆盖来源校验、依赖闭包、清单盘点与会话加载全链路。
"""

from pathlib import Path
import zipfile

import pytest

from utils.external_cases import (
    CASES,
    CASES as _CASES,
    ExternalCase,
    ExternalCaseError,
    download,
    external_root,
    locate,
    verify,
)
from envs.powergrid.case_inventory import (
    compare,
    dependency_closure,
    inventory,
    node_voltages,
)

pytestmark = [pytest.mark.requires_opendss]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def tiny_model(tmp_path):
    """复刻 UK 结构的微型 MV-LV 模型：redirect 子网 + 禁用中压负荷 + RegControl。"""
    root = tmp_path / "HV_UG_tiny"
    _write(
        root / "master.dss",
        "\n".join(
            [
                "clear",
                "set defaultbasefrequency=50",
                "new circuit.tiny_mvlv",
                "edit vsource.source basekV=33 pu=1.02 R1=0.0001 X1=0.0001 R0=0.0001 X0=0.0001",
                "new line.sourceZ bus1=SourceBus bus2=301 phases=3 r1=0.0001 x1=0.5445 r0=0.0001 x0=0.0001",
                "redirect transformers.dss",
                "redirect regcontrols.dss",
                "redirect lines.dss",
                "redirect loads.dss",
                "redirect lv_network\\lv.dss",
                "redirect lds_edit.dss",
                "set voltagebases=[33.0, 11.0, 0.416]",
                "calcvoltagebases",
                "solve",
            ]
        )
        + "\n",
    )
    _write(
        root / "transformers.dss",
        "new transformer.t1 phases=3 windings=2 buses=[301,1100] conns=[delta,wye]"
        " kVs=[33,11] %Rs=\"0.44,0.44\" xhl=19.97 kVA=5000 MaxTap=1.1 MinTap=0.9"
        " NumTaps=5 tap=1.0\n",
    )
    _write(
        root / "regcontrols.dss",
        "new regcontrol.rc1 transformer=t1 winding=2 vreg=104 band=2 ptratio=30"
        " maxtapchange=5\n",
    )
    _write(
        root / "lines.dss",
        "new line.mv1 bus1=1100 bus2=1101 phases=3 r1=0.2 x1=0.1 r0=0.6 x0=0.3\n",
    )
    _write(
        root / "loads.dss",
        "new load.load_1_1 phases=3 kV=11 conn=wye model=1 bus1=1101 kW=800"
        " kvar=200 enabled=True\n"
        "new load.load_1_2 phases=3 kV=11 conn=wye model=1 bus1=1101 kW=400"
        " kvar=100 enabled=True\n",
    )
    _write(
        root / "lv_network" / "lv.dss",
        "new transformer.lvtr1 phases=3 windings=2 buses=[1101,1_1101_1]"
        " conns=[delta,wye] kVs=[11,0.416] kVA=500 XHL=5.0 %Rs=\"0.5,0.5\" tap=1.0\n"
        "new line.lv1 bus1=1_1101_1 bus2=1_1101_11 phases=3 r1=0.1 x1=0.05"
        " r0=0.3 x0=0.1\n"
        "new load.LV1 phases=1 bus1=1_1101_11.1 kV=0.24 kW=1 PF=0.95\n"
        "new load.LV2 phases=1 bus1=1_1101_11.2 kV=0.24 kW=1 PF=0.95\n",
    )
    _write(root / "lds_edit.dss", "edit load.load_1_2 enabled=False\n")
    return root


def _compile(tiny_model):
    from dss import DSS

    ctx = DSS.NewContext()
    ctx.Text.Command = f'compile "{tiny_model / "master.dss"}"'
    return ctx


# ---------------------------------------------------------------------------
# 依赖闭包
# ---------------------------------------------------------------------------


def test_dependency_closure_follows_redirects_and_backslashes(tiny_model):
    closure = dependency_closure(tiny_model, "master.dss")
    assert closure["missing"] == []
    names = set(closure["files"])
    assert {
        "master.dss",
        "transformers.dss",
        "regcontrols.dss",
        "lines.dss",
        "loads.dss",
        "lv_network/lv.dss",
        "lds_edit.dss",
    } <= names
    assert all(len(info["sha256"]) == 64 for info in closure["files"].values())


def test_dependency_closure_reports_missing_file_and_line(tiny_model):
    master = tiny_model / "master.dss"
    lines = master.read_text(encoding="utf-8").splitlines()
    lines.insert(9, "redirect lv_network/absent.dss")
    master.write_text("\n".join(lines) + "\n", encoding="utf-8")
    closure = dependency_closure(tiny_model, "master.dss")
    assert len(closure["missing"]) == 1
    entry = closure["missing"][0]
    assert entry["from"].startswith("master.dss:")
    assert entry["from"].split(":")[1] == "10"
    assert "absent.dss" in entry["path"]


# ---------------------------------------------------------------------------
# 引擎清单
# ---------------------------------------------------------------------------


def test_inventory_counts_on_tiny_model(tiny_model):
    ctx = _compile(tiny_model)
    report = inventory(ctx, "tiny")
    assert report["circuit"]["frequency_hz"] == 50.0
    assert report["circuit"]["num_buses"] >= 5
    assert report["circuit"]["voltage_bases_kv"] == [33.0, 11.0, 0.416]
    assert report["sources"][0]["base_kv"] == 33.0
    loads = report["loads"]
    assert loads["total"] == 4
    assert loads["enabled"] == 3
    assert loads["disabled"] == 1
    assert loads["disabled_names"] == ["load_1_2"]
    assert loads["phases"] == {"3": 2, "1": 2}
    regulators = report["regcontrols"]
    assert len(regulators) == 1
    assert regulators[0]["numtaps"] == 5
    assert regulators[0]["vreg"] == 104.0
    assert regulators[0]["mintap"] == 0.9
    assert regulators[0]["maxtap"] == 1.1
    assert report["pvsystems"] == []
    assert report["storages"] == []
    assert report["capacitors"] == []
    assert report["generators"] == []
    assert report["lines"]["total"] == 3


def test_inventory_solved_sections_present(tiny_model):
    ctx = _compile(tiny_model)
    report = inventory(ctx, "tiny")
    assert report["solver"]["converged"] is True
    assert report["voltage_summary"]["phase_nodes"] > 0
    assert report["power"]["total_power_kw_kvar"][0] > 0
    assert report["power"]["losses_kw_kvar"][0] > 0


def test_inventory_refuses_unsolved_voltage_section(tiny_model):
    ctx = _compile(tiny_model)
    ctx.ActiveCircuit.Solution.Converged = 0
    with pytest.raises(RuntimeError, match="未收敛"):
        inventory(ctx, "tiny", solved=True)


# ---------------------------------------------------------------------------
# PowerZoo 会话加载（覆盖停用元件节点序扩展）
# ---------------------------------------------------------------------------


def test_session_loads_model_with_disabled_elements(tiny_model):
    """含编译期停用元件的模型必须能被 DSSSession 加载并原生控制求解。"""
    from envs.powergrid.session import DSSSession, _derived_node_order

    session = DSSSession(tiny_model, "master.dss")
    try:
        connection = session.connection("load.load_1_2")
        assert connection["enabled"] is False
        assert connection["nodes"][0] in (1, 2, 3)
        # 停用元件节点序与启用元件的原生 NodeOrder 语义一致（逐端子补中性 0）。
        session.require_element("load.load_1_1")
        assert _derived_node_order(session.native.ActiveCktElement) == list(
            map(int, session.native.ActiveCktElement.NodeOrder)
        )
        session.solve(native_controls=True)
        assert session.native.Solution.Converged
    finally:
        session.close()


def test_engine_vs_session_comparison_matches(tiny_model):
    """A2（引擎对齐会话设置）与 B（DSSSession）对照必须逐点一致。"""
    from envs.powergrid.session import DSSSession

    ctx = _compile(tiny_model)
    solution = ctx.ActiveCircuit.Solution
    solution.ControlMode = 0
    for command in ("CalcVoltageBases", "Set maxiterations=100 tolerance=1e-8"):
        ctx.Text.Command = command
    solution.SolveSnap()
    assert solution.Converged
    a = inventory(ctx, "A2")
    a["_node_voltages"] = node_voltages(ctx)

    session = DSSSession(tiny_model, "master.dss")
    try:
        session.solve(native_controls=True)
        b = inventory(session.dss, "B")
        b["_node_voltages"] = node_voltages(session.dss)
        result = compare(a, b)
        assert result["match"], result
        assert result["node_voltages"]["max_abs_diff_pu"] <= 2e-6
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 下载/校验/定位（file:// 伪算例，不触网）
# ---------------------------------------------------------------------------


@pytest.fixture
def packaged_tiny_case(tiny_model, tmp_path, monkeypatch):
    """把微型模型按登记算例的布局打包成 zip，注入临时 case 登记。"""
    zip_path = tmp_path / "HV_UG_tiny.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in sorted(tiny_model.rglob("*")):
            if file.is_file():
                archive.write(file, f"HV_UG_tiny/{file.relative_to(tiny_model).as_posix()}")
    from hashlib import sha256

    digest = sha256(zip_path.read_bytes()).hexdigest()
    case = ExternalCase(
        case_id="tiny_test",
        repo_url="https://example.invalid/repo",
        upstream_commit="0" * 40,
        zip_url=zip_path.as_uri(),
        zip_name="HV_UG_tiny.zip",
        zip_sha256=digest,
        zip_bytes=zip_path.stat().st_size,
        zip_top_dir="HV_UG_tiny",
        master="master.dss",
        description="测试用",
        citation="测试用",
        license_note="测试用",
    )
    monkeypatch.setitem(CASES, "tiny_test", case)
    return case, tmp_path / "external"


def test_download_verify_locate_roundtrip(packaged_tiny_case):
    case, root = packaged_tiny_case
    result = download(case.case_id, root)
    assert result["status"] == "downloaded"
    report = verify(case.case_id, root)
    assert report["status"] == "ok"
    assert report["upstream_commit"] == case.upstream_commit
    model = locate(case.case_id, root)
    assert (model / "master.dss").is_file()
    # 重复下载默认跳过
    assert download(case.case_id, root)["status"] == "already_present"


def test_verify_reports_precise_problems(packaged_tiny_case):
    case, root = packaged_tiny_case
    report = verify(case.case_id, root)
    assert report["status"] == "unavailable"
    joined = "；".join(report["problems"])
    assert str(root / case.case_id) in joined
    with pytest.raises(ExternalCaseError, match="download_external_case"):
        locate(case.case_id, root)


def test_download_rejects_tampered_zip(tiny_model, tmp_path, monkeypatch):
    """登记 SHA-256 与实际不符时必须删除产物并报错。"""
    zip_path = tmp_path / "tampered.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("HV_UG_tiny/master.dss", "tampered")
    case = ExternalCase(
        case_id="tampered",
        repo_url="https://example.invalid/repo",
        upstream_commit="0" * 40,
        zip_url=zip_path.as_uri(),
        zip_name="HV_UG_tiny.zip",
        zip_sha256="0" * 64,
        zip_bytes=zip_path.stat().st_size,
        zip_top_dir="HV_UG_tiny",
        master="master.dss",
        description="",
        citation="",
        license_note="",
    )
    monkeypatch.setitem(CASES, "tampered", case)
    root = tmp_path / "external"
    with pytest.raises(ExternalCaseError, match="校验失败"):
        download("tampered", root)
    assert not (root / "tampered" / "HV_UG_tiny.zip").exists()


def test_zip_member_escape_rejected(tmp_path, monkeypatch):
    """压缩包含越界成员（绝对路径/../）时拒绝解压。"""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../escape.txt", "x")
    case = ExternalCase(
        case_id="evil",
        repo_url="https://example.invalid/repo",
        upstream_commit="0" * 40,
        zip_url=zip_path.as_uri(),
        zip_name="evil.zip",
        zip_sha256="0" * 64,
        zip_bytes=zip_path.stat().st_size,
        zip_top_dir="HV_UG_tiny",
        master="master.dss",
        description="",
        citation="",
        license_note="",
    )
    monkeypatch.setitem(CASES, "evil", case)
    # 先放一个通过校验的假 zip：sha 不匹配会先被拦，这里直接检验安全成员检查
    from utils.external_cases import _safe_members

    with zipfile.ZipFile(zip_path) as archive, pytest.raises(Exception):
        _safe_members(archive, case)


# ---------------------------------------------------------------------------
# 真实 UK 算例（显式可选：未下载数据即 skip，不假装已运行）
# ---------------------------------------------------------------------------

UK_REPORT = verify("uk_mvlv_hv_ug")
UK_READY = UK_REPORT is not None and UK_REPORT["status"] == "ok"
requires_uk = pytest.mark.skipif(
    not UK_READY,
    reason="外部算例 uk_mvlv_hv_ug 未下载：python scripts/download_external_case.py --case uk_mvlv_hv_ug",
)


@pytest.mark.slow
@requires_uk
def test_uk_case_verifies_and_inventories():
    model = locate("uk_mvlv_hv_ug")
    closure = dependency_closure(model, "master_mvlv.dss")
    assert closure["missing"] == []
    assert len(closure["files"]) > 1000

    from dss import DSS

    ctx = DSS.NewContext()
    ctx.Text.Command = f'compile "{model / "master_mvlv.dss"}"'
    report = inventory(ctx, "uk")
    # 数值为登记版本（SHA-256 钉扎）+ 实测引擎的盘点结果，不凑参考值。
    assert report["circuit"]["num_buses"] == 37629
    assert report["circuit"]["num_nodes"] == 112887
    assert report["circuit"]["frequency_hz"] == 50.0
    assert report["loads"]["total"] == 19072
    assert report["loads"]["disabled"] == 75  # lds_edit.dss 禁用的中压集中负荷
    assert report["loads"]["enabled"] == 19072 - 75
    assert len(report["regcontrols"]) == 2
    assert all(reg["numtaps"] == 19 for reg in report["regcontrols"])
    assert report["pvsystems"] == []
    assert report["storages"] == []
    assert report["capacitors"] == []
    assert report["generators"] == []
    # 原始快照收敛但存在越限：按原样报告，不修改源模型。
    assert report["voltage_summary"]["above_1.05_pu"] > 50000
    assert report["power"]["total_power_kw_kvar"][0] == pytest.approx(19957.3, rel=1e-4)


@pytest.mark.slow
@requires_uk
def test_uk_engine_vs_session_baseline_matches():
    """UK 算例 A2（引擎对齐会话设置）vs B（DSSSession 原生控制）必须一致。"""
    from dss import DSS
    from envs.powergrid.session import DSSSession

    model = locate("uk_mvlv_hv_ug")
    master = model / "master_mvlv.dss"

    ctx = DSS.NewContext()
    ctx.Text.Command = f'compile "{master}"'
    solution = ctx.ActiveCircuit.Solution
    solution.ControlMode = 0
    for command in (
        "CalcVoltageBases",
        "Set maxiterations=100 tolerance=1e-8",
    ):
        ctx.Text.Command = command
    solution.SolveSnap()
    assert solution.Converged and solution.ControlActionsDone
    a = inventory(ctx, "A2")
    a["_node_voltages"] = node_voltages(ctx)

    session = DSSSession(model, "master_mvlv.dss")
    try:
        session.solve(native_controls=True)
        b = inventory(session.dss, "B")
        b["_node_voltages"] = node_voltages(session.dss)
        result = compare(a, b)
        assert result["match"], result
    finally:
        session.close()
