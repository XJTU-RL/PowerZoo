"""SMART-DS 真实逐设备时序的训练环境接入与按天留出测试。"""

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.envs_tools import make_power_env
from utils.external_cases import verify

pytestmark = [pytest.mark.requires_opendss]

_SMARTDS_OK = verify("smart_ds_sfo_p4udt4")["status"] == "ok"
requires_smartds = pytest.mark.skipif(
    not _SMARTDS_OK,
    reason="外部算例 smart_ds_sfo_p4udt4 未下载",
)

SERIES_SPEC = {
    "case": "smart_ds_sfo_p4udt4",
    "entry": "solar_high_none",
    "train_days": [0, 1],
    "validation_days": [2],
}


def _env(split=None, local=True):
    spec = dict(SERIES_SPEC)
    if split:
        spec["split"] = split
    values = {
        "env": {
            "smartds_series": spec,
            "max_episode_steps": 96,
            "pv_plan": "none",
            "pv_control_mode": "rl",
            "data": {"step_seconds": 900},
            "light_state_read": True,
        }
    }
    if local:
        values["env"]["observation_mode"] = "local"
    cfg = ConfigLoader().load(
        "happo", "smartgrid", system_ref="SMARTDS_P4UDT4", values=values
    )
    return make_power_env("smartgrid", cfg.env_args, 0, 0)


@requires_smartds
def test_train_split_only_sees_train_days():
    env = _env(split="train")
    try:
        days = [entry["id"] for entry, _ in env.env.dataset.episodes]
        assert days == ["day000", "day001"]
    finally:
        env.close()


@requires_smartds
def test_validation_split_holds_out_unseen_day():
    """评估只见未训练天（按天留出，杜绝相邻时刻泄漏）。"""
    env = _env(split="validation")
    try:
        days = [entry["id"] for entry, _ in env.env.dataset.episodes]
        assert days == ["day002"]
        assert set(days).isdisjoint({"day000", "day001"})
    finally:
        env.close()


@requires_smartds
def test_real_timeseries_written_per_device():
    """env 步进写入的逐负荷 kW 与官方曲线逐位一致（含日偏移与步进）。"""
    env = _env(split="train")
    try:
        core = env.env
        env.reset()
        series = core.dataset._ensure_series()
        day = core.entry["day"]
        name = list(series.load_shapes)[0]
        written = float(core.circuit.read_property(f"Load.{name}", "kW"))
        expect = series.row(day * 96)["loads"][name][0]
        assert written == pytest.approx(expect, abs=1e-9)
        actions = [
            s.n // 2 if s.__class__.__name__ == "Discrete"
            else np.zeros(s.shape, dtype=np.float32) + 0.3
            for s in env.action_space
        ]
        env.step(actions)  # t=0
        env.step(actions)  # t=1
        written = float(core.circuit.read_property(f"Load.{name}", "kW"))
        expect = series.row(day * 96 + 1)["loads"][name][0]
        assert written == pytest.approx(expect, abs=1e-9)
        # 全网写入量级核对：Σ铭牌×pu 与引擎总负荷同阶（无重复计负荷）。
        row = series.row(day * 96 + 1)
        total_expected = sum(v[0] for v in row["loads"].values())
        engine_total = float(core.circuit.total_load_power()[0])
        assert engine_total == pytest.approx(total_expected, rel=0.02)
    finally:
        env.close()


@requires_smartds
def test_two_days_data_differ():
    """留出的不同天数据不同（防止“每天同曲线”的假留出）。"""
    env = _env(split="train")
    try:
        series = env.env.dataset._ensure_series()
        name = list(series.load_shapes)[0]
        assert series.row(0)["loads"][name][0] != series.row(96)["loads"][name][0]
    finally:
        env.close()


# ---- 按天留出门禁（无外部数据即可验证；构造不读文件，bind 才读） ----------

from envs.powergrid.smartds_timeseries import SmartDSEpisodeDataset


def _dataset(train, validation, split="train"):
    return SmartDSEpisodeDataset(
        "/nonexistent/layout",
        "Master.dss",
        days={"train": train, "validation": validation},
        split=split,
    )


def test_day_holdout_rejects_overlapping_days():
    """train 与 validation 重叠必须立即失败（曾接受重叠配置，双方 reset 到同一天）。"""
    with pytest.raises(ValueError, match="不相交"):
        _dataset([0], [0])
    with pytest.raises(ValueError, match="不相交"):
        _dataset([0, 1], [1, 2])


def test_day_holdout_rejects_duplicate_days():
    with pytest.raises(ValueError, match="重复"):
        _dataset([2, 2], [])


@pytest.mark.parametrize("bad", [-1, 365, 0.5, "3", True, None])
def test_day_holdout_rejects_invalid_day_values(bad):
    """日号必须是 0..364 的整数（排除 bool），非法值立即报错而非 worker 内崩溃。"""
    with pytest.raises(ValueError):
        _dataset([bad], [])


def test_day_holdout_rejects_unknown_split():
    with pytest.raises(ValueError, match="split"):
        _dataset([0], [1], split="test")


def test_day_holdout_accepts_valid_disjoint_days():
    dataset = _dataset([0, 1], [2])
    assert dataset.day_indices == {"train": [0, 1], "validation": [2]}
    validation = _dataset([0, 1], [2], split="validation")
    assert validation.day_indices["validation"] == [2]


def test_evaluation_worker_requires_nonempty_holdout():
    """启用评估但 validation_days 为空：主进程 spawn 前明确报错。"""
    from utils.envs_tools import _vector_env

    env_args = {
        "smartds_series": {
            "case": "smart_ds_sfo_p4udt4",
            "entry": "solar_high_none",
            "train_days": [0, 1],
            "validation_days": [],
        },
        "data": {"step_seconds": 900},
    }
    with pytest.raises(ValueError, match="validation_days"):
        _vector_env("smartgrid_smartds_p4udt4", 0, 1, env_args, evaluation=True)


# ---- 支路统计三态契约：未采集 / 已采集无额定 / 已采集有读数 ----------

def test_branch_stats_uncollected_is_null_not_zero():
    """light 读取未采集支路明细：最大值与计数为 None + collected=False，不得记 0。"""
    from envs.powergrid.control_env import summarize_branch_stats

    assert summarize_branch_stats(None) == {
        "branch_stats_collected": False,
        "max_branch_loading_pu": None,
        "branch_overload_count": None,
        "unrated_branch_count": None,
    }


def test_branch_stats_collected_empty_network():
    from envs.powergrid.control_env import summarize_branch_stats

    stats = summarize_branch_stats({})
    assert stats["branch_stats_collected"] is True
    assert stats["max_branch_loading_pu"] is None
    assert stats["branch_overload_count"] == 0
    assert stats["unrated_branch_count"] == 0


def test_branch_stats_all_unrated_not_claimed_safe():
    """已采集但全无额定值：不得宣称整个网络没有过载。"""
    from envs.powergrid.control_env import summarize_branch_stats

    stats = summarize_branch_stats(
        {"line.l1": {"loading_pu": None}, "line.l2": {"loading_pu": None}}
    )
    assert stats["max_branch_loading_pu"] is None
    assert stats["branch_overload_count"] == 0
    assert stats["unrated_branch_count"] == 2


def test_branch_stats_partial_unrated_with_overload():
    from envs.powergrid.control_env import summarize_branch_stats

    stats = summarize_branch_stats(
        {
            "line.l1": {"loading_pu": 1.2},
            "line.l2": {"loading_pu": 0.8},
            "transformer.t1": {"loading_pu": None},
        }
    )
    assert stats["max_branch_loading_pu"] == 1.2
    assert stats["branch_overload_count"] == 1
    assert stats["unrated_branch_count"] == 1


@requires_smartds
def test_light_step_info_reports_branch_stats_uncollected():
    """真实 light 步 info：collected=False 且支路统计为 None（复现：曾记 0）。"""
    env = _env()
    try:
        env.reset()
        action = [
            space.sample() for space in env.action_space
        ]
        _, _, _, _, infos, _ = env.step(action)
        info = infos[0]
        assert info["branch_stats_collected"] is False
        assert info["max_branch_loading_pu"] is None
        assert info["branch_overload_count"] is None
        assert info["unrated_branch_count"] is None
    finally:
        env.close()


@requires_smartds
def test_full_step_info_reports_branch_stats_collected():
    """真实 full 步 info：collected=True 且计数为整数。"""
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_power_env

    cfg = ConfigLoader().load(
        "happo",
        "smartgrid",
        system_ref="SMARTDS_P4UDT4",
        values={
            "env": {
                "smartds_series": dict(SERIES_SPEC),
                "max_episode_steps": 4,
                "pv_plan": "none",
                "pv_control_mode": "rl",
                "data": {"step_seconds": 900},
                "light_state_read": False,
            }
        },
    )
    env = make_power_env("smartgrid", cfg.env_args, 0, 0)
    try:
        env.reset()
        action = [
            space.sample() for space in env.action_space
        ]
        _, _, _, _, infos, _ = env.step(action)
        info = infos[0]
        assert info["branch_stats_collected"] is True
        assert isinstance(info["branch_overload_count"], int)
        assert isinstance(info["unrated_branch_count"], int)
    finally:
        env.close()


def test_check_smartds_branch_summary_without_or_zero():
    """check_smartds 聚合：无额定支路不得经 `or 0` 被当作 0 载荷统计。"""
    from types import SimpleNamespace
    from scripts.check_smartds import branch_summary

    state = SimpleNamespace(
        branches={
            "line.l1": {"loading_pu": 1.5},
            "line.l2": {"loading_pu": None},
        }
    )
    summary = branch_summary(state)
    assert summary["overloaded_branches"] == 1
    assert summary["max_branch_loading_pu"] == 1.5
    assert summary["unrated_branches"] == 1

    empty = SimpleNamespace(branches={})
    assert branch_summary(empty)["max_branch_loading_pu"] is None
    assert branch_summary(empty)["overloaded_branches"] == 0


# ---- 真实时序数据冻结（复审 4d8af3b5：摘要只含路径，绑不住曲线内容） ------

def _copy_case(tmp_path):
    """只读复制真实 p4udt4 数据到临时目录，返回其 layout 与入口相对路径。"""
    import shutil

    from utils.external_cases import locate_entry

    folder, master = locate_entry("smart_ds_sfo_p4udt4", "solar_high_none")
    layout = folder.parents[4]
    target = tmp_path / "case"
    shutil.copytree(layout, target)
    return target, str(folder.relative_to(layout)) + "/" + master


def _days():
    return {"train": [0, 1], "validation": [2]}


@requires_smartds
def test_dataset_digest_binds_load_curve_content(tmp_path):
    """修改任一负荷曲线字节后摘要必须改变；恢复校验拒绝旧摘要。"""
    from envs.powergrid.smartds_timeseries import SmartDSEpisodeDataset

    layout, rel = _copy_case(tmp_path)
    before = SmartDSEpisodeDataset(layout, rel, days=_days()).sha256

    curves = sorted((layout / "profiles").rglob("*.csv"))
    assert curves, "测试预期 profiles 目录存在负荷曲线"
    victim = curves[0]
    victim.write_bytes(victim.read_bytes() + b"\n")

    after = SmartDSEpisodeDataset(layout, rel, days=_days()).sha256
    assert after != before, "负荷曲线内容未绑定进摘要"

    stale = SmartDSEpisodeDataset(
        layout, rel, days=_days(), content_sha256=before
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="摘要与训练快照不符"):
            _ = stale.sha256
    # 期望值正确时不拒绝
    ok = SmartDSEpisodeDataset(layout, rel, days=_days(), content_sha256=after)
    assert ok.sha256 == after


@requires_smartds
def test_dataset_digest_stable_across_directory_move(tmp_path):
    """内容不变仅搬目录：摘要不得因路径不同而误判版本。"""
    from envs.powergrid.smartds_timeseries import SmartDSEpisodeDataset

    layout, rel = _copy_case(tmp_path)
    original = SmartDSEpisodeDataset(layout, rel, days=_days()).sha256

    moved = tmp_path / "renamed"
    layout.rename(moved)
    assert SmartDSEpisodeDataset(moved, rel, days=_days()).sha256 == original


@requires_smartds
def test_dataset_digest_binds_pv_curve_and_mapping(tmp_path):
    """PV 辐照/温度曲线与设备→曲线映射脚本都参与摘要。"""
    from envs.powergrid.smartds_timeseries import SmartDSEpisodeDataset

    layout, rel = _copy_case(tmp_path)
    base = SmartDSEpisodeDataset(layout, rel, days=_days()).sha256

    solar = sorted((layout / "solar_data").glob("*.csv"))
    assert solar, "测试预期 solar_data 目录存在 PV 曲线"
    victim = solar[0]
    original = victim.read_bytes()
    victim.write_bytes(original + b"\n")
    assert (
        SmartDSEpisodeDataset(layout, rel, days=_days()).sha256 != base
    ), "PV/温度曲线未绑定进摘要"

    # 恢复数据后改映射脚本（LoadShape 定义所在 loadshapes.dss）
    victim.write_bytes(original)
    assert SmartDSEpisodeDataset(layout, rel, days=_days()).sha256 == base
    from envs.powergrid.smartds_timeseries import SmartDSTimeseries

    definitions = list(SmartDSTimeseries(layout, rel).shapes.values())
    first, second = definitions[:2]
    assert first["mult"] != second["mult"]
    script = layout / first["origin"].rsplit(":", 1)[0]
    script.write_text(
        script.read_text(encoding="utf-8").replace(first["mult"], second["mult"], 1),
        encoding="utf-8",
    )
    assert (
        SmartDSEpisodeDataset(layout, rel, days=_days()).sha256 != base
    ), "设备→曲线映射脚本未绑定进摘要"


@requires_smartds
def test_prepare_run_freezes_smartds_content_not_synthetic():
    """prepare_run 冻结真实数据摘要与外部清单；不再登记未使用的合成清单。"""
    from utils.config_pipeline import ConfigLoader
    from utils.training_setup import prepare_run

    cfg = ConfigLoader().load(
        "happo",
        "smartgrid",
        system_ref="SMARTDS_P4UDT4",
        values={
            "env": {
                "smartds_series": dict(SERIES_SPEC),
                "max_episode_steps": 4,
                "pv_plan": "none",
                "pv_control_mode": "rl",
                "data": {"step_seconds": 900},
            }
        },
    )
    cfg.main_args["env"] = "smartgrid_smartds_p4udt4"
    cfg.algo_args.setdefault("seed", {"seed_specify": True, "seed": 7})

    prepare_run(cfg)

    spec = cfg.env_args["smartds_series"]
    assert len(spec["dataset_sha256"]) == 64, "真实曲线内容摘要未冻结"
    assert len(spec["manifest_sha256"]) == 64, "外部清单指纹未冻结"
    assert len(spec.get("dataset_sha256_validation", "")) == 64, (
        "评估 split 摘要未冻结——评估 worker 会拿 train 摘要误判版本"
    )
    assert spec["dataset_sha256_validation"] != spec["dataset_sha256"], (
        "split 维度应参与摘要（train 与 validation 是不同数据切片）"
    )
    assert spec["days"] == {"train": [0, 1], "validation": [2]}
    assert spec["master"].endswith("Master.dss")
    # 未使用的合成场景清单不得被登记为数据来源：data 保持配置默认
    # （manifest_sha256 未被 runtime 写入），文件表与来源表只认真实数据。
    assert cfg.env_args["data"].get("manifest_sha256") is None
    assert not [
        p for p in cfg.files if "results/powergrid_data" in p
    ], "合成场景清单被登记进运行文件表"
    for key in ("env.data.path", "env.data.manifest_sha256"):
        assert not [
            e for e in cfg.history.get(key, []) if "dataset preparation" in e["source"]
        ], f"{key} 被合成数据准备写入"
    manifest_files = [
        p for p in cfg.files if p.endswith("smart_ds_sfo_p4udt4/manifest.json")
    ]
    assert manifest_files, "外部清单未登记进运行文件表"
    assert cfg.files[manifest_files[0]] == spec["manifest_sha256"]
    # 冻结的摘要与 worker 实际构造一致（恢复校验可通过）
    from utils.envs_tools import make_power_env

    env = make_power_env("smartgrid", cfg.env_args, 0, 0)
    try:
        env.reset()
        assert env.env.dataset.sha256 == spec["dataset_sha256"]
        trained_manifest = env.env.get_physics_manifest()
    finally:
        env.close()

    from copy import deepcopy
    from utils.envs_tools import make_eval_env
    from utils.physics_manifest import compare_manifests

    evaluation = make_eval_env("smartgrid", 0, 1, cfg.env_args)
    try:
        evaluation.reset()
        result = evaluation.step([[space.sample() for space in evaluation.action_space]])
        info = result[4][0][0]
        assert info["scenario_id"] == "day002"
        assert info["smartds_split"] == "validation"
        assert info["dataset_sha256"] == spec["dataset_sha256_validation"]
        evaluated_manifest = evaluation.get_physics_manifest()
        compare_manifests(trained_manifest, evaluated_manifest)
        assert trained_manifest["smartds_dataset"]["split_sha256"] == {
            "train": spec["dataset_sha256"],
            "validation": spec["dataset_sha256_validation"],
        }
        assert trained_manifest["smartds_dataset"]["mapping"]["load_shapes"]
        changed = deepcopy(evaluated_manifest)
        changed["smartds_dataset"]["days"]["train"] = [10]
        with pytest.raises(ValueError, match="smartds_dataset"):
            compare_manifests(trained_manifest, changed)
        legacy = deepcopy(trained_manifest)
        del legacy["smartds_dataset"]
        with pytest.raises(ValueError, match="smartds_dataset"):
            compare_manifests(legacy, evaluated_manifest)
    finally:
        evaluation.close()

    stale = deepcopy(cfg.env_args)
    stale["smartds_series"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="清单摘要"):
        make_power_env("smartgrid", stale, 0, 0)

    stale = deepcopy(cfg.env_args)
    stale["smartds_series"]["split"] = "validation"
    stale["smartds_series"]["dataset_sha256_validation"] = "0" * 64
    with pytest.raises(ValueError, match="摘要与训练快照"):
        make_power_env("smartgrid", stale, 0, 0)
    del stale["smartds_series"]["dataset_sha256_validation"]
    with pytest.raises(ValueError, match="缺少当前 split"):
        make_power_env("smartgrid", stale, 0, 0)

    original_validation = cfg.env_args["smartds_series"]["dataset_sha256_validation"]
    cfg.env_args["smartds_series"]["dataset_sha256_validation"] = "0" * 64
    with pytest.raises(ValueError, match="validation 摘要与训练快照"):
        prepare_run(cfg)
    cfg.env_args["smartds_series"]["dataset_sha256_validation"] = original_validation
    cfg.env_args["smartds_series"]["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="摘要与训练快照"):
        prepare_run(cfg)


@pytest.mark.parametrize("voltages,expected,count", [
    ([0.99, 1.01], 0.0, 0),
    ([0.95, 1.05], 0.0, 0),
    ([0.90, 1.01], 0.05, 1),
    ([0.99, 1.10], 0.05, 1),
])
def test_smartds_daily_voltage_violation_is_nonnegative(monkeypatch, voltages, expected, count):
    """执行实际逐步/汇总记录路径；合规不能记录成负越限幅度。"""
    from types import SimpleNamespace
    import scripts.check_smartds as check

    monkeypatch.setattr(check, "STEPS_PER_DAY", 1)
    state = SimpleNamespace(
        voltages_pu={"bus": dict(enumerate(voltages, 1))}, branches={},
        injections={}, losses=(0.0, 0.0), grid_import=(0.0, 0.0),
    )
    session = SimpleNamespace(
        reset=lambda **kwargs: None, begin_interval=lambda seconds: None,
        solve=lambda **kwargs: None, read_state=lambda: state,
        finish_interval=lambda: None,
    )
    steps, summary = check._run_day(session, SimpleNamespace(apply=lambda *args: None), "fixed")
    assert steps[0]["max_violation_pu"] == pytest.approx(expected)
    assert steps[0]["violating_nodes"] == count
    assert summary["max_violation_pu"] == pytest.approx(expected)
