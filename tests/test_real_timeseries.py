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
