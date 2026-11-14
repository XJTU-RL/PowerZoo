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
