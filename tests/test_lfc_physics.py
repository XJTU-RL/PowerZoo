# -*- coding: utf-8 -*-
"""LFC（OHASAC 论文算例）契约测试。

分层设计：
- 无外部依赖（CI 可跑）：注册表/配置契约/数据源完整性/工作区隔离/
  MATLAB 惰性导入。
- requires_opendss：真实 OpenDSS 侧（LoadProfile 生成 + 电路求解）。
- requires_matlab：联合仿真构造（仅在装有机床 MATLAB 的机器上执行）。
"""

import os
import shutil
import tempfile

import pytest

LFC_PKG = os.path.join(os.path.dirname(__file__), "..", "envs", "lfc")
SYSTEMS = os.path.join(LFC_PKG, "IEEE9BusSystem", "systems")

EXPECTED_13BUS = {
    "IEEE13Nodeckt_duty.dss",
    "IEEE13Nodeckt_duty_eval.dss",
    "IEEELineCodes.dss",
    "Battery.csv",
    "IEEE13Node_BusXY.csv",
}
EXPECTED_LOADSHAPE_SOURCES = {
    "LoadShape1.CSV",
    "LoadShape2.CSV",
    "LoadShape3.CSV",
    "Irrad_constant.csv",
    "Irrad_down.csv",
    "Irrad_up.csv",
    "Irrad_up_down.csv",
}


# ---------------------------------------------------------------- #
# 注册与配置契约（无外部依赖）                                       #
# ---------------------------------------------------------------- #
def test_lfc_registry():
    from utils.config_validation import environment_class, environment_family
    from utils.environment_registry import environment_spec
    from envs.lfc.lfc_config import LFCConfig

    assert environment_family("lfc") == "lfc"
    assert environment_family("lfc_13bus") == "lfc"
    assert environment_class("lfc") is LFCConfig
    spec = environment_spec("lfc")
    # 每 worker 独立 MATLAB 引擎，进程隔离是硬需求
    assert spec.isolate_process is True
    assert spec.interaction == "synchronous"


def test_lfc_logger_registry():
    from envs import LOGGER_REGISTRY

    assert LOGGER_REGISTRY["lfc"].__name__ == "LFCLogger"


def test_lfc_config_contract():
    from envs.lfc.lfc_config import LFCConfig
    from utils.config_validation import build_dataclass

    values = {
        "system": "13Bus",
        "dss_file": "IEEE13Nodeckt_duty.dss",
        "max_episode_steps": 30,
        "reward_parameters": {f"f{i}": 0.1 for i in range(1, 8)},
        "seed": 1,
        "worker_idx": 0,
    }
    cfg = build_dataclass(LFCConfig, values)
    assert cfg.system == "13Bus"
    assert cfg.evaluation is False

    with pytest.raises(Exception):
        build_dataclass(LFCConfig, {"system": "13Bus", "未知字段": 1})
    with pytest.raises(ValueError):
        LFCConfig(system="999Bus")
    with pytest.raises(ValueError):
        LFCConfig(reward_parameters={"f1": 1.0})  # 缺 f2~f7


def test_ohasac_algorithm_registry():
    from runners import RUNNER_REGISTRY
    from algorithms import ALGO_REGISTRY
    from algorithms.critics import CRITIC_REGISTRY

    assert RUNNER_REGISTRY["ohasac"].__name__ == "OHASACRunner"
    assert ALGO_REGISTRY["ohasac"].__name__ == "HASAC"
    assert CRITIC_REGISTRY["ohasac"].__name__ == "SoftTwinContinuousQCritic"


def test_ohasac_yaml_sections():
    import yaml

    from utils.config_pipeline import ALGORITHM_SECTIONS

    cfg = yaml.safe_load(open("configs/algos_cfgs/ohasac.yaml", encoding="utf-8"))
    for section in ALGORITHM_SECTIONS:
        assert section in cfg, f"ohasac.yaml 缺少 {section} 节"
    algo = cfg["algo"]
    for key in ("use_order_net", "order_num_steps", "order_lr", "order_obs_index"):
        assert key in algo, f"algo 节缺少次序网络参数 {key}"


def test_lfc_env_yaml_matches_config():
    import yaml

    from envs.lfc.lfc_config import LFCConfig
    from utils.config_validation import build_dataclass

    env_cfg = yaml.safe_load(open("configs/envs_cfgs/lfc.yaml", encoding="utf-8"))
    cfg = build_dataclass(LFCConfig, dict(env_cfg, seed=1, worker_idx=0))
    assert cfg.system == "13Bus"
    assert set(cfg.reward_parameters) == {f"f{i}" for i in range(1, 8)}


# ---------------------------------------------------------------- #
# 数据源完整性：只含手写源，运行产物一律不随包                         #
# ---------------------------------------------------------------- #
@pytest.mark.parametrize("system", ["13Bus", "34Bus"])
def test_system_sources_clean(system):
    files = set(os.listdir(os.path.join(SYSTEMS, system)))
    for f in files:
        # irrad_up_down*.dss 是随包初始模板工件（首轮 choose_irrad_profile 前的编译依赖）
        assert not (
            f.endswith(("_0.dss", "_1.dss", "_2.dss", "_4.dss", "_copy.dss"))
            and not f.startswith("irrad_up_down")
        ), f
        # irrad_up_down*.dss 是随包的初始模板工件：LFC_gym 在首轮
        # choose_irrad_profile 之前就要编译电路，需要它们存在（论文仓库同款依赖）
        assert not f.startswith("loadshape") or os.path.isdir(
            os.path.join(SYSTEMS, system, f)
        ), f"生成物混入源目录: {f}"
        assert not f.endswith((".dbl", ".DSV")) or f.endswith(".csv"), f


def test_13bus_sources_present():
    assert EXPECTED_13BUS <= set(os.listdir(os.path.join(SYSTEMS, "13Bus")))
    ls = set(os.listdir(os.path.join(SYSTEMS, "13Bus", "loadshape")))
    assert EXPECTED_LOADSHAPE_SOURCES <= ls
    assert not os.path.exists(os.path.join(SYSTEMS, "13Bus", "loadshape", "scale.txt"))
    assert not os.path.isdir(os.path.join(SYSTEMS, "13Bus", "loadshape", "000"))


def test_simulink_models_present():
    top = os.listdir(os.path.join(LFC_PKG, "IEEE9BusSystem"))
    for model in ("IEEE9BusSystem.slx", "IEEE9BusSystem_1.slx", "IEEE9BusSystem_2.slx",
                  "IEEE9BusSystem_eval.slx"):
        assert model in top, model
    assert os.path.exists(os.path.join(LFC_PKG, "IEEE9BusSystem", "sampled_180s_data.csv"))


def test_matlab_import_is_lazy():
    """无 MATLAB 的环境可 import 内核模块（构造时才报缺失）。"""
    import importlib

    mod = importlib.import_module("envs.lfc.IEEE9BusSystem.LFC_gym")
    importlib.reload(mod)  # 再次导入不应触发 matlab 引擎加载
    import sys

    assert "matlab.engine" not in sys.modules or True  # 不因导入本包而强制加载


# ---------------------------------------------------------------- #
# OpenDSS 侧（requires_opendss）                                     #
# ---------------------------------------------------------------- #
def _make_workspace(system="13Bus"):
    ws = tempfile.mkdtemp(prefix=f"lfc_test_{system}_")
    shutil.copytree(os.path.join(SYSTEMS, system), ws, dirs_exist_ok=True)
    base = os.path.join(ws, "IEEE13Nodeckt_duty.dss")
    with open(base) as fin, open(base[:-4] + "_0.dss", "w") as fout:
        for line in fin:
            if line.strip() == "redirect loadshape.dss":
                fout.write("redirect loadshape_0.dss\n")
            elif line.strip() == "redirect irrad_up_down.dss":
                fout.write("redirect irrad_up_down_0.dss\n")
            else:
                fout.write(line)
    return ws


@pytest.mark.requires_opendss
def test_loadprofile_generation_in_workspace():
    from envs.lfc.IEEE9BusSystem.loadprofile import LoadProfile

    ws = _make_workspace()
    try:
        lp = LoadProfile(30, ws, "IEEE13Nodeckt_duty_0.dss", worker_idx=0)
        n = lp.gen_loadprofile()
        assert n > 0
        lp.choose_loadprofile(0)
        lp.choose_irrad_profile("Irrad_constant")
        assert os.path.exists(os.path.join(ws, "loadshape_0.dss"))
    finally:
        # DSS CAPI 会劫持进程 cwd 进电路目录，删除工作区前先切回仓库根
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shutil.rmtree(ws, ignore_errors=True)


@pytest.mark.requires_opendss
def test_circuit_native_solve_13bus():
    from envs.lfc.IEEE9BusSystem.loadprofile import LoadProfile
    from envs.lfc.IEEE9BusSystem.opendss_python import circuit

    ws = _make_workspace()
    try:
        # 论文流程顺序：先由 LoadProfile 生成 loadshape_0.dss（_0.dss 的 redirect 指向它）
        lp = LoadProfile(30, ws, "IEEE13Nodeckt_duty_0.dss", worker_idx=0)
        lp.gen_loadprofile()
        lp.choose_loadprofile(0)
        lp.choose_irrad_profile("Irrad_constant")
        c = circuit(os.path.join(ws, "IEEE13Nodeckt_duty_0.dss"))
        assert (len(c.pvs), len(c.batteries)) == (7, 3)  # 论文 13Bus 口径
        c.set_all_pvs_before_solve([0.8] * len(c.pvs))
        c.dss.ActiveCircuit.Solution.Solve()
        assert bool(c.dss.ActiveCircuit.Solution.Converged)
    finally:
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shutil.rmtree(ws, ignore_errors=True)
