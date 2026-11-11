# -*- coding: utf-8 -*-
"""LFC（OHASAC 论文算例）安装与运行自检。

分层：
1. 注册与配置契约（无外部依赖）
2. 数据源完整性（无外部依赖）
3. OpenDSS 侧（dss 包）：LoadProfile 生成 + 13/34Bus 电路求解
4. MATLAB 引擎侧（可选，--matlab 或环境可用时）：联合仿真构造 + 1 步步进
   —— 需要 MATLAB R2024a~R2025b（含 Simscape Electrical / SPS）与
   DISPLAY（Xvfb 即可，Simulink 的 set_param 仿真命令需要显示会话）。

用法：
    python scripts/check_lfc.py            # 1-3 层
    python scripts/check_lfc.py --matlab   # 全部四层
"""

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = {}


def ok(name, detail=""):
    RESULTS[name] = (True, detail)
    print(f"[PASS] {name} {detail}")


def fail(name, detail):
    RESULTS[name] = (False, detail)
    print(f"[FAIL] {name} {detail}")


def check_registry():
    from utils.config_validation import environment_class, environment_family
    from envs.lfc.lfc_config import LFCConfig

    assert environment_family("lfc") == "lfc"
    assert environment_class("lfc") is LFCConfig
    from runners import RUNNER_REGISTRY

    assert "ohasac" in RUNNER_REGISTRY
    ok("registry", "lfc 家族 + ohasac 算法注册")


def check_config():
    import yaml

    from utils.config_pipeline import ALGORITHM_SECTIONS
    from envs.lfc.lfc_config import LFCConfig
    from utils.config_validation import build_dataclass

    cfg = yaml.safe_load(open("configs/algos_cfgs/ohasac.yaml", encoding="utf-8"))
    missing = [s for s in ALGORITHM_SECTIONS if s not in cfg]
    assert not missing, f"ohasac.yaml 缺节 {missing}"
    env_cfg = yaml.safe_load(open("configs/envs_cfgs/lfc.yaml", encoding="utf-8"))
    build_dataclass(LFCConfig, dict(env_cfg, seed=1, worker_idx=0))
    ok("config", "八节契约 + env dataclass 校验通过")


def check_sources():
    root = os.path.join("envs", "lfc", "IEEE9BusSystem", "systems")
    for system in ("13Bus", "34Bus"):
        files = os.listdir(os.path.join(root, system))
        polluted = [
            f
            for f in files
            if (
                f.endswith(("_0.dss", "_1.dss", "_2.dss", "_4.dss", "_copy.dss", ".dbl", ".DSV"))
                and not f.startswith("irrad_up_down")  # 初始模板工件（首轮 reset 前的编译依赖）
            )
            or (f.startswith("loadshape") and not os.path.isdir(os.path.join(root, system, f)))
        ]
        assert not polluted, f"{system} 源目录混入生成物: {polluted}"
    ok("sources", "13Bus/34Bus 源目录干净（生成物仅存在于运行工作区）")


def check_opendss():
    import numpy as np

    from envs.lfc.IEEE9BusSystem.loadprofile import LoadProfile
    from envs.lfc.IEEE9BusSystem.opendss_python import circuit

    ws = tempfile.mkdtemp(prefix="lfc_check_")
    try:
        shutil.copytree(
            os.path.join("envs", "lfc", "IEEE9BusSystem", "systems", "13Bus"),
            ws,
            dirs_exist_ok=True,
        )
        base = os.path.join(ws, "IEEE13Nodeckt_duty.dss")
        with open(base) as fin, open(base[:-4] + "_0.dss", "w") as fout:
            for line in fin:
                if line.strip() == "redirect loadshape.dss":
                    fout.write("redirect loadshape_0.dss\n")
                elif line.strip() == "redirect irrad_up_down.dss":
                    fout.write("redirect irrad_up_down_0.dss\n")
                else:
                    fout.write(line)
        lp = LoadProfile(30, ws, "IEEE13Nodeckt_duty_0.dss", worker_idx=0)
        n = lp.gen_loadprofile()
        lp.choose_loadprofile(0)
        lp.choose_irrad_profile("Irrad_constant")
        c = circuit(os.path.join(ws, "IEEE13Nodeckt_duty_0.dss"))
        c.set_all_pvs_before_solve([0.8] * len(c.pvs))
        c.dss.ActiveCircuit.Solution.Solve()
        assert bool(c.dss.ActiveCircuit.Solution.Converged)
        ok(
            "opendss",
            f"负荷曲线 {n} 条；13Bus {len(c.pvs)} PV + {len(c.batteries)} BESS 求解收敛",
        )
    finally:
        # DSS CAPI 编译会把进程 cwd 劫持进电路目录（论文内核行为）；
        # 删除工作区前必须先切回安全目录，否则后续 abspath/getcwd 悬空报错
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
        shutil.rmtree(ws, ignore_errors=True)


def check_matlab():
    """子进程执行：主进程经 runners 链会顶层拉入旧版 gym（qmix→agent_q_function），
    旧 gym 0.26 与 matlabengine 的 IPC 冲突（load_system 抛 std::exception，
    实测同 gymnasium 0.29 同族问题）。正式训练路径不受影响——LFC worker 由
    ShareSubprocVecEnv 隔离子进程，仅导入 envs.lfc 链（gymnasium 1.1.1 无毒）。
    此处以子进程复刻该拓扑做联合仿真检查。"""
    import subprocess
    import sys as _sys

    code = (
        "from envs.lfc.lfc_config import LFCConfig;"
        "from envs.lfc.lfc_env import LFCEnv;"
        "import numpy as np;"
        "cfg = LFCConfig(system='13Bus', dss_file='IEEE13Nodeckt_duty.dss', max_episode_steps=6);"
        "env = LFCEnv(cfg, rank=0);"
        "obs, share, avail = env.reset();"
        "actions = np.array([[0.5]] * env.pv_num + [[0.0]] * env.bat_num);"
        "o, s, r, d, i, a = env.step(actions);"
        "env.close();"
        "assert len(obs) == env.n_agents and len(r) == env.n_agents;"
        "print(f'MATLAB_SUBPROC_OK {env.n_agents} {env.pv_num} {env.bat_num} {float(r[0][0]):.4f}')"
    )
    # 父进程若已加载 cv2，其 __init__ 会重写 LD_LIBRARY_PATH/QT_* 等；
    # 子进程显式传净化后的环境（复原 MATLAB 库路径），不继承被污染版本
    import shutil as _sh

    _mb = _sh.which("matlab")
    _glnxa64 = (
        os.path.join(os.path.dirname(os.path.dirname(_mb)), "bin", "glnxa64")
        if _mb
        else None
    )
    _env = {k: v for k, v in os.environ.items() if not k.startswith(("QT_QPA", "SDL_"))}
    if _glnxa64 and os.path.isdir(_glnxa64):
        _env["LD_LIBRARY_PATH"] = _glnxa64 + os.pathsep + _env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, env=_env
    )
    out = (proc.stdout + proc.stderr).strip().splitlines()
    marker = [ln for ln in out if ln.startswith("MATLAB_SUBPROC_OK")]
    if proc.returncode != 0 or not marker:
        raise RuntimeError("matlab 子进程检查失败：\n" + "\n".join(out[-8:]))
    _, n_agents, pv, bat, rew = marker[-1].split()
    ok(
        "matlab",
        f"联合仿真（子进程隔离，同真实 worker 拓扑）：{n_agents} 智能体"
        f"（{pv} PV + {bat} BESS），reward={rew}",
    )


def _check_matlab_direct():
    from envs.lfc.lfc_config import LFCConfig
    from envs.lfc.lfc_env import LFCEnv

    cfg = LFCConfig(system="13Bus", dss_file="IEEE13Nodeckt_duty.dss", max_episode_steps=6)
    env = LFCEnv(cfg, rank=0)
    import numpy as np

    obs, share, avail = env.reset()
    actions = np.array([[0.5]] * env.pv_num + [[0.0]] * env.bat_num)
    o, s, r, d, i, a = env.step(actions)
    env.close()
    assert len(obs) == env.n_agents and len(r) == env.n_agents
    ok(
        "matlab",
        f"联合仿真构造+reset+step OK：{env.n_agents} 智能体（{env.pv_num} PV + {env.bat_num} BESS），"
        f"reward={float(r[0][0]):.4f}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", action="store_true", help="追加 MATLAB 联合仿真层检查")
    args = parser.parse_args()

    import traceback

    for fn in (check_registry, check_config, check_sources, check_opendss):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            fail(fn.__name__, repr(exc))
    if args.matlab or os.environ.get("LFC_CHECK_MATLAB") == "1":
        try:
            check_matlab()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            fail("matlab", repr(exc))
    else:
        print("[SKIP] matlab（--matlab 启用；需 MATLAB R2024a~R2025b + DISPLAY）")

    failed = [k for k, (passed, _) in RESULTS.items() if not passed]
    print("LFC_CHECK_RESULT=" + ("OK" if not failed else f"FAILED:{failed}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
