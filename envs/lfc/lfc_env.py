# -*- coding: utf-8 -*-
"""LFC 公共 MARL 入口。

忠实性说明：本包装器逐段取自论文实现的 ``LFC_env.py::LFC_HAEnv``
（P0 修复版），并按 PowerZoo 契约做三处适配（均不改变数值行为）：
1. 构造签名对齐 ``EnvironmentSpec.create``：``LFCEnv(config, rank=rank)``，
   ``LFCConfig`` 字段即论文 ``lfc.yaml`` env 节；
2. 奖励广播为 ``(n_agents, 1) float32`` 团队奖励（论文经 HARL insert 的
   ``np.mean`` 归约后等价，与 PowerZoo DeviceMARLAdapter 同型）；
3. DSS 工作区隔离（P0-7 融合落地）：每个 worker 把 ``systems/<system>``
   源目录复制到临时目录后再运行——论文在源目录原地生成 ``_N.dss`` 与
   loadshape 产物，此机制让源目录永久只读，行为与论文逐位一致。
评估语义（论文 ``make_eval_env`` 的 ranks=4）：``evaluation=True`` 时使用
``*_eval`` Simulink 模型与 ``<stem>_eval.dss``，worker 号取 4+rank。
"""

import os
import shutil
import tempfile

import numpy as np
from gymnasium.spaces import Box

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


class LFCEnv:
    """DN(OpenDSS) + TN(Simulink) 联合仿真的多智能体 LFC 环境。"""

    def __init__(self, config_or_args, rank=None):
        from envs.lfc.lfc_config import LFCConfig

        config = (
            config_or_args
            if isinstance(config_or_args, LFCConfig)
            else LFCConfig(**config_or_args)
        )
        info = {
            "env_name": config.env_name,
            "max_episode_steps": config.max_episode_steps,
            "reward_parameters": config.reward_parameters,
            "long_term_load_disturbance": config.long_term_load_disturbance,
            "use_irrad": config.use_irrad,
            "use_render": config.use_render,
            "irrad_style": config.irrad_style,
            "render_fault": config.render_fault,
            "dss_file": config.dss_file,
        }
        # 论文 make_eval_env 语义：评估用 *_eval 模型与 <stem>_eval.dss，
        # worker 号 4+rank（原实现 ranks=4 固定单评估线程）
        if config.evaluation:
            info["env_name"] = config.env_name + "_eval"
            info["dss_file"] = config.dss_file[:-4] + "_eval.dss"
            worker_idx = 4 + (rank if rank is not None else 0)
        else:
            worker_idx = rank if rank is not None else config.worker_idx

        # P0-7 工作区隔离：源目录只读，生成物全部落在临时副本目录；
        # 生命周期由本实例管理（close/析构时清理）
        source_dir = config.dss_folder_path or _system_dir(config.system)
        self._workspace = _make_workspace(source_dir)
        info["dss_folder_path"] = self._workspace

        from envs.lfc.IEEE9BusSystem.LFC_gym import LFCEnv as CoreLFCEnv

        _restore_matlab_lib_path()

        self.env = CoreLFCEnv(info, worker_idx)

        self.pv_num = self.env.pv_num
        self.bat_num = self.env.bat_num
        self.agents = list(range(self.pv_num + self.bat_num))
        self.n_agents = len(self.agents)
        self.env_name = info["env_name"]

        # 空间契约（与论文 LFC_HAEnv 一致：观测广播、动作逐设备拆分）
        self.share_observation_space = [self.env.observation_space] * self.n_agents
        self.observation_space = [self.env.observation_space] * self.n_agents
        self.action_space = self._agent_action_spaces()
        self.discrete = self.env.action_space.__class__.__name__ != "Box"
        self.avail_actions = None  # 连续动作无可用动作掩码

    # ------------------------------------------------------------------ #
    # MARL 契约（reset 三元组 / step 六元组）                              #
    # ------------------------------------------------------------------ #
    def step(self, actions):
        """动作数组形状 (n_agents, 1)：PV 出力比例 + BESS 充放电。"""
        actions = np.asarray(actions)
        if self.discrete:
            obs, rew, done, _, info = self.env.step(actions.flatten())
        else:
            obs, rew, done, _, info = self.env.step(actions)
        if done and info.get("TimeLimit.truncated", False):
            info["bad_transition"] = True
        rewards = np.full((self.n_agents, 1), float(rew), dtype=np.float32)
        dones = np.full((self.n_agents,), bool(done), dtype=bool)
        # PowerZoo 契约：share_obs 亦为 per-agent 列表（物理重构后的
        # _validate_shapes 强校验要求；DeviceMARLAdapter 同款广播）
        return (
            [obs] * self.n_agents,
            [obs.copy() for _ in range(self.n_agents)],
            rewards,
            dones,
            [info] * self.n_agents,
            None,
        )

    def reset(self):
        obs = self.env.reset()
        return [obs] * self.n_agents, [obs.copy() for _ in range(self.n_agents)], None

    # ------------------------------------------------------------------ #
    # 辅助（论文 LFC_HAEnv 同款语义）                                      #
    # ------------------------------------------------------------------ #
    def _agent_action_spaces(self):
        spaces = [Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)] * self.pv_num
        spaces += [Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)] * self.bat_num
        return spaces

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
        self._cleanup_workspace()

    def seed(self, seed):
        self.env.seed(seed)

    def _cleanup_workspace(self):
        """删除临时工作区。DSS CAPI 编译会把进程 cwd 劫持进电路目录，
        删除前必须先切走，否则 cwd 悬空。"""
        if getattr(self, "_workspace", None) and os.path.isdir(self._workspace):
            safe_cwd = _PKG_DIR if os.path.isdir(_PKG_DIR) else tempfile.gettempdir()
            try:
                if os.path.samefile(os.getcwd(), self._workspace):
                    os.chdir(safe_cwd)
            except OSError:
                os.chdir(safe_cwd)
            shutil.rmtree(self._workspace, ignore_errors=True)
            self._workspace = None

    def __del__(self):
        try:
            self._cleanup_workspace()
        except Exception:  # noqa: BLE001 - 析构期不抛错
            pass


def _system_dir(system):
    path = os.path.join(_PKG_DIR, "IEEE9BusSystem", "systems", system)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"LFC 算例目录不存在：{path}")
    return path


def _make_workspace(source_dir):
    """复制算例源目录到临时工作区（对齐 PowerZoo DSSSession 的只读源哲学）。"""
    workspace = tempfile.mkdtemp(prefix=f"lfc_{os.path.basename(source_dir)}_")
    shutil.copytree(source_dir, workspace, dirs_exist_ok=True)
    return workspace


def _restore_matlab_lib_path():
    """为 MATLAB 引擎净化进程环境（供主/子进程在 spawn 前调用）。

    cv2 被任何链拉入时会：
    1. 重写 LD_LIBRARY_PATH 指向自己的 lib64（毒害库解析）；
    2. 设置 QT_QPA_PLATFORM_PLUGIN_PATH / QT_QPA_FONTDIR 指向自带 Qt——
       MATLAB 本体是 Qt 应用，worker 内引擎起的 MATLAB 会加载这些错误
       插件而崩溃（python 侧表现为 load_system 抛 std::exception）。
    故：LD 复原 + 剥离 QT_*/SDL_* 污染项，使 spawn 的 worker 干净继承。
    """
    for key in list(os.environ):
        if key.startswith(("QT_QPA", "SDL_")):
            del os.environ[key]
    matlab_bin = shutil.which("matlab")
    if not matlab_bin:
        return  # 引擎启动时自会报缺 MATLAB
    glnxa64 = os.path.join(os.path.dirname(os.path.dirname(matlab_bin)), "bin", "glnxa64")
    if not os.path.isdir(glnxa64):
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if glnxa64 not in parts:
        os.environ["LD_LIBRARY_PATH"] = glnxa64 + (os.pathsep + current if current else "")
    elif parts and parts[0] != glnxa64:
        # 已含但被 cv2 路径压前——重排，glnxa64 置首
        parts.remove(glnxa64)
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([glnxa64] + parts)
