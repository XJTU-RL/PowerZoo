"""注册表包装器：CBESEnv 的 gymnasium 风格单智能体接口。

供集中控制/单智能体入口（`utils/centralized_cli.py --backend`）消费：
- `reset() -> (obs_166d, info)`；`step() -> 5 元组`，约束成本在
  `info["costs"] = {"VD", "VN", "LO"}`（算法侧拉格朗日的 cost 通道）；
- `zero_control_action` 属性提供零控制动作（κ=0），供预算估计等
  基线 rollout 使用；
- 母线特征/电价平铺为单步向量（W=12 窗口由研究侧编码器组装）。

注：本家族当前不提供 MARL 适配接口（`_vector_env` 有显式拦截）；
MARL 化属于后续研究线（见 docs/pistg-repro-results.md 与开发计划）。
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from .cbes_env import CBESEnv
from .config import CBESSConfig
from .physics import COST_KEYS
from .profile_store import ProfileStore

REPO_ROOT = Path(__file__).resolve().parents[2]


class CBESSTaskEnv:
    """CBESEnv 的单智能体任务包装（注册表 wrapper_class）。"""

    def __init__(self, config: CBESSConfig, rank: int = 0):
        self.config = config
        case_folder = REPO_ROOT / "node_systems" / config.system
        data_dir = Path(config.data_dir)
        if not data_dir.is_absolute():
            data_dir = REPO_ROOT / data_dir
        store = ProfileStore(data_dir)
        # config 即场景配置（research 侧以 ScenarioConfig 别名复用）。
        self.env = CBESEnv(case_folder, store, config)
        self._rng = np.random.default_rng(config.seed)

        obs_dim = self.env.n_bus * 5 + 1
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (self.env.action_dim,), dtype=np.float32
        )
        # 零控制：CBESS 静默 + κ=0（估计预算/零控制基线用）。
        zero = np.zeros(self.env.action_dim, dtype=np.float32)
        zero[2 * len(self.env.cbess_buses) : 2 * len(self.env.cbess_buses)
             + len(self.env.der_buses)] = -1.0
        self.zero_control_action = zero

    # ------------------------------------------------------------------
    def seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs = self.env.reset(self._rng, "train")
        return self._flatten(obs), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, reward, costs, done, info = self.env.step(action)
        info = dict(info)
        info["costs"] = {k: float(costs[k]) for k in COST_KEYS}
        return (
            self._flatten(obs),
            float(reward),
            bool(done),
            False,
            info,
        )

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    def _flatten(self, obs: dict) -> np.ndarray:
        flat = np.concatenate(
            [obs["bus_features"].ravel(), [np.float32(obs["price"])]]
        )
        return flat.astype(np.float32)
