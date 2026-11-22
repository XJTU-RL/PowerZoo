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
from .profile_store import ProfileStore

REPO_ROOT = Path(__file__).resolve().parents[2]


class CBESSTaskEnv(gym.Env):
    """CBESEnv 的单智能体任务包装（注册表 wrapper_class）。

    继承 gymnasium.Env（复审 P2）：SB3 等后端要求真 Env 子类，
    普通 Python 类会在 _patch_env 处被拒。"""

    def __init__(self, config: CBESSConfig, rank: int = 0):
        if config.split not in ("train", "validation"):
            raise ValueError("CBESS 单智能体任务只支持 train / validation")
        if config.step_seconds != 300:
            raise ValueError("CBESS 剖面和能量更新要求 300 秒步长")
        self.config = config
        self.split = config.split
        case_folder = REPO_ROOT / "node_systems" / config.system
        data_dir = Path(config.data_dir)
        if not data_dir.is_absolute():
            data_dir = REPO_ROOT / data_dir
        store = ProfileStore(data_dir / self.split, split=self.split)
        self.scenario_sha256 = store.fingerprint()
        # config 即场景配置（research 侧以 ScenarioConfig 别名复用）。
        self.env = CBESEnv(case_folder, store, config)
        if list(config.cbess_buses) != self.env.cbess_buses:
            raise ValueError("CBESS 配置母线顺序与算例不一致")
        days = len(store.day_keys)
        steps = (
            store.price_aud_per_mwh.shape[1] if store.price_aud_per_mwh.ndim == 2 else 0
        )
        expected_shapes = (
            (store.p_load_kw, (days, steps, self.env.n_bus)),
            (store.q_load_kvar, (days, steps, self.env.n_bus)),
            (store.pv_available_kw, (days, steps, len(self.env.der_buses))),
            (store.price_aud_per_mwh, (days, steps)),
        )
        if not 0 < config.steps_per_day <= steps or any(
            array.shape != shape or not np.isfinite(array).all()
            for array, shape in expected_shapes
        ):
            raise ValueError("CBESS 剖面维度、日内长度或数值无效")
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
        zero[
            2 * len(self.env.cbess_buses) : 2 * len(self.env.cbess_buses)
            + len(self.env.der_buses)
        ] = -1.0
        self.zero_control_action = zero

    # ------------------------------------------------------------------
    def seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def get_rng_state(self) -> dict:
        return self._rng.bit_generator.state

    def set_rng_state(self, state: dict):
        self._rng.bit_generator.state = state

    def get_physics_manifest(self):
        observation = [
            f"bus.{bus}.{field}"
            for bus in range(1, self.env.n_bus + 1)
            for field in (
                "load_kw",
                "load_kvar",
                "pv_available_kw",
                "voltage_pu",
                "energy_kwh",
            )
        ] + ["price_aud_per_mwh"]
        action = [
            f"{device}.{bus}.{field}"
            for device, buses, fields in (
                (
                    "cbess",
                    self.env.cbess_buses,
                    ("active_fraction", "reactive_fraction"),
                ),
                (
                    "pv",
                    self.env.der_buses,
                    ("curtailment_fraction", "reactive_fraction"),
                ),
            )
            for field in fields
            for bus in buses
        ]
        return {
            **self.env._ensure_session().manifest(),
            "task": "cbess",
            "observation_schema": observation,
            "action_schema": action,
            "pv_s_max_kva": self.env.pv_s_max_kva.tolist(),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs = self.env.reset(self._rng, self.split)
        return self._flatten(obs), {
            "split": self.split,
            "day_key": self.env.profile.day_key,
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, reward, costs, done, info = self.env.step(action)
        info = dict(info)
        info["costs"] = costs
        # 日内调度结束后独立重采样并重置 SOC，因此自然日末是终止状态。
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
        flat = np.concatenate([obs["bus_features"].ravel(), [np.float32(obs["price"])]])
        return flat.astype(np.float32)
