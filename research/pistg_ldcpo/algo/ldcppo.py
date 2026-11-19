"""LDCPPO：拉格朗日专用 critic PPO（论文式(27)-(36)）。

- 1 个奖励 critic + 3 个约束 critic（VD/VN/LO），独立 GAE(λ=0.95)、
  ζ 批内标准化（式(31)）、值回归目标（式(32)(33)）；
- 演员更新用 clipped surrogate（式(35)），拉格朗日优势
  A^L = A^R − Σ λ_k A^k（式(35c)）；
- 乘子对偶上升 λ_k ← [λ_k + η_Lag(Ĵ_C_k − d_k)]₊（式(36d)），
  Ĵ_C_k 为 minibatch 内回合折扣成本回报均值（式(28)）；
- target KL 提前停、梯度裁剪 0.5、零熵、value 系数 0.5（论文 p.10）。

critics="mixed" 时退化为单 critic 混合惩罚（论文未含此变体，供消融）；
state_mode ∈ {pistg, stg, flat} 决定状态表示（对应 PISTG-LDCPPO /
STG-LDCPPO / LDCPPO 三配置）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import LdcpoConfig
from .networks import Critic, GaussianActor

COST_KEYS = ("VD", "VN", "LO")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
):
    """单流 GAE（式(30)(31) 的未标准化形式）。"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_value = values[t]
    return adv


def standardize(x: np.ndarray) -> np.ndarray:
    """式(31) 的 ζ：批内零均值单位方差。"""
    std = x.std() + 1e-8
    return (x - x.mean()) / std


class LDCPPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: LdcpoConfig,
        critics: str = "dedicated",  # dedicated | mixed
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = device
        self.critics_mode = critics
        self.actor = GaussianActor(
            state_dim, action_dim, cfg.actor_hidden, cfg.log_std_init
        ).to(device)
        self.reward_critic = Critic(state_dim, cfg.critic_hidden).to(device)
        if critics == "dedicated":
            self.cost_critics = {
                k: Critic(state_dim, cfg.critic_hidden).to(device) for k in COST_KEYS
            }
            self.multipliers = {k: float(cfg.init_multipliers) for k in COST_KEYS}
            self.budgets: dict[str, float] = {}
        elif critics == "mixed":
            self.cost_critics = {
                "MIXED": Critic(state_dim, cfg.critic_hidden).to(device)
            }
            self.multipliers = {"MIXED": float(cfg.init_multipliers)}
            self.budgets = {}
        else:
            raise ValueError(f"未知 critics 模式：{critics}")
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), cfg.actor_lr)
        self.reward_opt = torch.optim.Adam(
            self.reward_critic.parameters(), cfg.critic_lr
        )
        self.cost_opts = {
            k: torch.optim.Adam(c.parameters(), cfg.critic_lr)
            for k, c in self.cost_critics.items()
        }

    # ------------------------------------------------------------------
    def set_budgets(self, budgets: dict[str, float]):
        self.budgets = dict(budgets)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        normal = self.actor.distribution(s)
        action = normal.mean if deterministic else normal.sample()
        logp = normal.log_prob(action).sum(-1)
        return (
            action.squeeze(0).cpu().numpy(),
            float(logp),
        )

    @torch.no_grad()
    def values(self, state: np.ndarray) -> dict[str, float]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        result = {"R": float(self.reward_critic(s))}
        for k, critic in self.cost_critics.items():
            result[k] = float(critic(s))
        return result

    # ------------------------------------------------------------------
    def update(
        self,
        states: np.ndarray,  # (T, state_dim)
        actions: np.ndarray,  # (T, action_dim)
        old_logps: np.ndarray,  # (T,)
        rewards: np.ndarray,  # (T,)
        costs: dict[str, np.ndarray],  # 每 key (T,)
        last_values: dict[str, float],  # V(s_T) 各流
    ) -> dict[str, float]:
        """单回合数据上的 PPO 更新（式(33)-(36)）。"""
        cfg = self.cfg
        T = len(rewards)
        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(old_logps, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            v_r = np.array(
                [float(self.reward_critic(s[i : i + 1])) for i in range(T)]
            )
            v_costs = {
                k: np.array([float(critic(s[i : i + 1])) for i in range(T)])
                for k, critic in self.cost_critics.items()
            }

        # 式(31)：GAE + ζ 标准化。
        adv_r = standardize(
            compute_gae(rewards, v_r, last_values["R"], cfg.gamma, cfg.gae_lambda)
        )
        adv_costs = {}
        for k, critic_keys in self._cost_stream_keys(costs).items():
            stream = self._aggregate_stream(costs, critic_keys)
            adv_costs[k] = standardize(
                compute_gae(
                    stream, v_costs[k], last_values[k], cfg.gamma, cfg.gae_lambda
                )
            )
        # 式(32)：值回归目标 = 未标准化 GAE + V（ζ 只作用于优势，不作用目标）。
        target_r = compute_gae(
            rewards, v_r, last_values["R"], cfg.gamma, cfg.gae_lambda
        ) + v_r
        target_costs = {}
        for k, critic_keys in self._cost_stream_keys(costs).items():
            stream = self._aggregate_stream(costs, critic_keys)
            target_costs[k] = (
                compute_gae(stream, v_costs[k], last_values[k],
                            cfg.gamma, cfg.gae_lambda)
                + v_costs[k]
            )

        stats = {"kl": 0.0, "actor_loss": 0.0, "critic_loss": 0.0,
                 "entropy": 0.0}
        stop = False
        for _ in range(cfg.ppo_epochs):
            if stop:
                break
            # --- 演员（式(35)）---
            normal = self.actor.distribution(s)
            logp = normal.log_prob(a).sum(-1)
            ratio = (logp - old_logp).exp()
            adv_lag = torch.tensor(adv_r, dtype=torch.float32, device=self.device)
            for k, lam in self.multipliers.items():
                adv_lag = adv_lag - lam * torch.tensor(
                    adv_costs[k], dtype=torch.float32, device=self.device
                )
            surrogate1 = ratio * adv_lag
            surrogate2 = (
                torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
                * adv_lag
            )
            actor_loss = -torch.minimum(surrogate1, surrogate2).mean()
            with torch.no_grad():
                kl = float(
                    (old_logp - logp).mean()
                )  # k1 近似（论文未指定 KL 估计器）
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           cfg.max_grad_norm)
            self.actor_opt.step()
            stats["actor_loss"] = float(actor_loss)
            stats["kl"] = kl
            if kl > cfg.target_kl:
                stop = True
            # --- 评论家（式(33)）---
            critic_loss = F.mse_loss(
                self.reward_critic(s), torch.tensor(target_r, dtype=torch.float32, device=self.device)
            )
            self.reward_opt.zero_grad()
            (cfg.value_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.reward_critic.parameters(), cfg.max_grad_norm
            )
            self.reward_opt.step()
            for k, critic in self.cost_critics.items():
                loss_k = F.mse_loss(
                    critic(s), torch.tensor(target_costs[k], dtype=torch.float32, device=self.device)
                )
                self.cost_opts[k].zero_grad()
                (cfg.value_coef * loss_k).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                self.cost_opts[k].step()
                critic_loss = critic_loss + loss_k
            stats["critic_loss"] = float(critic_loss)
            stats["entropy"] = float(normal.entropy().sum(-1).mean())

        # --- 乘子对偶上升（式(36d)）---
        discount = cfg.gamma ** np.arange(T)
        for k in self.multipliers:
            stream = self._aggregate_stream(costs, self._cost_stream_keys(costs)[k])
            j_hat = float((stream * discount).sum())
            self.multipliers[k] = max(
                0.0, self.multipliers[k] + cfg.dual_lr * (j_hat - self.budgets.get(k, 0.0))
            )
        stats.update({f"lambda_{k}": v for k, v in self.multipliers.items()})
        return stats

    # ------------------------------------------------------------------
    def _cost_stream_keys(self, costs: dict[str, np.ndarray]) -> dict[str, tuple]:
        """dedicated：三流独立；mixed：三流加权求和为单流（等权）。"""
        if self.critics_mode == "dedicated":
            return {k: (k,) for k in COST_KEYS}
        return {"MIXED": tuple(COST_KEYS)}

    @staticmethod
    def _aggregate_stream(costs, keys) -> np.ndarray:
        stream = np.zeros(len(next(iter(costs.values()))))
        for k in keys:
            stream = stream + costs[k]
        return stream
