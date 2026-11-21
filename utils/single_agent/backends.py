"""单智能体训练后端层：集中控制入口（utils/centralized_cli.py）的可插拔算法。

后端契约（签名统一）：
    run(cfg, env, output: Path, steps: int, resume: bool = False)

- `cfg`：ConfigLoader 产物（algo_args 八节结构）；
- `env`：gymnasium 风格环境（cbess 家族为 CBESSTaskEnv，其余家族为
  CentralizedAdapter）；约束 RL 后端从 `info["costs"]`（或
  `info["cost_voltage"]` 单流兼容）提取成本通道；
- `steps`：总环境步数预算（episode 驱动的后端按回合取整）；
- `output`：模型/日志输出前缀，`.environment.yaml` 伴随文件由入口统一写。

新增算法 = 在 BACKENDS 登记一个函数；集中控制算法本体放
`algorithms/centralized/`（CENTRALIZED_ALGOS 注册表）。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from utils.single_agent.costs import extract_costs


# ---------------------------------------------------------------------------
def run_sb3_ppo(cfg, env, output: Path, steps: int, resume: bool = False):
    """SB3 PPO 基线（原集中控制入口行为原样保留；resume 未支持，保持向后兼容）。"""
    from stable_baselines3 import PPO

    if resume:
        raise ValueError("sb3_ppo 后端暂不支持 --resume；用 model.load 恢复评估")
    n_steps = max(2, int(cfg.algo_args["train"].get("episode_length", 240)))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        learning_rate=cfg.algo_args["model"]["lr"] if "lr" in cfg.algo_args["model"] else 3e-4,
        gamma=cfg.algo_args["algo"].get("gamma", 0.99),
        gae_lambda=cfg.algo_args["algo"].get("gae_lambda", 0.95),
        policy_kwargs={"net_arch": cfg.algo_args["model"].get("hidden_sizes", [64, 64])},
        device="cpu",
        seed=cfg.algo_args["seed"]["seed"],
        verbose=1,
    )
    model.learn(total_timesteps=steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    output.with_suffix(".sb3.json").write_text(
        json.dumps(
            {
                "algorithm": "SB3 PPO",
                "steps": model.num_timesteps,
                "n_steps": n_steps,
                "batch_size": n_steps,
                "n_epochs": 1,
                "learning_rate": model.learning_rate,
                "gamma": model.gamma,
                "gae_lambda": model.gae_lambda,
                "device": "cpu",
                "action_adapter": (
                    "MultiDiscrete group choices"
                    if getattr(env, "action_space_type", "continuous") == "discrete"
                    else "Box with discrete-device quantization"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


# ---------------------------------------------------------------------------
def build_ldcpo_config(cfg) -> "object":
    """algo_args 八节 → LdcpoConfig（算法字段严格映射，训练节字段留在循环侧）。"""
    from algorithms.centralized.ldcppo import LdcpoConfig

    model, algo = cfg.algo_args["model"], cfg.algo_args["algo"]
    known = set(LdcpoConfig.__dataclass_fields__)
    values = {}
    for section in (model, algo):
        for key, value in section.items():
            if key in known:
                values[key] = value
    return LdcpoConfig(**values)


def estimate_ldcpo_budgets(env, algo_cfg, episodes: int, T: int) -> dict:
    """零控制基线期望折扣成本（式(28)口径）→ d_k = fraction × J_C_k。"""
    zero = getattr(env, "zero_control_action",
                   np.zeros(env.action_space.shape, dtype=np.float32))
    discount = algo_cfg.gamma ** np.arange(T)
    from algorithms.centralized.ldcppo import COST_KEYS

    returns = {k: [] for k in COST_KEYS}
    for _ in range(episodes):
        env.reset()
        step_costs = {k: np.zeros(T) for k in COST_KEYS}
        for t in range(T):
            _, _, _, _, info = env.step(zero)
            costs = extract_costs(info)
            if costs is None:
                break
            for k in COST_KEYS:
                step_costs[k][t] = costs.get(k, 0.0)
        for k in COST_KEYS:
            returns[k].append(float((step_costs[k] * discount).sum()))
    return {
        k: algo_cfg.cost_budget_fraction.get(k, 0.3) * float(np.mean(returns[k]))
        for k in COST_KEYS
    }


def run_ldcppo(cfg, env, output: Path, steps: int, resume: bool = False):
    """LDCPPO 后端：episode 驱动的约束 RL 训练（复现训练循环的框架化）。"""
    from algorithms.centralized.ldcppo import COST_KEYS, LDCPPO

    algo_cfg = build_ldcpo_config(cfg)
    train_cfg = cfg.algo_args["train"]
    T = int(train_cfg.get("episode_length", 288))
    total_episodes = int(train_cfg.get("total_episodes", steps // max(T, 1)))
    total_episodes = min(total_episodes, max(1, steps // max(T, 1)))

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = LDCPPO(obs_dim, action_dim, algo_cfg, critics="dedicated")

    out_dir = output  # run 目录：checkpoint.pt / train_log.csv / 伴随文件
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    start_episode = 1
    rng = np.random.default_rng(cfg.algo_args["seed"]["seed"])
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(ckpt)
        start_episode = int(ckpt.get("episode", 0)) + 1
        if "np_rng_state" in ckpt:
            rng.bit_generator.state = ckpt["np_rng_state"]
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if "env_rng_state" in ckpt and hasattr(env, "set_rng_state"):
            env.set_rng_state(ckpt["env_rng_state"])
        print(f"续训：自 ep{start_episode} 恢复（λ={agent.multipliers}）")
    else:
        # 仅新训练播种；续训保留 checkpoint 恢复的 RNG 状态（复审 P2）
        torch.manual_seed(cfg.algo_args["seed"]["seed"])
        if hasattr(env, "seed"):
            env.seed(cfg.algo_args["seed"]["seed"])

    budgets = dict(agent.budgets) if agent.budgets else None
    if budgets is None:
        # 约束成本通道探测 + 预算估计（resume 已载入预算时跳过）
        env.reset()
        probe = extract_costs(env.step(env.action_space.sample() * 0)[4])
        env.reset()
        if probe is None:
            print("提示：该环境 info 无约束成本通道（costs/cost_voltage），"
                  "LDCPPO 退化为纯 PPO 更新（λ 将保持 0）")
            budgets = {k: 0.0 for k in COST_KEYS}
        else:
            print("估计零控制预算 d_k …")
            budgets = estimate_ldcpo_budgets(
                env, algo_cfg, int(train_cfg.get("budget_estimate_episodes", 3)), T
            )
        agent.set_budgets(budgets)
    print("预算:", {k: round(v, 4) for k, v in budgets.items()})

    log_path = out_dir / "train_log.csv"
    new_log = start_episode == 1 or not log_path.exists()
    log_file = open(log_path, "a", newline="")
    logger = csv.writer(log_file)
    if new_log:
        logger.writerow(
            ["episode", "reward", *COST_KEYS, *[f"lambda_{k}" for k in COST_KEYS],
             "kl", "seconds"]
        )
    started = time.time()
    checkpoint_every = int(train_cfg.get("checkpoint_every", 100))
    for episode in range(start_episode, total_episodes + 1):
        obs, _ = env.reset()
        states = np.zeros((T, obs_dim), np.float32)
        actions = np.zeros((T, action_dim), np.float32)
        logps = np.zeros(T)
        rewards = np.zeros(T)
        costs = {k: np.zeros(T) for k in COST_KEYS}
        terminated = False
        n_done = 0  # 实际完成转移数（复审 P2：提前终止不传幽灵零转移）
        for t in range(T):
            if terminated:
                break
            states[t] = obs
            a, logp = agent.act(states[t])
            actions[t] = a
            logps[t] = logp
            obs, reward, terminated, _, info = env.step(a)
            rewards[t] = reward
            costs_step = extract_costs(info)
            if costs_step is not None:
                for k in COST_KEYS:
                    costs[k][t] = costs_step.get(k, 0.0)
            n_done = t + 1
        last_values = (
            agent.values(obs.astype(np.float32))
            if algo_cfg.bootstrap_at_episode_end
            else {"R": 0.0, **{k: 0.0 for k in agent.cost_critics}}
        )
        stats = agent.update(
            states[:n_done], actions[:n_done], logps[:n_done],
            rewards[:n_done], {k: v[:n_done] for k, v in costs.items()},
            last_values,
        )
        logger.writerow(
            [episode, round(float(rewards.sum()), 2),
             *[round(float(costs[k].sum()), 3) for k in COST_KEYS],
             *[round(agent.multipliers[k], 4) for k in COST_KEYS],
             round(stats["kl"], 4), round(time.time() - started, 1)]
        )
        if episode % 5 == 0 or episode == 1 or episode == total_episodes:
            print(
                f"ep{episode}: reward={rewards.sum():.1f} "
                f"costs={{{', '.join(f'{k}:{costs[k].sum():.2f}' for k in COST_KEYS)}}} "
                f"λ={{{', '.join(f'{k}:{agent.multipliers[k]:.3f}' for k in COST_KEYS)}}} "
                f"kl={stats['kl']:.4f}"
            )
        if episode % checkpoint_every == 0 or episode == total_episodes:
            ckpt = {**agent.state_dict(),
                    "episode": episode,
                    "backend": "ldcppo",
                    "budgets": budgets,
                    "np_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state()}
            if hasattr(env, "get_rng_state"):
                ckpt["env_rng_state"] = env.get_rng_state()
            torch.save(ckpt, ckpt_path)
    log_file.close()
    out_dir.joinpath("run.ldcppo.json").write_text(
        json.dumps(
            {"algorithm": "LDCPPO", "episodes": total_episodes,
             "budgets": budgets, "lambda": agent.multipliers,
             "steps_per_episode": T},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"完成：日志 {log_path}，checkpoint {ckpt_path}")
    return agent


# 后端注册表（入口按名分发）
BACKENDS = {
    "sb3_ppo": run_sb3_ppo,
    "ldcppo": run_ldcppo,
}
