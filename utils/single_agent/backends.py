"""单智能体后端，由集中控制入口分发。"""

from __future__ import annotations

import csv
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from gymnasium import spaces

from utils.single_agent.binding import (
    action_semantics,
    atomic_write,
    checkpoint_path,
    save_bound_sb3,
    validate_metadata,
    write_binding,
)
from utils.single_agent.costs import COST_KEYS, extract_costs

SB3_CLASSES = {"sb3_ppo": "PPO", "sb3_sac": "SAC", "sb3_td3": "TD3"}


def validate_backend(backend, family, env=None):
    if backend == "ldcppo" and family != "cbess":
        raise ValueError("LDCPPO 需要 CBESS 连续动作及 VD/VN/LO 全部约束成本")
    if backend in ("sb3_sac", "sb3_td3") and family == "dsr":
        raise ValueError(f"{backend} 不支持 DSR 离散动作")
    if env is not None and backend != "sb3_ppo":
        space = env.action_space
        if (
            action_semantics(env) != "continuous_box"
            or len(space.shape) != 1
            or space.shape[0] < 1
            or not np.isfinite(space.low).all()
            or not np.isfinite(space.high).all()
            or not (space.high > space.low).all()
        ):
            raise ValueError(f"{backend} 需要有界连续向量动作，不支持离散或量化设备")
        if backend == "ldcppo" and (
            not isinstance(env.observation_space, spaces.Box)
            or len(env.observation_space.shape) != 1
        ):
            raise ValueError("LDCPPO 需要向量 Box 观测")


def sb3_class(backend):
    import stable_baselines3

    return getattr(stable_baselines3, SB3_CLASSES[backend])


def run_sb3(backend, cfg, env, output: Path, steps: int, resume=False, *, binding):
    if resume:
        raise ValueError(f"{backend} 不支持 --resume")
    validate_backend(backend, cfg.main_args["env"], env)
    train, algo, model_cfg = (cfg.algo_args[key] for key in ("train", "algo", "model"))
    if backend == "sb3_ppo":
        n_steps = max(2, int(train.get("episode_length", 240)))
        options = {
            "n_steps": n_steps,
            "batch_size": n_steps,
            "n_epochs": 1,
            "gae_lambda": algo.get("gae_lambda", 0.95),
        }
    else:
        options = {
            key: algo[key]
            for key in ("buffer_size", "learning_starts", "batch_size")
            if key in algo
        }
    model = sb3_class(backend)(
        "MlpPolicy",
        env,
        **options,
        learning_rate=model_cfg.get("lr", 3e-4),
        gamma=algo.get("gamma", 0.99),
        policy_kwargs={"net_arch": model_cfg.get("hidden_sizes", [64, 64])},
        device="cpu",
        seed=cfg.algo_args["seed"]["seed"],
        verbose=1,
    )
    model.learn(total_timesteps=steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(output, backend)

    atomic_write(checkpoint, lambda path: save_bound_sb3(path, model))
    write_binding(checkpoint, binding)
    semantics = action_semantics(env)
    output.with_suffix(".sb3.json").write_text(
        json.dumps(
            {
                "algorithm": f"SB3 {SB3_CLASSES[backend]}",
                "steps": model.num_timesteps,
                **options,
                "learning_rate": model.learning_rate,
                "gamma": model.gamma,
                "device": "cpu",
                "action_adapter": {
                    "discrete": "MultiDiscrete group choices",
                    "quantized_box": "Box with discrete-device quantization",
                    "continuous_box": "Box with continuous devices",
                }[semantics],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def build_ldcpo_config(cfg):
    from algorithms.centralized.ldcppo import LdcpoConfig

    known = set(LdcpoConfig.__dataclass_fields__)
    values = {
        key: value
        for section in ("model", "algo")
        for key, value in cfg.algo_args[section].items()
        if key in known
    }
    return LdcpoConfig(**values)


def estimate_ldcpo_budgets(env, algo_cfg, episodes: int, horizon: int) -> dict:
    """折扣零控制成本，按实际完成的转移累加。"""
    totals = dict.fromkeys(COST_KEYS, 0.0)
    for _ in range(episodes):
        env.reset()
        for t in range(horizon):
            _, _, terminated, truncated, info = env.step(env.zero_control_action.copy())
            for key, cost in extract_costs(info).items():
                totals[key] += algo_cfg.gamma**t * cost
            if terminated or truncated:
                break
    return {
        key: algo_cfg.cost_budget_fraction[key] * total / episodes
        for key, total in totals.items()
    }


def collect_ldcppo_rollout(env, agent, horizon):
    obs, _ = env.reset()
    states, actions, logps, rewards = [], [], [], []
    costs = {key: [] for key in COST_KEYS}
    for _ in range(horizon):
        states.append(np.asarray(obs, dtype=np.float32).copy())
        action, logp = agent.act(states[-1])
        # PPO 的对数概率属于采样动作。环境在控制边界投影后执行。
        actions.append(action.copy())
        logps.append(logp)
        obs, reward, terminated, truncated, info = env.step(
            np.clip(action, env.action_space.low, env.action_space.high)
        )
        rewards.append(reward)
        for key, value in extract_costs(info).items():
            costs[key].append(value)
        if terminated or truncated:
            break
    if not terminated and agent.cfg.bootstrap_at_episode_end:
        last_values = agent.values(np.asarray(obs, dtype=np.float32))
    else:
        last_values = dict.fromkeys(("R", *COST_KEYS), 0.0)
    return (
        np.asarray(states),
        np.asarray(actions),
        np.asarray(logps),
        np.asarray(rewards),
        {key: np.asarray(values) for key, values in costs.items()},
        last_values,
    )


def make_ldcppo(cfg, env):
    from algorithms.centralized.ldcppo import LDCPPO

    return LDCPPO(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        build_ldcpo_config(cfg),
        critics="dedicated",
    )


def load_ldcppo_state(checkpoint, binding, *, resume=False, eval_scenario_sha256=None):
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("backend") != "ldcppo":
        raise ValueError("checkpoint 内容与 LDCPPO 后端不匹配")
    validate_metadata(
        state.get("binding"),
        binding,
        resume=resume,
        eval_scenario_sha256=eval_scenario_sha256,
    )
    return state


def run_ldcppo(cfg, env, output: Path, steps: int, resume=False, *, binding):
    import torch

    validate_backend("ldcppo", cfg.main_args["env"], env)
    train = cfg.algo_args["train"]
    horizon = min(int(train.get("episode_length", 288)), env.config.steps_per_day)
    budget_episodes = int(train.get("budget_estimate_episodes", 3))
    checkpoint_every = int(train.get("checkpoint_every", 100))
    target_episodes = int(train.get("total_episodes", steps))
    if min(horizon, budget_episodes, checkpoint_every, target_episodes) < 1:
        raise ValueError("LDCPPO 回合长度、总回合数、预算回合数、保存间隔必须为正")

    checkpoint = checkpoint_path(output, "ldcppo")
    if resume:
        ckpt = load_ldcppo_state(checkpoint, binding, resume=True)
    else:
        if checkpoint.exists():
            raise ValueError(
                f"checkpoint 已存在：{checkpoint}；请使用 --resume 或指定新输出目录"
            )
        seed = cfg.algo_args["seed"]["seed"]
        torch.manual_seed(seed)
        env.seed(seed)

    agent = make_ldcppo(cfg, env)
    episode = consumed = 0
    if resume:
        agent.load_state_dict(ckpt)
        episode, consumed = ckpt["episode"], ckpt["steps"]
        torch.set_rng_state(ckpt["torch_rng_state"])
        env.set_rng_state(ckpt["env_rng_state"])
        if consumed >= steps or episode >= target_episodes:
            raise ValueError("--resume 需要更大的总步数及足够的总回合数")
    else:
        agent.set_budgets(
            estimate_ldcpo_budgets(env, agent.cfg, budget_episodes, horizon)
        )

    out_dir = checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    fieldnames = [
        "episode",
        "reward",
        *COST_KEYS,
        *[f"lambda_{key}" for key in COST_KEYS],
        "kl",
        "seconds",
    ]
    # 若中断时日志比上一次 checkpoint 更新，续训从已提交的回合继续。
    rows = []
    if resume and log_path.is_file():
        with log_path.open(newline="") as stream:
            rows = [
                row for row in csv.DictReader(stream) if int(row["episode"]) <= episode
            ]
    started = time.monotonic()
    with log_path.open("w", newline="") as stream:
        logger = csv.DictWriter(stream, fieldnames)
        logger.writeheader()
        logger.writerows(rows)
        while episode < target_episodes and consumed < steps:
            rollout = collect_ldcppo_rollout(env, agent, min(horizon, steps - consumed))
            stats = agent.update(*rollout)
            rewards, costs = rollout[3:5]
            episode += 1
            consumed += len(rewards)
            logger.writerow(
                {
                    "episode": episode,
                    "reward": round(float(rewards.sum()), 2),
                    **{key: round(float(costs[key].sum()), 3) for key in COST_KEYS},
                    **{
                        f"lambda_{key}": round(agent.multipliers[key], 4)
                        for key in COST_KEYS
                    },
                    "kl": round(stats["kl"], 4),
                    "seconds": round(time.monotonic() - started, 1),
                }
            )
            if (
                episode % checkpoint_every == 0
                or episode == target_episodes
                or consumed == steps
            ):
                stream.flush()
                state = {
                    **agent.state_dict(),
                    "backend": "ldcppo",
                    "episode": episode,
                    "binding": binding,
                    "steps": consumed,
                    "torch_rng_state": torch.get_rng_state(),
                    "env_rng_state": env.get_rng_state(),
                }
                atomic_write(
                    checkpoint, lambda path, state=state: torch.save(state, path)
                )
    (out_dir / "run.ldcppo.json").write_text(
        json.dumps(
            {
                "algorithm": "LDCPPO",
                "episodes": episode,
                "steps": consumed,
                "budgets": agent.budgets,
                "lambda": agent.multipliers,
                "steps_per_episode": horizon,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"完成：日志 {log_path}，checkpoint {checkpoint}")
    return agent


BACKENDS = {
    **{name: partial(run_sb3, name) for name in SB3_CLASSES},
    "ldcppo": run_ldcppo,
}
