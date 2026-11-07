"""SB3独立PPO的顺序采样。每个主体独立策略/价值/优化器/GAE buffer。

算法校验保持纯Python；Torch/SB3/物理引擎仅在真正训练或恢复时导入。
这只是一般和顺序博弈的学习基线，不包含双层梯度或均衡收敛保证。
"""

from pathlib import Path
import copy
import json
import math
import time


def validate_settings(config):
    from utils.config_validation import ConfigError

    a = config.algo_args
    tr, alg = a["train"], a["algo"]
    if tr["n_rollout_threads"] != 1 or a["eval"]["n_eval_rollout_threads"] != 1:
        raise ConfigError("sequential_ppo首版只支持单采样进程；物理context仍独立")
    if a["render"]["use_render"]:
        raise ConfigError(
            "sequential_ppo使用实际checkpoint无头回放，未接入旧render模式"
        )
    if (
        tr["episode_length"] != config.env_args["max_episode_steps"]
        or tr["episode_length"] < 2
        or tr["num_env_steps"] % tr["episode_length"]
    ):
        raise ConfigError(
            "sequential_ppo按完整有限回合采样：episode_length须等于环境horizon且>=2，num_env_steps须整除"
        )
    if (
        not 2 <= alg["batch_size"] <= tr["episode_length"]
        or tr["episode_length"] % alg["batch_size"]
    ):
        raise ConfigError("algo.batch_size须>=2且整除episode_length")
    if alg["ppo_epoch"] < 1 or not 0 < alg["clip_param"] < 1:
        raise ConfigError("非法PPO轮数/裁剪范围")
    if not 0 <= alg["gamma"] <= 1 or not 0 <= alg["gae_lambda"] <= 1:
        raise ConfigError("gamma/gae_lambda须在[0,1]")
    if not math.isfinite(a["model"]["lr"]) or a["model"]["lr"] <= 0:
        raise ConfigError("model.lr须正有限")
    if (
        any(alg[k] < 0 for k in ("entropy_coef", "value_loss_coef"))
        or alg["max_grad_norm"] <= 0
    ):
        raise ConfigError("非法PPO损失/梯度系数")


def _space_env(observation_space, action_space):
    import gymnasium as gym
    import numpy as np

    class SpacesOnly(gym.Env):
        """仅给SB3声明尺寸；采样必须经过真实的阶段调度器。"""

        def __init__(self):
            self.observation_space, self.action_space = observation_space, action_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            raise RuntimeError("不能通过占位空间环境采样；使用顺序收集器")

    return SpacesOnly()


def _device(config):
    import torch

    cfg = config.algo_args["device"]
    torch.set_num_threads(cfg["torch_threads"])
    if cfg["cuda"] and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA但不可用")
    torch.backends.cudnn.deterministic = cfg["cuda_deterministic"]
    torch.backends.cudnn.benchmark = not cfg["cuda_deterministic"]
    return "cuda" if cfg["cuda"] else "cpu"


def make_policies(config, env, device, output):
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure

    a = config.algo_args
    alg = a["algo"]
    policies = []
    for i, (obs, action) in enumerate(zip(env.observation_spaces, env.action_spaces)):
        model = PPO(
            "MlpPolicy",
            _space_env(obs, action),
            device=device,
            seed=(a["seed"]["seed"] + i) % 2**32,
            n_steps=a["train"]["episode_length"],
            batch_size=alg["batch_size"],
            n_epochs=alg["ppo_epoch"],
            learning_rate=a["model"]["lr"],
            gamma=alg["gamma"],
            gae_lambda=alg["gae_lambda"],
            clip_range=alg["clip_param"],
            ent_coef=alg["entropy_coef"],
            vf_coef=alg["value_loss_coef"],
            max_grad_norm=alg["max_grad_norm"],
            policy_kwargs=dict(net_arch=a["model"]["hidden_sizes"]),
            verbose=0,
        )
        model.set_logger(configure(str(output / f"agent_{i}"), []))
        policies.append(model)
    return policies


def sample(model, observation, deterministic=False):
    import torch
    import numpy as np

    model.policy.set_training_mode(False)
    with torch.no_grad():
        obs, _ = model.policy.obs_to_tensor(
            np.asarray(observation, dtype=np.float32)[None]
        )
        action, value, log_prob = model.policy(obs, deterministic=deterministic)
    return int(action.item()), value, log_prob


def collect_episode(env, models, *, training, episode_index=0, trace=None):
    """本轮价格观测生成后才调用跟随者策略；一条工程转移分别写入各自buffer。"""
    import numpy as np
    import torch

    leader = env.reset(load_profile_idx=episode_index)
    totals = np.zeros(len(models), dtype=np.float64)
    records = []
    if training:
        for model in models:
            model.rollout_buffer.reset()
    for t in range(env.horizon):
        sampled = [sample(models[0], leader, not training)]
        followers = env.publish_price(sampled[0][0])
        observations = [leader, *followers]
        # 此处不能移到publish_price之前。
        sampled += [
            sample(model, obs, not training)
            for model, obs in zip(models[1:], followers)
        ]
        if trace is not None:
            trace.append(
                dict(
                    t=t,
                    price_index=sampled[0][0],
                    follower_observations=[o.tolist() for o in followers],
                )
            )
        leader, rewards, done, info = env.respond([s[0] for s in sampled[1:]])
        if training:
            for i, (model, obs, (action, value, log_prob)) in enumerate(
                zip(models, observations, sampled)
            ):
                model.rollout_buffer.add(
                    np.asarray(obs)[None],
                    np.array([[action]]),
                    np.array([rewards[i]], dtype=np.float32),
                    np.array([t == 0]),
                    value,
                    log_prob,
                )
                model.num_timesteps += 1
        totals += rewards
        records.append(info)
    if not done:
        raise RuntimeError("完整回合未自然结束")
    if training:
        for model in models:
            # 观测含有限回合进度；自然终止，无未来自举或跨主体价值复制。
            model.rollout_buffer.compute_returns_and_advantage(
                torch.zeros(1, device=model.device), np.ones(1, dtype=bool)
            )
    return totals, records


def save_checkpoint(config, env, models, output):
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.resolved.yaml")
    (output / "physics_manifest.json").write_text(
        json.dumps(env.physics_manifest(), indent=2, allow_nan=False), encoding="utf-8"
    )
    for i, model in enumerate(models):
        model.save(output / f"agent_{i}.zip")


def load_checkpoint(path, *, device="cpu", split="test"):
    from utils.config_pipeline import ConfigLoader
    from stable_baselines3 import PPO
    from envs.stackelberg.pricing_env import SequentialPricingEnv

    path = Path(path)
    config = ConfigLoader().load(
        experiment=path / "config.resolved.yaml", overrides={"env.data.split": split}
    )
    if config.main_args["algo"] != "sequential_ppo":
        raise ValueError("不是顺序独立PPO checkpoint")
    env = SequentialPricingEnv(config.env_args)
    try:
        expected = json.loads(
            (path / "physics_manifest.json").read_text(encoding="utf-8")
        )
        if env.physics_manifest() != expected:
            raise ValueError("checkpoint物理/经济/动作/观测契约不一致")
        models = [
            PPO.load(path / f"agent_{i}.zip", env=_space_env(obs, act), device=device)
            for i, (obs, act) in enumerate(
                zip(env.observation_spaces, env.action_spaces)
            )
        ]
        return env, models
    except Exception:
        env.close()
        raise


def train(config, output=None):
    import numpy as np
    import torch
    from envs.stackelberg.pricing_env import SequentialPricingEnv

    validate_settings(config)
    output = Path(
        output
        or Path(config.algo_args["logger"]["log_dir"])
        / "stackelberg"
        / config.main_args["exp_name"]
        / f"run_{time.time_ns()}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = SequentialPricingEnv(config.env_args)
    validation = None
    models = []
    try:
        device = _device(config)
        models = make_policies(config, env, device, output)
        a = config.algo_args
        if a["eval"]["use_eval"]:
            cfg = copy.deepcopy(config.env_args)
            cfg["data"]["split"] = "validation"
            validation = SequentialPricingEnv(cfg)
        updates = a["train"]["num_env_steps"] // env.horizon
        summary = dict(
            device=device,
            physical_steps=0,
            policy_updates=0,
            validation_episodes=0,
            agents=env.agent_names,
            episode_utilities_yuan=[],
            validation_utilities_yuan=[],
        )
        with (output / "intervals.jsonl").open("w", encoding="utf-8") as log:
            for update in range(1, updates + 1):
                totals, records = collect_episode(
                    env,
                    models,
                    training=True,
                    episode_index=(update - 1) % len(env.dataset.episodes),
                )
                for model in models:
                    model.train()
                    if not all(
                        torch.isfinite(p).all() for p in model.policy.parameters()
                    ):
                        raise RuntimeError("非有限策略参数")
                summary["physical_steps"] += env.horizon
                summary["policy_updates"] += 1
                summary["episode_utilities_yuan"].append(totals.tolist())
                for row in records:
                    log.write(
                        json.dumps(dict(update=update, **row), allow_nan=False) + "\n"
                    )
                if validation and update % a["train"]["eval_interval"] == 0:
                    for e in range(a["eval"]["eval_episodes"]):
                        values, _ = collect_episode(
                            validation,
                            models,
                            training=False,
                            episode_index=e % len(validation.dataset.episodes),
                        )
                        summary["validation_utilities_yuan"].append(values.tolist())
                        summary["validation_episodes"] += 1
                if update % a["train"]["save_interval"] == 0 or update == updates:
                    save_checkpoint(config, env, models, output / "checkpoint")
                if update % a["train"]["log_interval"] == 0:
                    print(
                        "SEQUENTIAL_PPO_UPDATE",
                        update,
                        summary["physical_steps"],
                        np.round(totals, 4).tolist(),
                        flush=True,
                    )
        summary["sb3_n_updates"] = [m._n_updates for m in models]
        summary["output"] = str(output)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        print("SEQUENTIAL_PPO_OK", str(output), flush=True)
        return summary
    finally:
        env.close()
        if validation:
            validation.close()
        for model in models:
            model.get_env().close()
