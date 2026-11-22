"""Compatibility checks for the public discrete VVC baselines."""

import math

from utils.config_validation import ConfigError

DISCRETE_VVC_ALGORITHMS = frozenset({"ippo", "qmix", "vdn", "coma"})


def validate_discrete_vvc(name, family, algorithm, env):
    """Validate the discrete VVC baselines before allocating data or workers."""
    if family != "vvc" or env.get("state_type") != "EP":
        raise ConfigError(f"{name} 需要采用 EP 团队状态的 VVC 环境")
    if env.get("observation_mode") != "local" or env["bat_act_num"] == math.inf:
        raise ConfigError(f"{name} 需要局部观测和离散设备动作")
    train, model, settings = (algorithm[key] for key in ("train", "model", "algo"))
    if (
        model["use_recurrent_policy"]
        or model["use_naive_recurrent_policy"]
        or algorithm["render"]["use_render"]
    ):
        raise ConfigError(f"{name} 通过公共入口支持前馈网络和训练模式")
    rollout_steps = train["episode_length"] * train["n_rollout_threads"]
    if rollout_steps < 1 or train["num_env_steps"] % rollout_steps:
        raise ConfigError("train.num_env_steps 必须为向量采样长度的整数倍")
    if name in {"ippo", "coma"} and settings["share_param"] is not False:
        raise ConfigError(f"{name} 需要独立 actor，请将 algo.share_param 设为 false")
    if name == "coma" and (
        train["use_valuenorm"]
        or not 0 <= settings["td_lambda"] <= 1
        or not 0 < settings["target_tau"] <= 1
    ):
        raise ConfigError(
            "COMA 需要关闭 ValueNorm，并设置有效的 td_lambda 与 target_tau"
        )
    if name in {"qmix", "vdn"}:
        capacity, batch = settings["buffer_size"], settings["batch_size"]
        if not 1 <= batch <= capacity or capacity < env["max_episode_steps"]:
            raise ConfigError(
                "回放容量须容纳完整环境回合，且 1 <= batch_size <= buffer_size"
            )
        if (
            train["train_interval"] < 1
            or settings["epsilon_anneal_time"] < 1
            or not 0 <= settings["epsilon_finish"] <= settings["epsilon_start"] <= 1
            or not 0 < settings["tau"] <= 1
        ):
            raise ConfigError("请检查更新间隔、epsilon 调度和目标网络的 tau")
    if name == "qmix" and (
        settings["hypernet_layers"] not in (1, 2)
        or min(settings["mixer_hidden_dim"], settings["hypernet_hidden_dim"]) < 1
    ):
        raise ConfigError(
            "QMIX 超网络须为 1 或 2 层，且 mixer 和超网络的隐层宽度须为正"
        )


def require_discrete_spaces(spaces):
    from gymnasium.spaces import Discrete

    if not spaces or not all(
        isinstance(space, Discrete) and space.start == 0 for space in spaces
    ):
        raise ValueError(
            "Discrete VVC baselines require one zero-based Discrete space per device"
        )
