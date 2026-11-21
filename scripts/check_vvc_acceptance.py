"""在隔离工作区验收一套 IEEE13 + PPO 参考组合。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.acceptance import (
    AcceptanceBlocked,
    AcceptanceChecks,
    run_acceptance,
    write_report,
)
from utils.capabilities import device_layout, preflight
from utils.config_pipeline import ConfigLoader

EXPERIMENT = "configs/experiments/vvc_13bus_smoke.yaml"
BACKEND = "sb3_ppo"


def _parameter_digest(policy):
    digest = sha256()
    for name, parameter in policy.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _native_readback(wrapper, ids):
    import math

    core = wrapper.env
    core.reset(load_profile_idx=0)
    circuit = core.circuit
    # 对第一电容执行翻转，调压器命令靠近初始档位，储能给小幅放电。
    actions = [int(circuit.capacitor_state(name)[0]) for name in core.cap_names]
    actions[0] = 1 - actions[0]
    actions += [
        round(
            (reg.initial - reg.mintap)
            / (reg.maxtap - reg.mintap)
            * (core.reg_act_num - 1)
        )
        for reg in circuit.regulators.values()
    ]
    actions += [core.bat_act_num // 2 + 1] * core.bat_num
    _, _, _, _, infos, _ = wrapper.step([[action] for action in actions])
    info = infos[0]
    readback = {}
    for element in ids:
        trace = circuit.trace[element]
        quantity, tolerance = {
            "Capacitor": ("state", 0),
            "RegControl": ("tap", 1e-6),
            "Storage": ("kw", 0.05),
        }[element.split(".", 1)[0]]
        requested = float(trace["requested"][quantity])
        applied = float(trace["applied"][quantity])
        measured = float(trace["measured"][quantity])
        readback[element] = {
            "quantity": quantity,
            "requested": requested,
            "applied": applied,
            "measured": measured,
            "tolerance": tolerance,
            "matched": math.isclose(requested, applied, rel_tol=0, abs_tol=tolerance)
            and math.isclose(applied, measured, rel_tol=0, abs_tol=tolerance),
        }
    return {
        "device_ids": ids,
        "readback": readback,
        "converged": info["dss_converged"],
        "active_balance_error_kw": float(info["active_balance_error_kw"]),
        "balance_tolerance_kw": 0.05,
        "model_sha256": circuit.model_sha256,
        "scenario_id": info["scenario_id"],
    }


class ReferenceChecks:
    def __init__(self, config, output, cleanup):
        self.config = config
        self.output = output
        self.cleanup = cleanup
        self.checkpoint = output / "policy.zip"

    def physical_readback(self):
        try:
            import dss
            from envs.powergrid.adapters import CentralizedAdapter
            from envs.powergrid.config import DataConfig
            from envs.powergrid.scenario import generate_dataset
            from utils.envs_tools import make_power_env
            from utils.training_setup import prepare_run
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise AcceptanceBlocked(f"参考运行缺少依赖：{exc}") from exc

        self.ppo = PPO
        self.make_power_env = make_power_env
        self.centralized = CentralizedAdapter
        data = self.config.env_args["data"]
        steps = self.config.env_args["max_episode_steps"]
        # 新合成夹具。实际环境只读取 train 或 validation 的 CSV。
        data["path"] = str(
            generate_dataset(self.output / "synthetic", DataConfig(**data), steps)
        )
        prepare_run(self.config)
        args = self.config.env_args
        wrapper = make_power_env("vvc", args, 0, args["seed"])
        self.cleanup.callback(wrapper.close)
        layout = device_layout(wrapper)
        bound = preflight(
            self.config,
            backend=BACKEND,
            layout=layout,
            checkpoint="policy_reload",
            independent_replay=True,
        )
        if not bound.compatible:
            raise ValueError("; ".join(bound.issues))
        self.env = CentralizedAdapter(wrapper.env)
        self.manifest = self.env.core.get_physics_manifest()
        result = _native_readback(wrapper, layout.device_ids)
        result["runtime"] = {
            "python": sys.version.split()[0],
            "dss_python": dss.__version__,
            "gymnasium": version("gymnasium"),
            "numpy": version("numpy"),
            "sb3": version("stable-baselines3"),
            "torch": version("torch"),
        }
        self.config.save(self.output / "resolved.yaml")
        return result

    def minimal_update(self):
        import gymnasium
        import torch

        scenarios = {}

        class TrainingEvidence(gymnasium.Wrapper):
            def step(self, action):
                entry = self.env.core.entry
                scenarios[entry["id"]] = entry["sha256"]
                transition = self.env.step(action)
                info = transition[4]
                if info.get("physical_failure") or not info["dss_converged"]:
                    raise RuntimeError("训练期间物理求解失败")
                return transition

        torch.set_num_threads(1)
        self.model = self.ppo(
            "MlpPolicy",
            TrainingEvidence(self.env),
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": [16, 16]},
            device="cpu",
            seed=self.config.env_args["seed"],
        )
        before = _parameter_digest(self.model.policy)
        self.model.learn(total_timesteps=4)
        return {
            "steps": self.model.num_timesteps,
            "updates": self.model._n_updates,
            "parameters_before_sha256": before,
            "parameters_after_sha256": _parameter_digest(self.model.policy),
            "training_scenarios": scenarios,
            "dataset_sha256": self.env.core.dataset.sha256,
        }

    def save_reload(self):
        import numpy as np

        from utils.centralized_cli import load_model

        obs, _ = self.env.reset(options={"scenario_index": 0})
        predicted, _ = self.model.predict(obs, deterministic=True)
        self.model.save(self.checkpoint)
        self.checkpoint.with_suffix(".physics.json").write_text(
            json.dumps(self.manifest, allow_nan=False), encoding="utf-8"
        )
        restored = load_model(self.checkpoint, self.env)
        loaded, _ = restored.predict(obs, deterministic=True)
        return {
            "checkpoint_sha256": sha256(self.checkpoint.read_bytes()).hexdigest(),
            "prediction_match": bool(np.array_equal(predicted, loaded)),
        }

    def independent_replay(self):
        import math

        from utils.centralized_cli import load_model

        args = deepcopy(self.config.env_args)
        args["data"]["split"] = "validation"
        wrapper = self.make_power_env("vvc", args, 0, args["seed"] + 1)
        self.cleanup.callback(wrapper.close)
        env = self.centralized(wrapper.env)
        policy = load_model(self.checkpoint, env)
        obs, _ = env.reset(options={"scenario_index": 0})
        entry = env.core.entry
        violations = 0
        balance = 0.0
        for step in range(1, env.core.horizon + 1):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if (
                info.get("physical_failure")
                or not info["dss_converged"]
                or not math.isfinite(reward)
            ):
                raise RuntimeError(f"验证工况第 {step} 步求解或奖励无效")
            error = abs(float(info["active_balance_error_kw"]))
            if not math.isfinite(error) or error > 0.05:
                raise RuntimeError(f"验证工况第 {step} 步有功不平衡：{error} kW")
            balance = max(balance, error)
            violations += int(info["voltage_violation_count"])
            if terminated or truncated:
                break
        return {
            "split": entry["split"],
            "scenario_id": entry["id"],
            "scenario_sha256": entry["sha256"],
            "dataset_sha256": env.core.dataset.sha256,
            "checkpoint_sha256": sha256(self.checkpoint.read_bytes()).hexdigest(),
            "steps": step,
            "expected_steps": env.core.horizon,
            "terminated": terminated,
            "truncated": truncated,
            "physical_failure": bool(info.get("physical_failure")),
            "phase_violations": violations,
            "max_balance_error_kw": balance,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="全新输出目录")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True)
    config = ConfigLoader().load(
        experiment=EXPERIMENT,
        overrides={"seed.seed": 17, "seed.seed_specify": True, "env.data.seed": 118603},
    )
    result = preflight(
        config, backend=BACKEND, checkpoint="policy_reload", independent_replay=True
    )
    with ExitStack() as cleanup:
        hooks = ReferenceChecks(config, output, cleanup)
        report = run_acceptance(
            result,
            AcceptanceChecks(
                hooks.physical_readback,
                hooks.minimal_update,
                hooks.save_reload,
                hooks.independent_replay,
            ),
        )
        write_report(report, output / "report.json")
    print(output / "report.json")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    sys.exit(main())
