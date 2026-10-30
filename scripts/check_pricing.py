"""顺序需求响应：原生账本、穷举参考、Windows隔离和CPU/CUDA独立PPO实测。"""

from pathlib import Path
import argparse
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def native(case):
    import numpy as np
    from utils.config_pipeline import ConfigLoader
    from utils.training_setup import prepare_run
    from envs.stackelberg.pricing_env import SequentialPricingEnv

    report = dict(episodes=[], reference=[], max_balance_error_kw=0.0)
    for split in ("train", "validation", "test"):
        cfg = prepare_run(
            ConfigLoader().load(
                "sequential_ppo", case, overrides={"env.data.split": split}
            )
        )
        env = SequentialPricingEnv(cfg.env_args)
        try:
            for index in range(len(env.dataset.episodes)):
                env.reset(load_profile_idx=index)
                reference = env.exact_reference()
                best = reference["best"]
                assert env.circuit.time_s == env.profile[0, 0]
                assert (
                    max(env.unilateral_gains(best["price_index"], best["response"]))
                    <= 1e-7
                )
                report["reference"].append(
                    dict(split=split, scenario=env.entry["id"], **reference)
                )
                for rule in ("full_service", "exact_current_price_response"):
                    env.reset(load_profile_idx=index)
                    totals = np.zeros(3)
                    energy = np.zeros(2)
                    max_regret = 0.0
                    for t in range(env.horizon):
                        price = 2
                        response = (2, 2)
                        if rule == "exact_current_price_response":
                            table, equilibria = env.exact_response_table(price)
                            response = equilibria[0]
                            max_regret = max(
                                max_regret, *env.unilateral_gains(price, response)
                            )
                            assert max_regret <= 1e-7
                        initial = env.circuit.time_s
                        env.publish_price(price)
                        assert env.circuit.time_s == initial
                        _, utilities, done, info = env.respond(response)
                        assert np.isfinite(utilities).all() and info["dss_converged"]
                        assert abs(info["balance_error_kw"]) < 0.05
                        assert info["time_s"] == initial + env.config.data.step_seconds
                        assert math.isclose(
                            sum(utilities),
                            -info["wholesale_cost_yuan"]
                            - info["network_cost_yuan"]
                            - sum(info["consumer_discomfort_yuan"]),
                            abs_tol=1e-5,
                        )
                        totals += utilities
                        energy += info["consumer_energy_kwh"]
                        report["max_balance_error_kw"] = max(
                            report["max_balance_error_kw"],
                            abs(info["balance_error_kw"]),
                        )
                    assert done
                    report["episodes"].append(
                        dict(
                            split=split,
                            scenario=env.entry["id"],
                            rule=rule,
                            steps=env.horizon,
                            utilities_yuan=totals.tolist(),
                            consumer_energy_kwh=energy.tolist(),
                            max_follower_deviation_yuan=max_regret,
                        )
                    )
            report["physics"] = env.physics_manifest()
        finally:
            env.close()
    return report


def worker(case):
    from utils.config_pipeline import ConfigLoader
    from envs.stackelberg.pricing_env import SequentialPricingEnv

    env = SequentialPricingEnv(ConfigLoader().load("sequential_ppo", case).env_args)
    try:
        env.reset(load_profile_idx=0)
        start = env.circuit.time_s
        for episode in range(2):
            for step in range(env.horizon):
                env.publish_price(1)
                _, _, done, info = env.respond([1, 1])
            assert done
            if episode == 0:
                env.reset(load_profile_idx=0)
        return dict(
            start_s=start,
            end_s=info["time_s"],
            physical_steps=48,
            actions=[int(s.n) for s in env.action_spaces],
        )
    finally:
        env.close()


def training(case, output, require_cuda):
    from utils.config_pipeline import ConfigLoader
    from utils.training_setup import prepare_run
    from envs.stackelberg.training import (
        train,
        load_checkpoint,
        collect_episode,
        sample,
    )

    report = {}
    for device in ["cpu", "cuda"] if require_cuda else ["cpu"]:
        cfg = prepare_run(
            ConfigLoader().load(
                "sequential_ppo", case, overrides={"device.cuda": device == "cuda"}
            )
        )
        result = train(cfg, output / f"{device}_{time.time_ns()}")
        assert result["physical_steps"] == 240 and result["policy_updates"] == 10
        assert (
            result["sb3_n_updates"] == [10, 10, 10]
            and result["validation_episodes"] == 4
        )
        env, models = load_checkpoint(
            Path(result["output"]) / "checkpoint", device=device, split="test"
        )
        try:
            totals, records = collect_episode(
                env, models, training=False, episode_index=0
            )
            assert len(records) == 24 and len(totals) == 3
            assert all(m.num_timesteps == 240 for m in models)
            # 逐主体单边偏离；领导者偏离重新调用已训练响应策略，不能复用旧价格下的用户动作。
            leader = env.reset(load_profile_idx=0)
            chosen = sample(models[0], leader, True)[0]
            responses, profits = {}, []
            for price in range(env.action_spaces[0].n):
                env.reset(load_profile_idx=0)
                followers = env.publish_price(price)
                response = [
                    sample(m, o, True)[0] for m, o in zip(models[1:], followers)
                ]
                responses[price] = response
                profits.append(float(env.evaluate_response(price, response)[0][0]))
            env.reset(load_profile_idx=0)
            follower_gains = env.unilateral_gains(chosen, responses[chosen])
            result["replay"] = dict(
                steps=24,
                actors=3,
                utilities_yuan=totals.tolist(),
                follower_one_interval_deviation_yuan=follower_gains,
                leader_policy_response_deviation_yuan=max(profits) - profits[chosen],
                price_conditioned_responses=responses,
            )
        finally:
            env.close()
            for model in models:
                model.get_env().close()
        report[device] = result
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    output = ROOT / "results/validation/pricing"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for case in ("stackelberg", "stackelberg_13bus"):
        entry = dict(native=native(case))
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            entry["spawn"] = list(pool.map(worker, [case, case]))
        if args.train:
            entry["training"] = training(case, output, args.require_cuda)
        report[case] = entry
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        print("PRICING_CASE_OK", case, flush=True)
    print("PRICING_RUNTIME_OK", flush=True)


if __name__ == "__main__":
    main()
