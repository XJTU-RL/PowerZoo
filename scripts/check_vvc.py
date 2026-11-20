"""三馈线VVC物理、双worker、训练和检查点实际回放验收。"""

from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_logged(command, path):
    with path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=240,
        )
    content = path.read_text(encoding="utf-8")
    assert (
        "Traceback (most recent call last)" not in content
        and "| ERROR |" not in content
    )


def check_discrete_validation_equivalence(snapshot):
    """Opt-in native gate: frozen validation only; never read test trajectories."""
    import numpy as np
    from dataclasses import asdict
    from envs.vvc.runtime_env import Env
    from envs.vvc.vvc.vvc_config import VVCConfig
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import build_dataclass
    from dataclasses import replace
    from envs.powergrid.adapters import DeviceMARLAdapter
    from envs.vvc.single_agent import SingleAgentVVCEnv

    cfg = ConfigLoader().load(experiment=snapshot)
    base = build_dataclass(VVCConfig, cfg.env_args)
    assert base.system_name == "34Bus" and base.data.split == "train"
    # Isolated reward sensitivity fixture, not a change to the benchmark config.
    config = replace(base, data=replace(base.data, split="validation"), soc_w=5.)
    single = SingleAgentVVCEnv(config, "discrete")
    multi = DeviceMARLAdapter(Env(str(ROOT), config))
    a, b = single.core, multi.env
    rows = []
    calls = [0, 0]
    native_calls = [0, 0]

    def equal(left, right):
        if isinstance(left, dict):
            assert set(left) == set(right)
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                equal(x, y)
        elif isinstance(left, (float, int, np.number)):
            np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-7)
        else:
            assert left == right

    try:
        sa, _ = single.reset(seed=17, options={"scenario_index":0})
        b.reset(load_profile_idx=0, seed=17)
        assert a.entry["split"] == b.entry["split"] == "validation"
        assert a.entry["id"] == b.entry["id"]
        assert single.action_space.nvec.tolist() == [2, 2, *([33]*8)]
        np.testing.assert_allclose(sa, b.get_obs(), atol=1e-7, rtol=1e-7)
        neutral = [int(a.circuit.capacitor_state(n)[0]) for n in a.cap_names] + [
            round((a.circuit.regulator_tap(n)-r.mintap)/(r.maxtap-r.mintap)*32)
            for n,r in a.circuit.regulators.items()] + [16,16]
        for index, core in enumerate((a,b)):
            original = core.circuit.solve
            def counted(*args, _index=index, _original=original, **kwargs):
                calls[_index] += 1
                return _original(*args, **kwargs)
            core.circuit.solve = counted
            original_native = core.circuit._solve
            def counted_native(*args, _index=index, _original=original_native, **kwargs):
                native_calls[_index] += 1
                return _original(*args, **kwargs)
            core.circuit._solve = counted_native
        action = neutral.copy()
        for step in range(24):
            if step == 0:
                action[0] = 1-neutral[0]
                action[2] = neutral[2]+1
                action[3] = neutral[3]-1
                action[-2:] = [17,15]
            else:
                action[-2:] = [16,16]
            so, reward, done, truncated, info = single.step(np.asarray(action, dtype=np.int64))
            mo, state, rewards, dones, infos, masks = multi.step([[x] for x in action])
            assert info["dss_converged"] and infos[0]["dss_converged"]
            assert not truncated and done == bool(dones.all()) == (step == 23)
            np.testing.assert_allclose(so, mo[0], atol=1e-7, rtol=1e-7)
            # One scalar team reward; repeated per-agent copies are not summed.
            np.testing.assert_allclose(rewards, reward, atol=1e-6, rtol=1e-6)
            equal(asdict(a.physical_state), asdict(b.physical_state))
            equal(a.circuit.trace, b.circuit.trace)
            for i,name in enumerate(a.cap_names):
                trace = a.circuit.trace["Capacitor."+name]
                assert trace["requested"]["state"] == action[i]
                assert trace["measured"]["state"] == action[i]
            for i,(name,reg) in enumerate(a.circuit.regulators.items(),2):
                expected = reg.mintap + action[i]/32*(reg.maxtap-reg.mintap)
                trace = a.circuit.trace["RegControl."+name]
                np.testing.assert_allclose(trace["requested"]["tap"], expected, atol=1e-7)
                np.testing.assert_allclose(trace["measured"]["tap"], expected, atol=1e-7)
            for i,name in enumerate(a.bat_names):
                expected = (2*action[-2+i]/32-1)*a.circuit.batteries[name].max_kw
                trace = a.circuit.trace["Storage."+name]
                np.testing.assert_allclose(trace["requested"]["kw"], expected, atol=1e-7)
                np.testing.assert_allclose(trace["applied"]["kw"], expected, atol=1e-7)
                # Terminal power may differ from the command under native device physics.
                # Check the independently read terminal quantity, not a widened command tolerance.
                native_pq = -a.circuit.powers("Storage."+name)
                np.testing.assert_allclose(trace["measured"]["kw"], native_pq[0], atol=1e-7)
                np.testing.assert_allclose(trace["measured"]["kvar"], native_pq[1], atol=1e-7)
                native_soc = float(a.circuit.read_property("Storage."+name, "%stored"))/100
                np.testing.assert_allclose(a.circuit.batteries[name].soc, native_soc, atol=1e-7)
            assert native_calls[0] == native_calls[1]
            assert a.circuit.time_s == b.circuit.time_s == (step+1)*3600
            assert calls == [step+1,step+1]
            rows.append({"step":step+1,"action":action.copy(),"reward":reward,
                         "time_s":a.circuit.time_s,"trace":a.circuit.trace,
                         "state":asdict(a.physical_state)})
        deviation = sum(abs(dev.soc-dev.initial_soc) for dev in a.circuit.batteries.values())
        assert deviation > 1e-5
        np.testing.assert_allclose(info["ctrl_reward"], -5*deviation, atol=1e-7)
        report = {"scenario_id":a.entry["id"],"split":"validation","seed":17,
                  "steps_per_arm":24,"solve_calls":calls,"native_solve_calls":native_calls,
                  "regulator_bindings":{n:{"transformer":r.transformer,"tap_winding":r.tap_winding}
                                        for n,r in a.circuit.regulators.items()},"terminal_soc_deviation":deviation,
                  "terminal_ctrl_reward":info["ctrl_reward"],"rows":rows}
        return report
    finally:
        single.close()
        multi.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--equivalence-config", type=Path)
    parser.add_argument("--output", type=Path, help="equivalence JSON private output path")
    args = parser.parse_args()
    if args.equivalence_config:
        if args.train or args.require_cuda or not args.output:
            parser.error("equivalence requires --output and cannot combine with training")
        report = check_discrete_validation_equivalence(args.equivalence_config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, allow_nan=False))
        print("VVC_DISCRETE_EQUIVALENCE_OK", report["steps_per_arm"], flush=True)
        return
    from scripts.check_smartgrid import check_native, check_spawn

    output = ROOT / "results/validation/vvc"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for system in ("13Bus", "34Bus", "123Bus"):
        profile = f"configs/experiments/vvc_{system.lower()}_smoke.yaml"
        entry = {
            "native": check_native(profile, steps=24),
            "spawn": check_spawn(profile),
        }
        if args.train:
            import numpy as np
            import torch

            before = set((output / "runs").rglob("*.pt"))
            devices = (
                ["cpu", "cuda"] if args.require_cuda and system == "13Bus" else ["cpu"]
            )
            for device in devices:
                run_logged(
                    [
                        "scripts/train.py",
                        "--config",
                        profile,
                        "--exp-name",
                        f"check_{system}_{device}",
                        "--set",
                        f"device.cuda={str(device=='cuda').lower()}",
                        "--set",
                        f"logger.log_dir={(output/'runs').as_posix()}",
                        "--set",
                        "eval.use_eval=true",
                        "--set",
                        "eval.n_eval_rollout_threads=1",
                        "--set",
                        "eval.eval_episodes=6",
                        "--set",
                        "train.eval_interval=5",
                    ],
                    output / f"{system}_{device}.log",
                )
                entry["happo_" + device] = {
                    "steps": 240,
                    "updates": 10,
                    "validation_episodes": 12,
                }
            created = set((output / "runs").rglob("*.pt")) - before
            assert created
            for path in created:
                state = torch.load(path, map_location="cpu", weights_only=True)

                def finite(value):
                    if isinstance(value, torch.Tensor):
                        assert torch.isfinite(value).all(), path
                    elif isinstance(value, dict):
                        for item in value.values():
                            finite(item)

                finite(state)
            from envs.powergrid.checkpoint import load_checkpoint
            from envs.vvc.render.engine.episode_runner import VVCEpisodeRunner

            checkpoint = max(
                (p.parent for p in created if p.name == "actor_agent0.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            engine = load_checkpoint(checkpoint, expected_family="vvc")
            runner = VVCEpisodeRunner(engine.config, engine, project_root=str(ROOT))
            try:
                episode = runner.run_episode(seed=5, max_steps=24)
                assert episode.episode_length == 24
                assert np.isclose(
                    episode.total_reward,
                    sum(s["info"]["reward_total"] for s in episode.snapshots[1:]),
                    rtol=1e-6,
                )
                entry["happo_replay"] = {
                    "actors": engine.n_agents,
                    "steps": 24,
                    "finite_model_files": len(created),
                }
            finally:
                runner.close()
            model = output / f"{system}_ppo"
            run_logged(
                [
                    "scripts/train_centralized.py",
                    "--config",
                    profile,
                    "--steps",
                    "96",
                    "--output",
                    str(model),
                ],
                output / f"{system}_sb3.log",
            )
            from utils.config_pipeline import ConfigLoader
            from utils.envs_tools import make_power_env
            from envs.powergrid.adapters import CentralizedAdapter
            from utils.centralized_cli import load_model

            cfg = ConfigLoader().load(experiment=model.with_suffix(".environment.yaml"))
            wrapper = make_power_env("vvc", cfg.env_args, 0, cfg.env_args["seed"])
            central = CentralizedAdapter(wrapper.env)
            try:
                policy = load_model(model, central)
                obs, _ = central.reset(seed=7)
                for _ in range(24):
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = central.step(action)
                    assert np.isfinite(reward) and info["dss_converged"]
                assert done and not truncated
                entry["sb3_ppo"] = {"steps": 96, "updates": 4, "replay_steps": 24}
            finally:
                central.close()
        report[system] = entry
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        print("VVC_CASE_OK", system, flush=True)
    print("VVC_RUNTIME_OK", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
