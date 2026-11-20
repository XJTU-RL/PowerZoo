"""Export IEEE 34Bus hold-initial and DSS static-regulator validation baselines."""

import argparse
import math
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

from scripts.export_vvc_validation import _normalize
from scripts.vvc_benchmark_metrics import summarize_episode
from scripts.vvc_device_metrics import device_output, measure_devices, reset_policy_input_sha256


STOCK_34_CLOSURE = {
    "node_systems/34Bus/network.dss":
        "97fa112cb3c9b1d8a726811b37d71955b0ef0774ac98b9bfe3182ec2eaee2427",
    "node_systems/34Bus/IEEELineCodes.dss":
        "f4cef900cd68b11959ab689082e5d579f56ebacde222389bf06f5baa5f46d5fc",
}
MODES = ("hold_initial", "native_regulator")
# Deliberately closed to the classic 34Bus inventory. Any new element class
# could add a second writer and must be audited before a native comparison.
ELEMENT_CLASSES = {
    "vsource",
    "line",
    "load",
    "transformer",
    "capacitor",
    "storage",
    "regcontrol",
}


def _runtime(resolved_config):
    from envs.vvc.vvc.env_register import make_base_env
    from envs.vvc.vvc.vvc_config import VVCConfig
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import build_dataclass, environment_family
    from utils.physics_manifest import compare_manifests
    import yaml

    raw = yaml.safe_load(Path(resolved_config).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("config_kind") != "resolved":
        raise ValueError("provide the exact training config.resolved.yaml")
    cfg = ConfigLoader().load(experiment=resolved_config)
    if environment_family(cfg.main_args["env"]) != "vvc":
        raise ValueError("baseline requires a VVC config")
    return build_dataclass(VVCConfig, cfg.env_args), make_base_env, compare_manifests


def _inventory(circuit):
    elements = tuple(circuit.native.AllElementNames)
    classes = {name.partition(".")[0].lower() for name in elements}
    if not elements or classes - ELEMENT_CLASSES:
        raise ValueError(
            f"unaudited DSS element classes: {sorted(classes - ELEMENT_CLASSES)}"
        )
    names = {
        name.split(".", 1)[1].lower()
        for name in elements
        if name.lower().startswith("regcontrol.")
    }
    if names != set(circuit.regulators) or len(names) != 6:
        raise ValueError("expected all six classic IEEE 34Bus RegControls")
    rows, owners = [], {}
    for name, reg in sorted(circuit.regulators.items()):
        el = f"RegControl.{name}"
        circuit.require_element(el)
        enabled = bool(circuit.native.ActiveCktElement.Enabled)
        target = (reg.transformer.lower(), int(reg.tap_winding))
        max_tap_change = int(circuit.read_property(el, "maxtapchange"))
        if enabled:
            if max_tap_change <= 0 or target in owners:
                raise ValueError(f"ambiguous or immobile native regulator: {el}")
            owners[target] = name
        rows.append(
            {
                "controller": name,
                "enabled": enabled,
                "transformer": target[0],
                "tap_winding": target[1],
                "max_tap_change": max_tap_change,
            }
        )
    if not owners:
        raise ValueError("no enabled native regulator controllers")
    return rows, owners


def _states(circuit):
    return {
        "capacitors": {
            n: [int(state) for state in circuit.capacitor_state(n)]
            for n in circuit.capacitors
        },
        "tap_pu": {n: float(circuit.regulator_tap(n)) for n in circuit.regulators},
        "storage_soc": {n: float(dev.soc) for n, dev in circuit.batteries.items()},
    }


def _step(core, native):
    circuit = core.circuit
    try:
        circuit.begin_interval(float(core.profile[core.t, 0]))
        core._apply_data(core.t)
        circuit.apply({f"Storage.{n}": {"kw": 0.0} for n in circuit.batteries})
        circuit.solve(native_controls=native)
        circuit.finish_interval()
        core.t += 1
        core._observe()
    except RuntimeError as exc:
        if not circuit._failed:
            raise
        core._diverged = True
        _, _, _, failure = core._divergence_transition(exc)
        return failure
    return core._info()


def _run(core, saved_physics, compare_manifests, index, seed, native, *, split="validation", paired=False):
    core.seed(seed)
    core.reset(load_profile_idx=index)
    compare_manifests(saved_physics, core.get_physics_manifest())
    circuit = core.circuit
    inventory, owners = _inventory(circuit)
    identity = {
        "scenario_id": core.entry["id"],
        "split": core.entry["split"],
        "scenario_index": core._episode_index,
        "dataset_sha256": core.dataset.sha256,
        "physics_model_sha256": circuit.model_sha256,
    }
    if (
        identity["split"] != split
        or identity["dataset_sha256"] != core.config.data.manifest_sha256
        or identity["scenario_index"] != index
        or not identity["scenario_id"]
    ):
        raise ValueError("native scenario identity, split, or manifest mismatch")
    if paired:
        identity["reset_policy_input_sha256"] = reset_policy_input_sha256(core)
    initial = final = _states(circuit)
    snapshots = [{"step": 0}]
    samples = [{"step": 0, "devices": measure_devices(circuit)}]
    for step in range(1, core.horizon + 1):
        info = _step(core, native)
        if info.get("physical_failure") is True:
            snapshots.append({"step": step, "physical_failure": True})
            break
        if info.get("physics_model_sha256") != identity["physics_model_sha256"]:
            raise RuntimeError("native step physics digest changed")
        final = _states(circuit)
        if final["capacitors"] != initial["capacitors"]:
            raise RuntimeError("baseline capacitor states changed")
        for name, reg in circuit.regulators.items():
            if not native or (reg.transformer.lower(), reg.tap_winding) not in owners:
                if final["tap_pu"][name] != initial["tap_pu"][name]:
                    raise RuntimeError("baseline observed an unowned tap change")
        for name in circuit.batteries:
            if any(abs(float(p)) > 1e-6 for p in circuit.powers(f"Storage.{name}")):
                raise RuntimeError("baseline storage is not electrically idle")
        snapshots.append({"step": step, "info": info})
        samples.append({"step": step, "devices": measure_devices(circuit)})

    records = _normalize(snapshots, identity)
    metrics = summarize_episode(
        records,
        core.config.max_episode_steps,
        voltage_lower_limit_pu=core.config.voltage_min,
        voltage_upper_limit_pu=core.config.voltage_max,
    )
    return {
        **identity,
        "mode": MODES[int(native)],
        "seed": seed,
        "dss_control_mode": "STATIC" if native else "OFF",
        "regcontrol_inventory": inventory,
        "dss_owned_regulator_count": len(owners) if native else 0,
        "initial_device_state": initial,
        "last_converged_device_state": final,
        "records": records,
        "metrics": metrics,
        **device_output(samples, records, metrics),
    }


def export_baselines(resolved_config, physics_manifest, manifest, indices, *, seed,
                     split="validation", _paired=False):
    if split != "validation" and not (split == "test" and _paired is True):
        raise ValueError("only validation is enabled by default")
    indices = tuple(indices)
    if (
        not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("provide distinct nonnegative validation indices")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    manifest = Path(manifest).resolve(strict=True)
    digest = sha256(manifest.read_bytes()).hexdigest()
    config, make_env, compare_manifests = _runtime(resolved_config)
    data = config.data
    if config.system_name != "34Bus" or config.smartds_series:
        raise ValueError("only classic 34Bus is audited for these baselines")
    if data.split != "train" or not data.path or data.manifest_sha256 != digest:
        raise ValueError(
            "training config must bind a frozen train manifest with matching SHA256"
        )
    saved_physics = json.loads(Path(physics_manifest).read_text(encoding="utf-8"))
    config = replace(config, data=replace(data, path=str(manifest), split=split))
    episodes = []
    for index in indices:
        for native in (False, True):
            core = make_env(config)
            try:
                episodes.append(
                    _run(core, saved_physics, compare_manifests, index, seed, native, split=split, paired=_paired)
                )
            finally:
                core.close()
        hold, local = episodes[-2:]
        for key in ("scenario_id", "initial_device_state", "regcontrol_inventory"):
            if hold[key] != local[key]:
                raise ValueError(f"paired baseline {key} disagrees after reset")
    return {"split": split, "dataset_sha256": digest, "episodes": episodes}


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _find_sidecar(folder, name):
    for parent in (folder, folder.parent, folder.parent.parent):
        candidate = parent / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"checkpoint is missing {name}")


def _checkpoint_inputs(folder, algorithm):
    import yaml

    files = [_find_sidecar(folder, n) for n in ("config.resolved.yaml", "physics.manifest.json")]
    config = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    physics = json.loads(files[1].read_text(encoding="utf-8"))
    if (config.get("config_kind"), config["run"]["algo"]) != ("resolved", algorithm):
        raise ValueError(f"expected the frozen {algorithm} training checkpoint")
    if algorithm == "mappo" and config["algo"].get("share_param") is not False:
        raise ValueError("paired MAPPO must be explicitly nonshared")
    actors = sorted(folder.glob("actor_agent*.pt"))
    if not actors:
        raise ValueError("checkpoint folder has no actors")
    seed = config["seed"]["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid frozen algorithm training seed")
    return files, config, physics, seed, {p.name: _sha(p) for p in (*files, *actors)}


def _verified_checkpoint_report(details, expected):
    files, config, physics, seed, hashes = details
    if expected != hashes:
        raise ValueError("checkpoint snapshot, physics, or actor digests differ")
    train = config["train"]
    return {"training_seed": seed, "checkpoint_files_sha256": dict(hashes),
            "configured_training": {k: train[k] for k in
                                    ("num_env_steps", "episode_length", "n_rollout_threads")}}


def _trace_key(values):
    # Ignore absolute timestamp labels: relabeling the same exogenous episode is
    # not holdout independence. IEEE floats normalize equivalent CSV spellings.
    return sha256(json.dumps([[float(v).hex() if v != 0 else "0" for v in row[1:]]
                              for row in values], separators=(",", ":")).encode()).hexdigest()


def _paired_sources(manifest, data, steps, split, indices):
    from envs.powergrid.scenario import ScenarioDataset

    wanted = data.get("scenario_names")
    step_seconds = data["step_seconds"]
    selected = ScenarioDataset(manifest, split, steps, step_seconds, wanted)
    if split == "test":
        by_trace = {}
        # Audit signal overlap only; this does not score/access policies against
        # any test trajectory until all checkpoint and manifest bindings match.
        for group in ("train", "validation", "test"):
            dataset = ScenarioDataset(manifest, group, steps, step_seconds)
            for entry, values in dataset.episodes:
                key = _trace_key(values)
                if key in by_trace and by_trace[key] != group:
                    raise ValueError("reused effective exogenous trace across dataset splits")
                by_trace[key] = group
    if max(indices) >= len(selected.episodes):
        raise ValueError("scenario index outside the frozen split")
    return [selected.episodes[i][0]["id"] for i in indices]


def _check_pair(rows, identity):
    fields = ("split", "seed", "scenario_id", "scenario_index",
              "dataset_sha256", "physics_model_sha256")
    for row in rows:
        if any(row.get(key) != identity[key] for key in fields):
            raise ValueError("paired scenario/clock/model identity differs")
    resets = [row.get("reset_policy_input_sha256") for row in rows]
    if not resets[0] or any(x != resets[0] for x in resets[1:]):
        raise ValueError("paired measured reset observation/action masks differ")
    starts = [row["device_endpoints"][0] for row in rows]
    if any(x != starts[0] for x in starts[1:]):
        raise ValueError("paired measured reset endpoints differ")
    # Same coverage for every admissible arm; failed episodes remain returned
    # and cannot qualify even when another arm completes.
    clocks = []
    for row in rows:
        ticks = [record["step_seconds"] for record in row["records"]
                 if record.get("dss_converged") is True]
        if not row["metrics"]["eligible_for_day_ranking"]:
            return False
        clocks.append(ticks)
    if any(clock != clocks[0] for clock in clocks[1:]):
        raise ValueError("paired native step durations differ")
    return True


def run_paired_plan(path):
    from utils.physics_manifest import compare_manifests
    from scripts.export_vvc_validation import export_validation

    repo = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repo:
        raise ValueError("run paired acceptance from its own repository root")
    if any(_sha(repo / file) != expected for file, expected in STOCK_34_CLOSURE.items()):
        raise ValueError("shared stock ieee34 Redirect closure differs")
    path = Path(path).resolve(strict=True)
    raw_plan = path.read_bytes()
    plan = json.loads(raw_plan)
    base = path.parent
    split = plan["split"]
    seeds, indices = plan["environment_seeds"], plan["scenario_indices"]
    if split not in ("validation", "test"):
        raise ValueError("paired replay must explicitly name validation or test")
    for values, limit in ((seeds, 2**32), (indices, math.inf)):
        if not values or any(type(x) is not int or not 0 <= x < limit for x in values) or len(set(values)) != len(values):
            raise ValueError("provide distinct nonnegative evaluation seeds and scenario indices")
    manifest = (base / plan["manifest"]).resolve(strict=True)
    manifest_sha = _sha(manifest)
    if manifest_sha != plan["manifest_sha256"]:
        raise ValueError("paired scenario manifest digest differs")
    arm_info, checkpoint_bindings = {}, {}
    for algo in ("happo", "mappo"):
        spec = plan["checkpoints"][algo]
        folder = (base / spec["path"]).resolve(strict=True)
        details = _checkpoint_inputs(folder, algo)
        checkpoint_bindings[algo] = _verified_checkpoint_report(details, spec["sha256"])
        arm_info[algo] = (folder, *details)
    left, right = (arm_info[x] for x in ("happo", "mappo"))
    compare_manifests(left[3], right[3])
    if any(not left[3].get(k) for k in ("model_sha256", "action_schema_sha256", "observation_schema_sha256")):
        raise ValueError("missing paired physics/action/observation binding")
    envs = [deepcopy(arm[2]["env"]) for arm in (left, right)]
    for config in envs:
        if (config["system_name"], config["dss_folder"], config["dss_file"]) != (
            "34Bus", "node_systems/34Bus", "network.dss"
        ) or config.get("smartds_series") or left[3]["overlays"]:
            raise ValueError("paired scope is stock classic IEEE34 only, not 34Bus_PV or another overlay")
        config.pop("seed", None)  # algorithm initialization, not shared evaluation seed
        data = config["data"]
        if data["split"] != "train" or data["manifest_sha256"] != manifest_sha or not data.get("path"):
            raise ValueError("each checkpoint must bind the same frozen training manifest")
        data.pop("path")  # identical frozen digest can be mounted at a different path
    if envs[0] != envs[1]:
        raise ValueError("paired environment and action/observation conditions differ")
    steps = envs[0]["max_episode_steps"]
    if steps * envs[0]["data"]["step_seconds"] != 24 * 3600:
        raise ValueError("paired acceptance expects a complete 24-hour scenario")
    ppo = plan["checkpoints"].get("ppo")
    if ppo is not None:
        if split != "validation":
            raise ValueError("five-arm PPO acceptance only enables validation")
        prefix = (base / ppo["path"]).resolve().with_suffix("")
        names = {prefix.with_suffix(suffix).name for suffix in
                 (".zip", ".environment.yaml", ".physics.json", ".sb3.json")}
        hashes = ppo["sha256"]
        if set(hashes) != names or any(_sha(prefix.parent / n) != d for n, d in hashes.items()):
            raise ValueError("PPO checkpoint or sidecar digests differ")
        import yaml
        ppo_cfg = yaml.safe_load(prefix.with_suffix(".environment.yaml").read_text())
        ppo_env = deepcopy(ppo_cfg["env"])
        ppo_env.pop("seed", None)
        ppo_env["data"].pop("path", None)
        if ppo_env != envs[0]:
            raise ValueError("PPO environment differs from the paired frozen environment")
        compare_manifests(left[3], json.loads(prefix.with_suffix(".physics.json").read_text()))
        metadata = json.loads(prefix.with_suffix(".sb3.json").read_text())
        checkpoint_bindings["ppo"] = {"training_seed": metadata["training_seed"],
                                      "actual_training_steps": metadata["steps"],
                                      "checkpoint_files_sha256": hashes}
    chosen = _paired_sources(manifest, envs[0]["data"], steps, split, indices)
    results = []
    for seed in seeds:
        hold = export_baselines(*left[1], manifest, indices, seed=seed, split=split,
                                _paired=True)["episodes"]
        policy = {algo: export_validation(arm_info[algo][0], manifest, indices, seed=seed,
                                         split=split, allow_mappo=algo == "mappo", _paired=True)["episodes"]
                  for algo in ("happo", "mappo")}
        if ppo is not None:
            policy["ppo"] = export_validation(prefix, manifest, indices, seed=seed,
                                               split=split, sb3=True, _paired=True)["episodes"]
        if len(hold) != 2*len(indices) or any(len(x) != len(indices) for x in policy.values()):
            raise ValueError("paired runner omitted or added an evaluation episode")
        for i, (index, scenario_id) in enumerate(zip(indices, chosen)):
            baseline = hold[2*i:2*i+2]
            if tuple(row["mode"] for row in baseline) != MODES:
                raise ValueError("paired baseline mode order differs")
            arms = (*MODES, *policy)
            rows = [*baseline, *(episodes[i] for episodes in policy.values())]
            identity = dict(seed=seed, split=split, scenario_id=scenario_id, scenario_index=index,
                            dataset_sha256=manifest_sha, physics_model_sha256=left[3]["model_sha256"])
            comparable = _check_pair(rows, identity)
            results.append({**identity, "all_four_full_day": _check_pair(rows[:4], identity),
                            "arms": dict(zip(arms, rows)),
                            **({"all_five_full_day": comparable} if ppo is not None else {})})
    return {"split": split, "plan_sha256": sha256(raw_plan).hexdigest(), "dataset_sha256": manifest_sha,
            "node_case": "ieee34", "training_seeds": {k: v["training_seed"] for k,v in checkpoint_bindings.items()},
            "environment_seeds": seeds, "checkpoints": checkpoint_bindings,
            "topology": {k: left[3][k] for k in ("model_sha256", "action_schema_sha256",
                                                "observation_schema_sha256", "master")},
            "stock_closure_sha256": STOCK_34_CLOSURE,
            "split_scope": ("synthetic/real effective trace disjointness checked, no selection"
                            if split == "test" else "validation acceptance; no holdout claim"),
            "results": results}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--paired-plan":
        parser = argparse.ArgumentParser(description="Run a fixed classic IEEE34 paired replay plan")
        parser.add_argument("--paired-plan", type=Path, required=True)
        args = parser.parse_args(argv)
        try:
            output = run_paired_plan(args.paired_plan)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            print(f"VVC paired replay failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(output, sort_keys=True, allow_nan=False))
        return 0 if all(row["all_four_full_day"] for row in output["results"]) else 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--physics-manifest", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenario-index", type=int, required=True, action="append")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        output = export_baselines(
            args.resolved_config,
            args.physics_manifest,
            args.manifest,
            args.scenario_index,
            seed=args.seed,
        )
        print(json.dumps(output, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"VVC baseline export failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
