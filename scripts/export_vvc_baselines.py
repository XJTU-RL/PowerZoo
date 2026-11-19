"""Export IEEE 34Bus hold-initial and DSS static-regulator validation baselines."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

from scripts.export_vvc_validation import _normalize
from scripts.vvc_benchmark_metrics import summarize_episode
from scripts.vvc_device_metrics import device_output, measure_devices


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


def _run(core, saved_physics, compare_manifests, index, seed, native):
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
        identity["split"] != "validation"
        or identity["dataset_sha256"] != core.config.data.manifest_sha256
        or identity["scenario_index"] != index
        or not identity["scenario_id"]
    ):
        raise ValueError("native scenario identity, split, or manifest mismatch")
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


def export_baselines(resolved_config, physics_manifest, manifest, indices, *, seed):
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
    config = replace(config, data=replace(data, path=str(manifest), split="validation"))
    episodes = []
    for index in indices:
        for native in (False, True):
            core = make_env(config)
            try:
                episodes.append(
                    _run(core, saved_physics, compare_manifests, index, seed, native)
                )
            finally:
                core.close()
        hold, local = episodes[-2:]
        for key in ("scenario_id", "initial_device_state", "regcontrol_inventory"):
            if hold[key] != local[key]:
                raise ValueError(f"paired baseline {key} disagrees after reset")
    return {"split": "validation", "dataset_sha256": digest, "episodes": episodes}


def main(argv=None):
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
