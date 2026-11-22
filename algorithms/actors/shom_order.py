"""Order selection for SHOM when same-point physical sensitivities are available."""

import numpy as np


def physical_agent_scores(mode, transitions, device_ids, model_hash, action_hash):
    """Return device scores from one complete episode, bound to its physical inputs."""
    if not transitions or not device_ids or len(set(device_ids)) != len(device_ids):
        raise ValueError(
            "SHOM requires a complete episode and unique physical device IDs"
        )
    expected_devices = set(device_ids)
    step_scores, rewards = [], []
    for reward, info in transitions:
        sample = info.get("voltage_sensitivity")
        if info.get("physical_failure") or not isinstance(sample, dict):
            detail = info.get("voltage_sensitivity_unavailable") or info.get(
                "failure_reason"
            )
            raise ValueError(
                f"SHOM sensitivity is unavailable for this transition: {detail or 'no response'}"
            )
        if (
            sample.get("mode") != mode
            or not model_hash
            or not action_hash
            or sample.get("model_sha256") != model_hash
            or sample.get("action_schema_sha256") != action_hash
            or not info.get("scenario_id")
            or sample.get("scenario_id") != info["scenario_id"]
            or "time_s" not in info
            or sample.get("time_s") != info["time_s"]
        ):
            raise ValueError(
                "SHOM sensitivity does not match the transition's model, devices or time"
            )
        matrix = np.asarray(sample["matrix"], dtype=np.float64)
        if matrix.ndim != 2 or not all(matrix.shape) or not np.isfinite(matrix).all():
            raise ValueError("SHOM needs a finite voltage sensitivity matrix")
        if mode == "paper":
            nodes = sample.get("nodes", [])
            mapping = sample.get("device_nodes", {})
            if matrix.shape != (len(nodes), len(nodes)) or len(set(nodes)) != len(
                nodes
            ):
                raise ValueError("SHOM paper mode requires a unique row per node")
            if set(mapping) != expected_devices:
                raise ValueError(
                    "SHOM paper mode requires an unambiguous node for every device"
                )
            row_totals = dict(zip(nodes, np.abs(matrix).sum(axis=1)))
            scores = []
            for device in device_ids:
                device_nodes = mapping[device]
                if isinstance(device_nodes, str):
                    device_nodes = [device_nodes]
                if (
                    not device_nodes
                    or len(set(device_nodes)) != len(device_nodes)
                    or not set(device_nodes) <= row_totals.keys()
                ):
                    raise ValueError(f"Invalid SHOM paper node mapping for {device}")
                scores.append(sum(row_totals[node] for node in device_nodes))
            step_scores.append(scores)
        elif mode == "native":
            columns = sample.get("device_columns", [])
            nodes = sample.get("nodes", [])
            if (
                sample.get("control_basis") != "one_normalized_device_command"
                or len(nodes) != matrix.shape[0]
                or len(set(nodes)) != len(nodes)
                or len(columns) != matrix.shape[1]
                or set(columns) != expected_devices
                or len(set(columns)) != len(columns)
            ):
                raise ValueError(
                    "SHOM native mode requires one normalized response column per device"
                )
            column_totals = np.abs(matrix).sum(axis=0)
            step_scores.append(
                [column_totals[columns.index(device)] for device in device_ids]
            )
        else:
            raise ValueError(f"Unknown SHOM sensitivity mode: {mode}")
        rewards.append(reward)

    rewards = np.asarray(rewards, dtype=np.float64)
    if not np.isfinite(rewards).all():
        raise ValueError("SHOM order requires finite rewards")
    if mode == "paper":
        # (28) absolute row sum, (29) signed reward divided by total reward.
        total = rewards.sum()
        if (
            not np.isfinite(total)
            or abs(total) <= np.finfo(float).eps * np.abs(rewards).sum()
        ):
            raise ValueError(
                "SHOM paper reward weights are undefined for zero total reward"
            )
        return (rewards / total) @ np.asarray(step_scores)
    # The native extension averages response to a common normalized device command.
    return np.mean(step_scores, axis=0)


def physical_agent_order(scores, reverse=False, device_ids=None):
    """Preserve paper ordering and break native ties by stable device ID."""
    if device_ids is not None:
        direction = 1 if reverse else -1
        return sorted(
            range(len(scores)), key=lambda i: (direction * scores[i], device_ids[i])
        )
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[::-1] if reverse else order
