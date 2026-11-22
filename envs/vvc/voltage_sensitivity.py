"""SHOM 的同工况 V-Q 雅可比和逐设备原生电压响应。"""

import numpy as np

from envs.powergrid.scenario import apply_operating_row
from envs.powergrid.session import DSSSession


def voltage_reactive_sensitivity(j_qv, j_qtheta, j_ptheta, j_pv):
    """Compute dV/dQ at fixed P from consistently ordered Jacobian blocks."""
    qv, qtheta, ptheta, pv = (
        np.asarray(block, dtype=np.float64)
        for block in (j_qv, j_qtheta, j_ptheta, j_pv)
    )
    if (
        qv.ndim != 2
        or ptheta.ndim != 2
        or qv.shape[0] != qv.shape[1]
        or ptheta.shape[0] != ptheta.shape[1]
        or not all(qv.shape)
        or not all(ptheta.shape)
        or qtheta.shape != (qv.shape[0], ptheta.shape[0])
        or pv.shape != (ptheta.shape[0], qv.shape[0])
        or not all(np.isfinite(block).all() for block in (qv, qtheta, ptheta, pv))
    ):
        raise ValueError("V-Q sensitivity requires compatible, finite Jacobian blocks")
    try:
        reduced = qv - qtheta @ np.linalg.solve(ptheta, pv)
        return np.linalg.solve(reduced, np.eye(qv.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "V-Q sensitivity is unavailable for a singular Jacobian"
        ) from exc


def polar_power_jacobian(admittance, voltage, voltage_base, free):
    """Return fixed-reference Jacobian blocks for S = V * conjugate(Y V).

    Voltage columns are derivatives with respect to per-unit magnitude. Power
    rows are derivatives in VA and angle columns are in radians.
    """
    current = admittance @ voltage
    unit = voltage / np.abs(voltage)
    angular = 1j * voltage
    magnitude = unit * voltage_base

    def derivative(dv):
        return (
            np.diag(dv * current.conj())
            + voltage[:, None] * admittance.conj() * dv.conj()[None, :]
        )[np.ix_(free, free)]

    by_angle = derivative(angular)
    by_magnitude = derivative(magnitude)
    return by_magnitude.imag, by_angle.imag, by_angle.real, by_magnitude.real


def _as_complex(raw, size):
    array = np.asarray(raw)
    if array.size == size and np.iscomplexobj(array):
        return array.reshape(-1)
    if array.size == 2 * size and not np.iscomplexobj(array):
        pairs = array.reshape(-1, 2)
        return pairs[:, 0] + 1j * pairs[:, 1]
    raise ValueError("OpenDSS returned an unexpected nodal complex array shape")


def _connected_nodes(session, element, terminals=None):
    connection = session.connection(element)
    if not connection["enabled"]:
        raise ValueError(f"No sensitivity binding for disabled device {element}")
    if terminals is None:
        terminals = range(len(connection["buses"]))
    width = connection["conductors"]
    result = []
    for terminal in terminals:
        bus = connection["buses"][terminal].split(".", 1)[0].lower()
        nodes = connection["nodes"][terminal * width : (terminal + 1) * width]
        opened = connection["open"][terminal]
        result.extend(
            f"{bus}.{node}"
            for node, is_open in zip(nodes, opened)
            if node and not is_open
        )
    return sorted(set(result))


def native_reactive_matrix(session, action_schema):
    """Read the converged nodal solution and construct Eq. (27) without a new solve."""
    circuit = session.native
    if not circuit.Solution.Converged:
        raise ValueError("A converged OpenDSS solution is required for sensitivity")
    node_order = [str(node).lower() for node in circuit.YNodeOrder]
    size = len(node_order)
    voltage = _as_complex(circuit.YNodeVarray, size)
    # OpenDSS exports the system matrix in column-major order.
    admittance = _as_complex(circuit.SystemY, size * size).reshape(
        (size, size), order="F"
    )
    if (
        not np.isfinite(voltage).all()
        or not np.isfinite(admittance).all()
        or np.any(voltage == 0)
    ):
        raise ValueError("OpenDSS sensitivity requires finite, energized nodes")

    sources = [
        name for name in circuit.AllElementNames if name.lower().startswith("vsource.")
    ]
    slack = set()
    for source in sources:
        if session.connection(source)["enabled"]:
            slack.update(_connected_nodes(session, source, terminals=[0]))
    if not slack or not slack <= set(node_order):
        raise ValueError("OpenDSS sensitivity requires an explicit slack-node binding")
    free = [i for i, node in enumerate(node_order) if node not in slack]
    nodes = [node_order[i] for i in free]

    voltage_base = {}
    for node in node_order:
        bus = node.rsplit(".", 1)[0]
        if bus not in voltage_base:
            if circuit.SetActiveBus(bus) < 0:
                raise ValueError(f"No voltage base for node {node}")
            voltage_base[bus] = float(circuit.ActiveBus.kVBase) * 1000
    base_vector = np.array(
        [voltage_base[node.rsplit(".", 1)[0]] for node in node_order]
    )
    if np.any(base_vector <= 0) or not np.isfinite(base_vector).all():
        raise ValueError("OpenDSS sensitivity requires positive voltage bases")

    device_nodes = {}
    available = set(nodes)
    for binding in action_schema:
        device = binding["element"]
        # The regulator is represented at both transformer terminals, matching
        # the devices-to-phase-nodes convention of the authors' published artifact.
        element = binding.get("actuator", {}).get("element", device)
        mapped = _connected_nodes(session, element)
        if not mapped or not set(mapped) <= available:
            raise ValueError(
                f"No complete non-reference sensitivity binding for {device}"
            )
        device_nodes[device] = mapped

    blocks = polar_power_jacobian(admittance, voltage, base_vector, free)
    # Convert pu/var to pu/Mvar using a common power base for the ranking.
    return nodes, device_nodes, 1e6 * voltage_reactive_sensitivity(*blocks)


def _voltages(session):
    circuit = session.native
    nodes = [node.lower() for node in circuit.AllNodeNames]
    volts = np.asarray(circuit.AllBusVmagPu, dtype=np.float64)
    if not nodes or len(set(nodes)) != len(nodes) or volts.shape != (len(nodes),):
        raise ValueError("SHOM 电压读数与节点编号不一致")
    if not np.isfinite(volts).all():
        raise ValueError("SHOM 电压读数包含非有限值")
    return dict(zip(nodes, volts))


def _perturbations(env, device, original):
    kind, name = device.split(".", 1)
    if kind == "Capacitor":
        yield "state", 1 - original["state"], 1.0
    elif kind == "RegControl":
        reg = env.circuit.regulators[name]
        span = reg.maxtap - reg.mintap
        index = round((original["tap"] - reg.mintap) / span * (env.reg_act_num - 1))
        neighbor = index + 1 if index < env.reg_act_num - 1 else index - 1
        yield "tap", reg.mintap + span * neighbor / (env.reg_act_num - 1), span
    elif kind == "Storage":
        rated = env.circuit.batteries[name].max_kw
        base = original["kw"] / rated
        if np.isinf(env.bat_act_num):
            lower, upper = max(-1, base - 0.05), min(1, base + 0.05)
        else:
            grid = np.linspace(-1, 1, int(env.bat_act_num))
            lower = max(grid[grid < base - 1e-10], default=base)
            upper = min(grid[grid > base + 1e-10], default=base)
        for candidate in (lower, upper) if base > 0 else (upper, lower):
            if not np.isclose(candidate, base, rtol=0, atol=1e-10):
                yield "kw", float(candidate * rated), rated
    else:
        raise ValueError(f"SHOM native 不支持设备 {device}")


def _measured_command(session, device, field):
    _, name = device.split(".", 1)
    if field == "state":
        return session.capacitor_state(name)[0]
    if field == "tap":
        return session.regulator_tap(name)
    return float(-session.powers(device)[0])


class NativeVoltageProbe:
    """在独立 DSS 上重建已求解的工况，每次只扰动一个真实设备指令。"""

    def __init__(self, live):
        self.session = DSSSession(
            live.source,
            live.master,
            live.overlays,
            live.step_seconds,
            live.svg_elements,
        )
        if self.session.model_sha256 != live.model_sha256:
            self.close()
            raise ValueError("SHOM native 副本与运行模型的摘要不一致")

    def close(self):
        self.session.close()

    def measure(self, env, device_ids):
        live, probe = env.circuit, self.session
        if not live.native.Solution.Converged:
            raise ValueError("SHOM native 需要已收敛的工况")
        expected = _voltages(live)
        nodes = sorted(expected)
        probe.reset(seconds=live.time_s)
        apply_operating_row(
            probe, env.current_row, env.dataset, env.config.data.load_groups
        )
        for name, dev in live.batteries.items():
            probe.command(f"Edit Storage.{name} %stored={dev.soc * 100:.16g}")
            probe.batteries[name].soc = dev.soc

        requests = {device: live.trace[device]["requested"] for device in device_ids}
        baseline_trace = probe.apply(requests)
        probe.solve()
        voltages = _voltages(probe)
        if set(voltages) != set(nodes):
            raise ValueError("SHOM native 副本与运行模型的节点不一致")
        baseline = np.array([voltages[node] for node in nodes])
        if not np.allclose(
            baseline, [expected[node] for node in nodes], rtol=0, atol=1e-7
        ):
            raise ValueError("SHOM native 副本无法重现原始工况电压")

        columns, applied_probes = [], {}
        for device in device_ids:
            original = baseline_trace[device]["applied"]
            baseline_measured = None
            for field, requested, scale in _perturbations(env, device, original):
                base = original[field]
                if baseline_measured is None:
                    baseline_measured = _measured_command(probe, device, field)
                trace = probe.apply({device: {field: requested}})
                applied = trace[device]["applied"][field]
                delta = (applied - base) / scale
                if abs(delta) <= 1e-10:
                    continue
                probe.solve()
                measured = _measured_command(probe, device, field)
                if field != "kw" and not np.isclose(
                    measured, applied, rtol=0, atol=1e-7
                ):
                    raise ValueError(f"SHOM native 未观测到 {device} 的设备动作")
                volts = _voltages(probe)
                if set(volts) != set(nodes):
                    raise ValueError(f"SHOM native 扰动 {device} 后节点发生变化")
                columns.append(
                    (np.array([volts[node] for node in nodes]) - baseline) / delta
                )
                applied_probes[device] = {
                    "field": field,
                    "baseline": base,
                    "requested": requested,
                    "applied": applied,
                    "measured_before": baseline_measured,
                    "measured_after": measured,
                    "normalized_delta": delta,
                }
                break
            else:
                raise ValueError(f"SHOM native 对 {device} 无可用的归一化扰动")
            probe.apply({device: requests[device]})
            probe.solve()
            restored = _voltages(probe)
            if set(restored) != set(nodes) or not np.allclose(
                [restored[node] for node in nodes], baseline, rtol=0, atol=1e-7
            ):
                raise ValueError(f"SHOM native 扰动 {device} 后无法恢复副本工况")
        return nodes, np.column_stack(columns), applied_probes
