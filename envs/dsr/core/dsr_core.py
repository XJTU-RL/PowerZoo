"""有源静态恢复：永久故障、真实开关/负荷、相供电和实际服务计量。"""

from hashlib import sha256
import json
from pathlib import Path

import gymnasium as gym
import networkx as nx
import numpy as np

from envs.dsr.core.config import DSRConfig
from envs.dsr.core.scenario import RestorationDataset
from envs.powergrid.adapters import parse_actions
from envs.powergrid.session import DSSSession
from envs.powergrid.topology import (
    energized_phases,
    is_closed,
    operating_graph,
    source_nodes,
    terminal_nodes,
)


class DSRCoreEnv:
    def __init__(self, config, project_root=None):
        if not isinstance(config, DSRConfig):
            raise TypeError("恢复核心需要唯一加载器构造的DSRConfig")
        self.config, self.horizon = config, config.max_episode_steps
        self.dataset = RestorationDataset(config.data)
        root = Path(project_root or Path(__file__).resolve().parents[3])
        model = Path(config.dss_folder or f"node_systems/{config.system_name}")
        if not model.is_absolute():
            model = root / model
        self.circuit = DSSSession(
            model, config.dss_file, config.overlays, config.data.step_seconds
        )
        self._closed = False
        try:
            self._bind()
            self.agent_spaces = [
                gym.spaces.Discrete(len(choices)) for choices in self.choices
            ]
            self.action_space = gym.spaces.Tuple(self.agent_spaces)
            self.seed(config.seed)
            obs = self.reset(load_profile_idx=0)
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, obs.shape, dtype=np.float32
            )
        except Exception:
            self.close()
            raise

    def _bind(self):
        c = self.circuit
        supported = {
            "vsource",
            "line",
            "transformer",
            "load",
            "pvsystem",
            "storage",
            "capacitor",
            "regcontrol",
            "capcontrol",
            "monitor",
            "energymeter",
        }
        self.elements = {e.lower(): e for e in c.native.AllElementNames}
        unknown = [
            e
            for e in self.elements.values()
            if e.split(".")[0].lower() not in supported
        ]
        if unknown or any(len(s) != 1 for s in c.capacitors.values()):
            raise ValueError(f"静态恢复尚未验收这些设备/控制语义：{unknown}")
        self.loads = {
            self.elements["load." + n]: (d.kw, d.kvar) for n, d in c.loads.items()
        }
        if (
            any(p < 0 for p, q in self.loads.values())
            or not sum(p for p, q in self.loads.values()) > 0
        ):
            raise ValueError("恢复负荷需要非负且总计为正的原生kW基准")
        self.base_load_kw = sum(p for p, q in self.loads.values())
        self.weights = {e: 1.0 for e in self.loads}
        for element, value in self.config.priorities.items():
            canonical = self.elements.get(element.lower())
            if canonical not in self.loads:
                raise ValueError(f"优先级没有实际Load消费者：{element}")
            self.weights[canonical] = value
        self.reference_kwh = sum(
            self.weights[e] * p for e, (p, q) in self.loads.items()
        )
        if self.reference_kwh <= 0:
            raise ValueError("加权负荷基准必须为正")
        groups = self.config.control_groups
        if groups:
            self.group_names = [g.name for g in groups]
            controls = [g.controls for g in groups]
        else:
            switches = [
                e
                for e in self.elements.values()
                if e.startswith("Line.")
                and c.read_property(e, "switch").lower() in ("yes", "true")
            ]
            controls, self.group_names = [sorted(switches) + list(self.loads)], [
                "central_dispatcher"
            ]
        self.controls, self.choices = [], []
        for group in controls:
            choices = [None]
            for element in group:
                canonical = self.elements.get(element.lower())
                if canonical is None or canonical in self.controls:
                    raise ValueError(f"控制设备不存在或重复归属：{element}")
                if canonical.startswith("Line."):
                    if c.read_property(canonical, "switch").lower() not in (
                        "yes",
                        "true",
                    ):
                        raise ValueError(
                            f"{canonical}未声明物理开关，不能将普通线路冒充开关"
                        )
                elif canonical not in self.loads:
                    raise ValueError(f"恢复首版只控制真实开关和Load：{canonical}")
                self.controls.append(canonical)
                choices.extend([(canonical, False), (canonical, True)])
            self.choices.append(choices)
        for entry in self.dataset.episodes:
            for element in entry["unavailable"]:
                canonical = self.elements.get(element.lower())
                if canonical is None or canonical.split(".")[0] not in (
                    "Line",
                    "Transformer",
                    "Vsource",
                ):
                    raise ValueError(f"故障未绑定真实支路/电压源：{element}")

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self._next_episode = int(self.rng.integers(len(self.dataset.episodes)))
        for i, s in enumerate(self.agent_spaces):
            s.seed((seed + i) % 2**32)

    def reset(self, load_profile_idx=None, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        index = self._next_episode if load_profile_idx is None else load_profile_idx
        if not 0 <= index < len(self.dataset.episodes):
            raise ValueError("故障场景编号越界")
        self._next_episode = (index + 1) % len(self.dataset.episodes)
        self.entry = self.dataset.episodes[index]
        self.t, self.operations, self.invalid_actions, self.reward = 0, 0, 0, 0.0
        c = self.circuit
        c.reset()
        self.demand = {
            e: p * self.entry["load_multiplier"] for e, (p, q) in self.loads.items()
        }
        c.apply(
            {
                e: {
                    "kw": p * self.entry["load_multiplier"],
                    "kvar": q * self.entry["load_multiplier"],
                }
                for e, (p, q) in self.loads.items()
            }
        )
        # 预故障原生调压整定，不推进时间；恢复期间冻结机械控制。
        c.solve(native_controls=True)
        self.original_der_enabled = {
            e: c.connection(e)["enabled"]
            for e in self.elements.values()
            if e.startswith(("PVSystem.", "Storage."))
        }
        self.unavailable = {self.elements[e.lower()] for e in self.entry["unavailable"]}
        c.enable_elements({e: False for e in self.unavailable})
        for e in self.unavailable:
            if e in self.controls and e.startswith("Line."):
                c.apply({e: {"closed": 0}})
        self._settle()
        if not self.graph.graph["radial"]:
            raise ValueError("恢复初始网络不满足声明的母线级径向约束")
        return self.get_obs()

    def positions(self):
        return {
            e: (
                is_closed(self.circuit.connection(e))
                if e.startswith("Line.")
                else self.circuit.connection(e)["enabled"]
            )
            for e in self.controls
        }

    def _legal(self, proposed):
        if any(
            e not in self.controls
            or type(v) is not bool
            or (v and e in self.unavailable)
            for e, v in proposed.items()
        ):
            return False
        switches = {e: v for e, v in proposed.items() if e.startswith("Line.")}
        return not switches or operating_graph(self.circuit, switches).graph["radial"]

    def available_actions(self):
        current = self.positions()
        masks = []
        for choices in self.choices:
            mask = [1.0]
            mask.extend(
                float(current[e] != value and self._legal({e: value}))
                for e, value in choices[1:]
            )
            masks.append(np.array(mask, dtype=np.float32))
        return masks

    def _apply_positions(self, positions):
        # 零时间整帧下发也遵循先开后合；所有数值/设备校验仍在写入前完成。
        ordered = sorted(
            positions.items(),
            key=lambda item: (item[1], not item[0].startswith("Line."), item[0]),
        )
        self.circuit.apply(
            {
                e: {"closed" if e.startswith("Line.") else "enabled": int(value)}
                for e, value in ordered
            }
        )

    def _settle(self):
        reached = energized_phases(self.circuit)
        states = {
            e: original
            and set(terminal_nodes(self.circuit.connection(e), 0)) <= reached
            for e, original in self.original_der_enabled.items()
        }
        self.circuit.enable_elements(
            {
                e: value
                for e, value in states.items()
                if self.circuit.connection(e)["enabled"] != value
            }
        )
        self.circuit.solve()
        self._observe()

    def _observe(self):
        self.physical_state = self.circuit.read_state()
        self.energized = energized_phases(self.circuit)
        self.graph = operating_graph(self.circuit)
        roots = sorted({bus for bus, phase in source_nodes(self.circuit)})
        self.load_metrics = {}
        for element, demand in self.demand.items():
            connection = self.circuit.connection(element)
            needed = set(terminal_nodes(connection, 0))
            bus = connection["buses"][0].split(".")[0].lower()
            p, q = self.circuit.powers(element)
            if p < -1e-5:
                raise RuntimeError(
                    f"{element}出现非负荷方向的有功，不能取绝对值伪装供电"
                )
            delivered = min(demand, max(0.0, float(p)))
            supplied = bool(needed) and needed <= self.energized
            voltage_ok = all(
                phase in self.physical_state.voltages_pu[b]
                and self.config.voltage_min
                <= self.physical_state.voltages_pu[b][phase]
                <= self.config.voltage_max
                for b, phase in needed
            )
            capacity_ok = False
            for root in roots:
                if not nx.has_path(self.graph, root, bus):
                    continue
                path = nx.shortest_path(self.graph, root, bus)
                elements = [
                    e
                    for a, b in zip(path, path[1:])
                    for e in self.graph[a][b]["elements"]
                ]
                if all(
                    self.physical_state.branches[e]["loading_pu"] is not None
                    and self.physical_state.branches[e]["loading_pu"] <= 1 + 1e-6
                    for e in elements
                ):
                    capacity_ok = True
            qualified = (
                connection["enabled"] and supplied and voltage_ok and capacity_ok
            )
            self.load_metrics[element] = {
                "requested_kw": demand,
                "delivered_kw": delivered,
                "qualified_served_kw": delivered if qualified else 0.0,
                "enabled": connection["enabled"],
                "source_connected": supplied,
                "voltage_ok": voltage_ok,
                "capacity_ok": capacity_ok,
                "priority": self.weights[element],
            }
        self.obs = {
            "bus_voltages": {
                b: list(v.values()) for b, v in self.physical_state.voltages_pu.items()
            },
            "time": self.t,
        }

    def step(self, action):
        if self.t >= self.horizon:
            raise RuntimeError("恢复回合已结束，请reset")
        parts = parse_actions(action, self.agent_spaces)
        masks = self.available_actions()
        proposed = {}
        invalid = 0
        for choices, part, mask in zip(self.choices, parts, masks):
            index = int(part[0])
            if not mask[index]:
                invalid += 1
            if choices[index] is not None:
                e, value = choices[index]
                proposed[e] = value
        if not invalid and not self._legal(proposed):
            invalid = len(proposed)
        # 同步整帧拒绝，不能因agent顺序不同而静默执行不同的部分动作。
        if invalid:
            proposed = {}
        self.circuit.begin_interval(self.t * self.config.data.step_seconds)
        self._apply_positions(proposed)
        self._settle()
        self.circuit.finish_interval()
        self.t += 1
        self.operations, self.invalid_actions = len(proposed), invalid
        self._observe()
        info = self._info()
        dt = self.config.data.step_seconds / 3600
        energized_v = [
            v
            for b, values in self.physical_state.voltages_pu.items()
            for phase, v in values.items()
            if (b, phase) in self.energized
        ]
        violations = np.maximum(
            self.config.voltage_min - np.array(energized_v), 0
        ) + np.maximum(np.array(energized_v) - self.config.voltage_max, 0)
        loadings = [
            v["loading_pu"]
            for v in self.physical_state.branches.values()
            if v["loading_pu"] is not None
        ]
        voltage = float(np.mean(violations**2)) if len(violations) else 0.0
        thermal = (
            float(np.mean(np.maximum(np.array(loadings) - 1, 0) ** 2))
            if loadings
            else 0.0
        )
        self.reward = -(
            info["weighted_service_unserved_kwh"] / self.reference_kwh
            + self.config.operation_cost * self.operations
            + self.config.invalid_action_cost * self.invalid_actions
            + dt
            * (self.config.voltage_cost * voltage + self.config.thermal_cost * thermal)
        )
        info["reward_total"] = self.reward
        return self.get_obs(), float(self.reward), self.t == self.horizon, info

    def evaluate_positions(self, positions):
        """专用参考规划器的零时间试算，恢复原状态；不得按多个区间累计其功率。"""
        if not self._legal(positions):
            return None
        saved = self.positions()
        try:
            self._apply_positions(positions)
            self._settle()
            return self._info()
        finally:
            self._apply_positions(saved)
            self._settle()

    def _info(self):
        state = self.physical_state
        dt = self.config.data.step_seconds / 3600
        requested = sum(self.demand.values())
        delivered = sum(m["delivered_kw"] for m in self.load_metrics.values())
        served = sum(m["qualified_served_kw"] for m in self.load_metrics.values())
        values = [v for phases in state.voltages_pu.values() for v in phases.values()]
        live = [
            v
            for b, phases in state.voltages_pu.items()
            for p, v in phases.items()
            if (b, p) in self.energized
        ]
        loadings = [
            b["loading_pu"]
            for b in state.branches.values()
            if b["loading_pu"] is not None
        ]
        return {
            "step": self.t,
            "time_s": state.time_s,
            "step_seconds": self.config.data.step_seconds,
            "scenario_id": self.entry["id"],
            "scenario": self.entry["id"],
            "dataset_sha256": self.dataset.sha256,
            "unavailable": sorted(self.unavailable),
            "positions": self.positions(),
            "loads": self.load_metrics,
            "delivered_kw": delivered,
            "qualified_served_kw": served,
            "requested_kw": requested,
            "delivered_ratio": delivered / requested if requested else 1.0,
            "restored_ratio": served / requested if requested else 1.0,
            "unserved_kwh": max(0, requested - delivered) * dt,
            "weighted_service_unserved_kwh": sum(
                self.weights[e] * (m["requested_kw"] - m["qualified_served_kw"])
                for e, m in self.load_metrics.items()
            )
            * dt,
            "operations": self.operations,
            "invalid_actions": self.invalid_actions,
            "reward_total": self.reward,
            "total_load_kw": float(self.circuit.total_load_power()[0]),
            "total_power_kw": state.grid_import[0],
            "power_loss_kw": state.losses[0],
            "pv_power_kw": sum(
                p[0] for e, p in state.injections.items() if e.startswith("PVSystem.")
            ),
            "voltage_min_pu": min(values) if values else None,
            "voltage_max_pu": max(values) if values else None,
            "inactive_phase_count": sum(
                len(phases) for phases in self.circuit.bus_phase_nodes.values()
            )
            - len(values),
            "voltage_violation_count": sum(
                v < self.config.voltage_min or v > self.config.voltage_max for v in live
            ),
            "deenergized_phase_count": len(values) - len(live),
            "max_branch_loading_pu": max(loadings) if loadings else None,
            "branch_overload_count": sum(v > 1 + 1e-6 for v in loadings),
            "active_balance_error_kw": state.active_balance_error_kw,
            "capacitor_ctrl": 0.0,
            "regulator_ctrl": 0.0,
            "dss_converged": bool(self.circuit.native.Solution.Converged),
            "control_trace": dict(self.circuit.trace),
            "terminated": self.t == self.horizon,
            "truncated": False,
            "TimeLimit.truncated": False,
        }

    def _fields(self):
        fields = {}
        for bus, phases in self.circuit.bus_phase_nodes.items():
            for phase in phases:
                present = phase in self.physical_state.voltages_pu[bus]
                value = self.physical_state.voltages_pu[bus].get(phase, 0.0)
                fields[f"{bus}.{phase}.voltage_pu"] = value
                fields[f"{bus}.{phase}.in_solution"] = float(present)
                fields[f"{bus}.{phase}.source_connected"] = float(
                    (bus, phase) in self.energized
                )
        for element in sorted(self.elements.values()):
            if element.startswith(("Line.", "Transformer.", "Vsource.")):
                fields[element + ".available"] = float(
                    self.circuit.connection(element)["enabled"]
                )
        for element, position in self.positions().items():
            fields[element + ".position"] = float(position)
        for element, meter in self.load_metrics.items():
            fields.update(
                {
                    element
                    + "."
                    + key: float(value)
                    / (self.base_load_kw if key.endswith("_kw") else 1)
                    for key, value in meter.items()
                }
            )
        for name, reg in self.circuit.regulators.items():
            fields["RegControl." + name + ".tap_pu"] = self.circuit.regulator_tap(name)
        for name in self.circuit.capacitors:
            fields["Capacitor." + name + ".state"] = self.circuit.capacitor_state(name)[
                0
            ]
        for name, bat in self.circuit.batteries.items():
            fields["Storage." + name + ".soc"] = bat.soc
        for name, pv in self.circuit.pvs.items():
            fields["PVSystem." + name + ".p_rated"] = (
                -self.circuit.powers("PVSystem." + name)[0] / pv.pmpp
            )
        fields["episode_progress"] = self.t / self.horizon
        return fields

    def get_obs(self):
        result = np.array(list(self._fields().values()), dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("恢复观测非有限，不能补默认值")
        return result

    def get_physics_manifest(self):
        schema = [
            {
                "group": name,
                "choices": [
                    (
                        None
                        if choice is None
                        else {"element": choice[0], "closed_or_enabled": choice[1]}
                    )
                    for choice in choices
                ],
            }
            for name, choices in zip(self.group_names, self.choices)
        ]
        fields = list(self._fields())
        digest = lambda value: sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()
        return {
            **self.circuit.manifest(),
            "task": type(self.config).__name__,
            "action_schema": schema,
            "action_schema_sha256": digest(schema),
            "observation_schema": fields,
            "observation_schema_sha256": digest(fields),
        }

    def close(self):
        if not self._closed:
            self._closed = True
            self.circuit.close()
