"""同一交流网络上的区域DER协同调度；没有虚拟交换或第二份SOC。"""

from hashlib import sha256
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from envs.district_dispatch.core.config import DistrictDispatchConfig
from envs.district_dispatch.core.metering import region_meters
from envs.powergrid.adapters import parse_actions
from envs.powergrid.scenario import prepare_dataset, apply_operating_row
from envs.powergrid.session import DSSSession


class DistrictDispatchCore:
    def __init__(self, config, project_root=None):
        if not isinstance(config, DistrictDispatchConfig):
            raise TypeError("District需要已校验的DistrictDispatchConfig")
        self.config = config
        self.horizon = config.max_episode_steps
        self.dataset = prepare_dataset(config.data, self.horizon)
        root = Path(project_root or Path(__file__).resolve().parents[3])
        system = Path(config.dss_folder or f"node_systems/{config.system_name}")
        if not system.is_absolute():
            system = root / system
        self.dss_folder_path, self.dss_file = str(system.resolve()), config.dss_file
        svg = [
            e for r in config.regions for e in r.controls if e.startswith("Generator.")
        ]
        self.circuit = DSSSession(
            system, config.dss_file, config.overlays, config.data.step_seconds, svg
        )
        self._closed = False
        try:
            self._bind_regions()
            self.agent_spaces = [
                gym.spaces.Box(
                    -1,
                    1,
                    (sum(self._dimension(e) for e in r.controls),),
                    dtype=np.float32,
                )
                for r in config.regions
            ]
            self.action_space = gym.spaces.Tuple(self.agent_spaces)
            self.seed(config.seed)
            initial = self.reset(load_profile_idx=0)
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, initial.shape, dtype=np.float32
            )
            self.local_observation_space = gym.spaces.Box(
                -np.inf, np.inf, self.get_local_obs()[0].shape, dtype=np.float32
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _dimension(element):
        return 2 if element.startswith("PVSystem.") else 1

    def _bind_regions(self):
        supported = {
            "line",
            "transformer",
            "load",
            "vsource",
            "isource",
            "capacitor",
            "pvsystem",
            "storage",
            "generator",
            "regcontrol",
            "capcontrol",
            "monitor",
            "energymeter",
        }
        unknown = [
            e
            for e in self.circuit.native.AllElementNames
            if e.split(".", 1)[0].lower() not in supported
        ]
        if unknown:
            raise ValueError(
                f"区域首版尚未验收这些元件/自动控制器，不能静默忽略：{unknown}"
            )
        if any(len(states) != 1 for states in self.circuit.capacitors.values()):
            raise ValueError("区域首版尚未验收多级电容状态，不能压成单个二状态")
        regions = self.config.regions
        if not regions:
            raise ValueError("必须显式指定区域母线与控制设备")
        self.owners, self.control_owners = {}, {}
        for i, region in enumerate(regions):
            for bus in region.buses:
                if bus != bus.lower() or bus in self.owners:
                    raise ValueError(f"区域母线重复或未规范为小写：{bus}")
                self.owners[bus] = i
            for element in region.controls:
                if element in self.control_owners:
                    raise ValueError(f"控制设备重复归属：{element}")
                self.control_owners[element] = i
        if set(self.owners) != set(self.circuit.bus_names):
            raise ValueError(
                f"区域必须恰好覆盖原生母线：{set(self.owners)^set(self.circuit.bus_names)}"
            )
        expected = set(
            ["PVSystem." + n for n in self.circuit.pvs]
            + ["Storage." + n for n in self.circuit.batteries]
            + ["Generator." + n for n in self.circuit.svgs]
        )
        if set(self.control_owners) != expected:
            raise ValueError(
                f"区域控制必须恰好绑定全部PV/Storage及显式SVG：{set(self.control_owners)^expected}"
            )
        for element, region in self.control_owners.items():
            bus = self.circuit.element_info(element)["buses"][0].split(".")[0].lower()
            if self.owners[bus] != region:
                raise ValueError(f"{element}不在其声明区域")
        generators = {
            e.lower()
            for e in self.circuit.native.AllElementNames
            if e.lower().startswith("generator.")
        }
        if generators != {("Generator." + n).lower() for n in self.circuit.svgs}:
            raise ValueError("禁止虚拟交换或未建模发电机；Generator仅限显式SVG")
        unknown = set(self.config.data.load_groups) - set(self.circuit.loads)
        if unknown:
            raise ValueError(f"负荷分组包含未知设备：{unknown}")
        self.base_load_kw = sum(d.kw for d in self.circuit.loads.values())
        if self.base_load_kw <= 0:
            raise ValueError("需要正的原生负荷基准")
        self.region_base_kw = np.zeros(len(regions))
        for name, load in self.circuit.loads.items():
            bus = (
                self.circuit.element_info("Load." + name)["buses"][0]
                .split(".")[0]
                .lower()
            )
            self.region_base_kw[self.owners[bus]] += load.kw
        self.region_base_kw = np.maximum(self.region_base_kw, 1.0)
        self.neighbors = [set() for _ in regions]
        for _, _, edge in self.circuit.topology.edges(data=True):
            buses = self.circuit.element_info(edge["element"])["buses"]
            owners = {self.owners[b.split(".")[0].lower()] for b in buses}
            for owner in owners:
                self.neighbors[owner].update(owners - {owner})
        self.neighbors = [sorted(values) for values in self.neighbors]

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self._next_episode = int(self.rng.integers(len(self.dataset.episodes)))
        for i, space in enumerate(self.agent_spaces):
            space.seed((seed + i) % 2**32)

    def _apply_data(self, index):
        self.current_row = row = self.profile[index]
        apply_operating_row(
            self.circuit, row, self.dataset, self.config.data.load_groups
        )

    def reset(self, load_profile_idx=None, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        index = self._next_episode if load_profile_idx is None else load_profile_idx
        if not 0 <= index < len(self.dataset.episodes):
            raise ValueError("场景编号越界")
        self._next_episode = (index + 1) % len(self.dataset.episodes)
        self.entry, self.profile = self.dataset.episodes[index]
        self.t, self.costs = 0, {}
        self.control_changes = (0.0, 0.0)
        self.normalized_actions = {}
        self.circuit.reset(float(self.profile[0, 0]))
        self._apply_data(0)
        self.available_pv_kw = self.circuit.available_pv_power()
        self.circuit.solve(native_controls=True)
        self.initial_energy = self._energy()
        self._observe()
        return self.get_obs()

    def _energy(self):
        return {
            name: float(self.circuit.read_property("Storage." + name, "kWhStored"))
            for name in self.circuit.batteries
        }

    def step(self, action):
        if self.t >= self.horizon:
            raise RuntimeError("回合已结束，请reset")
        parts = parse_actions(action, self.agent_spaces)
        normalized = {}
        for region, part in zip(self.config.regions, parts):
            offset = 0
            for element in region.controls:
                size = self._dimension(element)
                normalized[element] = part[offset : offset + size]
                offset += size
        before = self._energy()
        self.normalized_actions = {
            element: values.tolist() for element, values in normalized.items()
        }
        old_caps = {
            n: self.circuit.capacitor_state(n)[0] for n in self.circuit.capacitors
        }
        old_taps = {n: self.circuit.regulator_tap(n) for n in self.circuit.regulators}
        self.interval_start_s = float(self.profile[self.t, 0])
        self.circuit.begin_interval(self.interval_start_s)
        self._apply_data(self.t)
        requests = {}
        for element, values in normalized.items():
            name = element.split(".", 1)[1]
            value = float(values[0])
            if element.startswith("Storage."):
                requests[element] = {"kw": value * self.circuit.batteries[name].max_kw}
            elif element.startswith("Generator."):
                dev = self.circuit.svgs[name]
                requests[element] = {
                    "kvar": value * (dev.kvar_max if value >= 0 else dev.kvar_max_abs)
                }
        self.circuit.apply(requests)
        self.available_pv_kw = self.circuit.available_pv_power()
        requests = {}
        for element, values in normalized.items():
            if not element.startswith("PVSystem."):
                continue
            name = element.split(".", 1)[1]
            pv = self.circuit.pvs[name]
            kw = (float(values[0]) + 1) / 2 * self.available_pv_kw[name]
            qcommand = float(values[1])
            qlimit = min(
                np.sqrt(max(0, pv.kva**2 - kw**2)),
                pv.kvar_max if qcommand >= 0 else pv.kvar_max_abs,
            )
            requests[element] = {"kw": kw, "kvar": qcommand * qlimit}
        self.circuit.apply(requests)
        self.circuit.solve(native_controls=True)
        self.circuit.finish_interval()
        self.control_changes = (
            sum(
                abs(self.circuit.capacitor_state(n)[0] - old)
                for n, old in old_caps.items()
            ),
            sum(
                abs(self.circuit.regulator_tap(n) - old)
                / (
                    self.circuit.regulators[n].maxtap
                    - self.circuit.regulators[n].mintap
                )
                * self.circuit.regulators[n].numtaps
                for n, old in old_taps.items()
            ),
        )
        self.t += 1
        self._observe()
        self.costs = self._costs(before)
        reward = -sum(self.costs.values()) / self.config.costs.scale_yuan
        if not np.isfinite(reward):
            raise RuntimeError("调度成本或奖励非有限")
        info = self._info()
        info.update(
            reward_total=float(reward),
            total_cost_yuan=float(sum(self.costs.values())),
            **self.costs,
        )
        return self.get_obs(), float(reward), self.t == self.horizon, info

    def _observe(self):
        self.physical_state = self.circuit.read_state()
        self.meters, self.crossing_losses = region_meters(
            self.circuit, self.physical_state, self.owners, len(self.config.regions)
        )
        self.injections = {
            e.lower(): p for e, p in self.physical_state.injections.items()
        }
        self.obs = {
            "bus_voltages": {
                n: list(v.values()) for n, v in self.physical_state.voltages_pu.items()
            },
            "time": self.t,
        }

    def _costs(self, before):
        cfg, state = self.config, self.physical_state
        c, dt = cfg.costs, cfg.data.step_seconds / 3600
        buy, sell = cfg.market.price_at(self.interval_start_s)
        import_kwh = max(0, state.grid_import[0]) * dt
        export_kwh = max(0, -state.grid_import[0]) * dt
        phases = np.array(
            [v for values in state.voltages_pu.values() for v in values.values()]
        )
        violation = np.maximum(cfg.voltage_min - phases, 0) + np.maximum(
            phases - cfg.voltage_max, 0
        )
        loadings = [
            b["loading_pu"]
            for b in state.branches.values()
            if b["loading_pu"] is not None
        ]
        energy = self._energy()
        curtailed = (
            sum(
                max(0, available - self.injections["pvsystem." + name][0])
                for name, available in self.available_pv_kw.items()
            )
            * dt
        )
        return {
            "grid_bill_yuan": import_kwh * buy - export_kwh * sell,
            "carbon_cost_yuan": import_kwh
            * cfg.market.carbon_intensity_kg_kwh
            * cfg.market.carbon_price_yuan_kg,
            "battery_wear_cost_yuan": sum(abs(energy[n] - before[n]) for n in energy)
            * c.battery_wear_yuan_kwh,
            "curtailment_cost_yuan": curtailed * c.curtailment_yuan_kwh,
            "voltage_cost_yuan": float(np.mean(violation**2))
            * dt
            * c.voltage_yuan_pu2_h,
            "thermal_cost_yuan": (
                float(np.mean(np.maximum(np.array(loadings) - 1, 0) ** 2))
                * dt
                * c.thermal_yuan_pu2_h
                if loadings
                else 0.0
            ),
            "terminal_energy_cost_yuan": (
                sum(abs(energy[n] - self.initial_energy[n]) for n in energy)
                * c.terminal_energy_yuan_kwh
                if self.t == self.horizon
                else 0.0
            ),
        }

    def _local_fields(self, i):
        region, meter = self.config.regions[i], self.meters[i]
        voltages = [
            v
            for bus in region.buses
            for v in self.physical_state.voltages_pu[bus].values()
        ]
        base = self.region_base_kw[i]
        fields = {
            "voltage_min_pu": min(voltages),
            "voltage_mean_pu": float(np.mean(voltages)),
            "voltage_max_pu": max(voltages),
        }
        for key in ("load", "net_import"):
            fields.update(
                {
                    key + "_p_base": meter[key][0] / base,
                    key + "_q_base": meter[key][1] / base,
                }
            )
        for element in region.controls:
            name = element.split(".", 1)[1]
            p, q = self.injections[element.lower()]
            if element.startswith("PVSystem."):
                pv = self.circuit.pvs[name]
                fields.update(
                    {
                        element + ".p_rated": p / pv.pmpp,
                        element + ".q_rated": q / pv.kva,
                        element
                        + ".available_rated": self.available_pv_kw[name] / pv.pmpp,
                    }
                )
            elif element.startswith("Storage."):
                bat = self.circuit.batteries[name]
                fields.update(
                    {element + ".soc": bat.soc, element + ".p_rated": p / bat.max_kw}
                )
            else:
                fields[element + ".q_rated"] = q / self.circuit.svgs[name].kva
        if self.config.use_neighbor_obs:
            for neighbor in self.neighbors[i]:
                buses = self.config.regions[neighbor].buses
                values = [
                    v
                    for b in buses
                    for v in self.physical_state.voltages_pu[b].values()
                ]
                prefix = "neighbor." + self.config.regions[neighbor].name
                fields[prefix + ".voltage_mean_pu"] = float(np.mean(values))
                fields[prefix + ".net_import_base"] = (
                    self.meters[neighbor]["net_import"][0]
                    / self.region_base_kw[neighbor]
                )
        buy, sell = self.config.market.price_at(self.circuit.time_s)
        fields.update(
            buy_yuan_kwh=buy,
            sell_yuan_kwh=sell,
            time_sin=float(np.sin(2 * np.pi * self.circuit.time_s / 86400)),
            time_cos=float(np.cos(2 * np.pi * self.circuit.time_s / 86400)),
            episode_progress=self.t / self.horizon,
        )
        return fields

    def get_local_obs(self):
        fields = [self._local_fields(i) for i in range(len(self.config.regions))]
        size = max(map(len, fields))
        return [
            np.pad(np.array(list(f.values()), dtype=np.float32), (0, size - len(f)))
            for f in fields
        ]

    def _global_fields(self):
        fields = {
            f"bus.{bus}.phase{phase}.voltage_pu": value
            for bus, phases in self.physical_state.voltages_pu.items()
            for phase, value in phases.items()
        }
        fields.update(
            {
                "Capacitor." + n + ".state": self.circuit.capacitor_state(n)[0]
                for n in self.circuit.capacitors
            }
        )
        fields.update(
            {
                "RegControl." + n + ".tap_pu": self.circuit.regulator_tap(n)
                for n in self.circuit.regulators
            }
        )
        for i, region in enumerate(self.config.regions):
            fields.update(
                {
                    region.name + "." + key: value
                    for key, value in self._local_fields(i).items()
                }
            )
        fields.update(
            irradiance_kw_m2=self.current_row[4],
            array_temperature_c_div100=self.current_row[5] / 100,
        )
        fields.update(
            {
                "load_multiplier." + str(i): value
                for i, value in enumerate(self.current_row[1:4])
            }
        )
        return fields

    def get_obs(self):
        values = np.array(list(self._global_fields().values()), dtype=np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError("调度观测非有限")
        return values

    def _info(self):
        state, cfg = self.physical_state, self.config
        phases = [v for values in state.voltages_pu.values() for v in values.values()]
        loadings = [
            b["loading_pu"]
            for b in state.branches.values()
            if b["loading_pu"] is not None
        ]
        start = self.interval_start_s if self.t else float(self.profile[0, 0])
        buy, sell = cfg.market.price_at(start)
        return {
            "step": self.t,
            "time_s": state.time_s,
            "step_seconds": cfg.data.step_seconds,
            "interval_start_s": start,
            "buy_price_yuan_kwh": buy,
            "sell_price_yuan_kwh": sell,
            "total_power_kw": state.grid_import[0],
            "total_power_kvar": state.grid_import[1],
            "total_load_kw": sum(m["load"][0] for m in self.meters),
            "power_loss_kw": state.losses[0],
            "pv_power_kw": sum(
                p[0] for e, p in self.injections.items() if e.startswith("pvsystem.")
            ),
            "voltage_min_pu": min(phases),
            "voltage_max_pu": max(phases),
            "voltage_violation_count": sum(
                v < cfg.voltage_min or v > cfg.voltage_max for v in phases
            ),
            "max_branch_loading_pu": max(loadings) if loadings else None,
            "branch_overload_count": sum(v > 1 for v in loadings),
            "unrated_branch_count": len(state.branches) - len(loadings),
            "active_balance_error_kw": state.active_balance_error_kw,
            "regions": {r.name: meter for r, meter in zip(cfg.regions, self.meters)},
            "crossing_losses_kw_kvar": self.crossing_losses,
            "storage_energy_kwh": self._energy(),
            "storage_soc": state.storage_soc,
            "scenario_id": self.entry["id"],
            "scenario": self.entry["scenario"],
            "dataset_sha256": self.dataset.sha256,
            "physics_model_sha256": self.circuit.model_sha256,
            "control_trace": {
                k: v for k, v in self.circuit.trace.items() if not k.startswith("Load.")
            },
            "normalized_actions": self.normalized_actions,
            "dss_converged": bool(self.circuit.native.Solution.Converged),
            "capacitor_ctrl": self.control_changes[0],
            "regulator_ctrl": self.control_changes[1],
            "terminated": self.t == self.horizon,
            "truncated": False,
            "TimeLimit.truncated": False,
        }

    def get_physics_manifest(self):
        schema = [
            {
                "region": r.name,
                "controls": [
                    {
                        "element": e,
                        "components": (
                            ["available_command", "q_fraction"]
                            if e.startswith("PVSystem.")
                            else (
                                ["p_fraction"]
                                if e.startswith("Storage.")
                                else ["q_fraction"]
                            )
                        ),
                    }
                    for e in r.controls
                ],
                "space": str(s),
            }
            for r, s in zip(self.config.regions, self.agent_spaces)
        ]
        local = [list(self._local_fields(i)) for i in range(len(self.config.regions))]
        size = max(map(len, local))
        observation = {
            "critic": list(self._global_fields()),
            "actors": [
                names + [f"padding.{n}" for n in range(size - len(names))]
                for names in local
            ],
            "regions": [
                {"name": r.name, "buses": r.buses} for r in self.config.regions
            ],
        }
        digest = lambda value: sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()
        return {
            **self.circuit.manifest(),
            "task": type(self.config).__name__,
            "action_schema": schema,
            "action_schema_sha256": digest(schema),
            "observation_schema": observation,
            "observation_schema_sha256": digest(observation),
        }

    def close(self):
        if not self._closed:
            self._closed = True
            self.circuit.close()
