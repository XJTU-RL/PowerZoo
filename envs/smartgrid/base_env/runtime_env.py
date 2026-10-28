"""唯一仿真内核；单/多智能体适配器共用数据、动作、DSS、奖励与时间语义。"""

from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import gymnasium as gym
from envs.smartgrid.base_env.env_config import SmartGridConfig
from envs.smartgrid.circuit_system.runtime import SmartGridCircuit
from envs.smartgrid.data_process.dataset import prepare_dataset
from envs.smartgrid.rewards.vvc_reward import VVCReward
from envs.smartgrid.rewards.lagrangian import LagrangianUpdater


class Env(gym.Env):
    def __init__(self, folder_path, config_or_info, dss_act=False):
        self.config = (
            config_or_info
            if isinstance(config_or_info, SmartGridConfig)
            else SmartGridConfig.from_env_args(config_or_info)
        )
        cfg = self.config
        if dss_act:
            raise ValueError("新内核由策略控制设备；dss_act/useS 尚无正式适配")
        self.dataset = prepare_dataset(cfg.data, cfg.max_episode_steps)
        system = Path(cfg.dss_folder or f"node_systems/{cfg.system_name}")
        if not system.is_absolute():
            system = Path(folder_path) / system
        self.circuit = SmartGridCircuit(
            system, cfg.dss_file, cfg.pv_plan, cfg.data.step_seconds, cfg.svg_elements
        )
        unknown_loads = set(cfg.data.load_groups) - set(self.circuit.loads)
        if unknown_loads:
            self.circuit.close()
            raise ValueError(f"data.load_groups 包含不存在的 DSS 负荷：{unknown_loads}")
        self.dss_folder_path, self.dss_file = str(system.resolve()), cfg.dss_file
        self.horizon = cfg.max_episode_steps
        self.reg_act_num, self.bat_act_num, self.pv_act_num = cfg.RBP_act_num
        if any(
            reg.numtaps % (self.reg_act_num - 1)
            for reg in self.circuit.regulators.values()
        ):
            self.circuit.close()
            raise ValueError(
                "regulator.action_num 的间隔必须对齐 DSS NumTaps 的实际档位"
            )
        self.pv_control_enabled = cfg.pv_control_enabled
        self.cap_names, self.reg_names, self.bat_names, self.pv_names = map(
            list,
            (
                self.circuit.capacitors,
                self.circuit.regulators,
                self.circuit.batteries,
                self.circuit.pvs,
            ),
        )
        self.cap_num, self.reg_num, self.bat_num, self.pv_num = map(
            len, (self.cap_names, self.reg_names, self.bat_names, self.pv_names)
        )
        self.svg_names = list(self.circuit.svgs)
        self.all_bus_names, self.topology = (
            self.circuit.bus_names,
            self.circuit.topology,
        )
        self.useS, self.use_render = False, False
        self.wrap_observation, self.observe_load = True, False
        self.agent_spaces = (
            [gym.spaces.Discrete(2) for _ in self.cap_names]
            + [gym.spaces.Discrete(self.reg_act_num) for _ in self.reg_names]
            + [self._space(self.bat_act_num, 1) for _ in self.bat_names]
            + (
                [self._space(self.pv_act_num, 2) for _ in self.pv_names]
                if self.pv_control_enabled
                else []
            )
            + [self._space(float("inf"), 1) for _ in self.svg_names]
        )
        self.base_load_kw = sum(load.kw for load in self.circuit.loads.values())
        if self.base_load_kw <= 0:
            self.circuit.close()
            raise ValueError("SmartGrid需要正的原生负荷基准")
        if any(len(states) != 1 for states in self.circuit.capacitors.values()):
            self.circuit.close()
            raise ValueError("SmartGrid当前只支持单级电容器")
        self.action_space = gym.spaces.Tuple(self.agent_spaces)
        self.reward_func = VVCReward(self, cfg)
        self.use_cmdp = cfg.use_cmdp
        self.lagrangian_updater = (
            LagrangianUpdater(
                init_lambda=cfg.lambda_init,
                lr=cfg.lambda_lr,
                target_cost=cfg.target_cost,
                lambda_max=cfg.lambda_max,
                lambda_min=cfg.lambda_min,
                update_strategy=cfg.lambda_update_strategy,
                momentum=cfg.lambda_momentum,
                adaptive_lr=cfg.lambda_adaptive_lr,
            )
            if cfg.use_cmdp
            else None
        )
        self._closed = False
        self.seed(cfg.seed)
        self.reset(load_profile_idx=0)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=self.wrap_obs(self.obs).shape, dtype=np.float32
        )

    @staticmethod
    def _space(n, dimension):
        return (
            gym.spaces.Box(-1, 1, (dimension,), dtype=np.float32)
            if n == float("inf")
            else gym.spaces.Discrete(int(n))
        )

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self._next_episode = int(self.rng.integers(len(self.dataset.episodes)))
        for i, space in enumerate(self.agent_spaces):
            space.seed((seed + i) % 2**32)

    def _apply_data(self, index):
        row = self.profile[index]
        requests = {}
        for name, base in self.circuit.loads.items():
            group = self.dataset.group_for(name, self.config.data.load_groups)
            ratio = float(row[1 + group])
            requests["Load." + name] = {
                "kw": base.kw * ratio,
                "kvar": base.kvar * ratio,
            }
        self.circuit.apply(requests)
        self.circuit.set_pv_conditions(float(row[4]), float(row[5]))
        self.current_row = row

    def _pv_action(self, name, action):
        pv = self.circuit.pvs[name]
        if self.pv_act_num == float("inf"):
            fraction, qcommand = (float(action[0]) + 1) / 2, float(action[1])
        else:
            fraction, qcommand = float(action[0]) / (self.pv_act_num - 1), 0.0
        kw = fraction * self.available_pv_kw[name]
        headroom = np.sqrt(max(0, pv.kva**2 - kw**2))
        qlimit = min(headroom, pv.kvar_max if qcommand >= 0 else pv.kvar_max_abs)
        q = qcommand * qlimit
        diff = abs(pv.fraction - fraction) + abs(pv.q_fraction - qcommand)
        pv.fraction, pv.q_fraction = fraction, qcommand
        return {"kw": kw, "kvar": q}, diff

    def _storage_action(self, name, action):
        fraction = (
            float(action[0])
            if self.bat_act_num == float("inf")
            else 2 * float(action[0]) / (self.bat_act_num - 1) - 1
        )
        return {"kw": fraction * self.circuit.batteries[name].max_kw}

    def _actions(self, action):
        values = np.asarray(action, dtype=float).reshape(-1)
        expected = sum(
            1 if isinstance(s, gym.spaces.Discrete) else s.shape[0]
            for s in self.agent_spaces
        )
        if len(values) != expected or not np.isfinite(values).all():
            raise ValueError(f"动作应有 {expected} 个有限数值，收到 {values}")
        parts, i = [], 0
        for space in self.agent_spaces:
            n = 1 if isinstance(space, gym.spaces.Discrete) else space.shape[0]
            part, i = values[i : i + n], i + n
            if isinstance(space, gym.spaces.Discrete):
                if part[0] != int(part[0]) or not 0 <= part[0] < space.n:
                    raise ValueError(f"无效离散动作 {part[0]}，空间 {space}")
            elif (np.abs(part) > 1.00001).any():
                raise ValueError(f"连续动作超出 [-1,1]：{part}")
            parts.append(
                np.clip(part, -1, 1)
                if not isinstance(space, gym.spaces.Discrete)
                else part
            )
        return parts

    def step(self, action):
        if self.t >= self.horizon:
            raise RuntimeError("回合已结束，请先 reset")
        parts = iter(self._actions(action))
        self.circuit.begin_interval(float(self.profile[self.t, 0]))
        self._apply_data(self.t)
        capdiff, regdiff, pvdiff = [], [], []
        requests = {}
        for name in self.cap_names:
            old = self.circuit.capacitor_state(name)[0]
            value = int(next(parts)[0])
            requests["Capacitor." + name] = {"state": value}
            capdiff.append(abs(value - old))
        for name, reg in self.circuit.regulators.items():
            index = int(next(parts)[0])
            tap = reg.mintap + index / (self.reg_act_num - 1) * (
                reg.maxtap - reg.mintap
            )
            regdiff.append(
                abs(tap - self.circuit.regulator_tap(name)) / (reg.maxtap - reg.mintap)
            )
            requests["RegControl." + name] = {"tap": tap}
        for name in self.bat_names:
            requests["Storage." + name] = self._storage_action(name, next(parts))
        self.circuit.apply(requests)
        # 可用AC功率来自同一引擎的无时间探测，保留Pmpp、效率和温度基准。
        self.available_pv_kw = self.circuit.available_pv_power()
        requests = {}
        if self.pv_control_enabled:
            for name in self.pv_names:
                request, diff = self._pv_action(name, next(parts))
                requests["PVSystem." + name] = request
                pvdiff.append(diff)
        for name in self.svg_names:
            value = float(next(parts)[0])
            dev = self.circuit.svgs[name]
            requests["Generator." + name] = {
                "kvar": value * (dev.kvar_max if value >= 0 else dev.kvar_max_abs)
            }
        self.circuit.apply(requests)
        self.circuit.solve()
        self.circuit.finish_interval()
        self.applied_control_vector = self._control_vector()
        soc_err, dis_err = [], []
        for name, bat in self.circuit.batteries.items():
            power = -self.circuit.powers("Storage." + name)[0]
            soc_err.append(abs(bat.soc - bat.initial_soc))
            dis_err.append(max(0, power) / bat.max_kw)
        self.t += 1
        self._observe()
        reward, info = self.reward_func.composite_reward(
            capdiff, regdiff, soc_err, dis_err, pvdiff
        )
        self.episode_costs.append(info["cost_voltage"])
        if self.lagrangian_updater:
            info["lambda"] = self.lagrangian_updater.lmbda
            if self.config.use_lagrangian_reward:
                info["reward_before_lagrangian"] = reward
                reward -= self.lagrangian_updater.lmbda * info["cost_voltage"]
        done = self.t == self.horizon
        if done and self.lagrangian_updater:
            self.lagrangian_updater.update(float(np.mean(self.episode_costs)))
        info.update(self._info())
        info.update(
            reward_total=float(reward),
            capacitor_ctrl=float(sum(capdiff)),
            regulator_ctrl=float(
                sum(
                    d * reg.numtaps
                    for d, reg in zip(regdiff, self.circuit.regulators.values())
                )
            ),
        )
        # 场景是完整的有限时域任务，进度在观测中；自然终止，不是外部时间截断。
        info.update(terminated=done, truncated=False)
        info["TimeLimit.truncated"] = False
        if not np.isfinite(reward):
            raise RuntimeError("奖励出现非有限值")
        return self.wrap_obs(self.obs), float(reward), done, info

    def _control_vector(self):
        """各控制量的实际下发设定，统一用于平滑成本；不对动作变化量再作差分。"""
        trace, values = self.circuit.trace, []
        for name in self.cap_names:
            values.append(trace["Capacitor." + name]["applied"]["state"])
        for name, reg in self.circuit.regulators.items():
            tap = trace["RegControl." + name]["applied"]["tap"]
            values.append((tap - reg.mintap) / (reg.maxtap - reg.mintap))
        for name, bat in self.circuit.batteries.items():
            values.append(trace["Storage." + name]["applied"]["kw"] / bat.max_kw)
        if self.pv_control_enabled:
            for name, pv in self.circuit.pvs.items():
                applied = trace["PVSystem." + name]["applied"]
                values.extend(
                    [
                        applied["kw"] / max(self.available_pv_kw[name], 1e-9),
                        applied["kvar"] / pv.kva,
                    ]
                )
        for name, svg in self.circuit.svgs.items():
            values.append(trace["Generator." + name]["applied"]["kvar"] / svg.kva)
        return values

    def _observe(self):
        self.physical_state = self.circuit.read_state()
        powers = {n.lower(): v for n, v in self.physical_state.injections.items()}
        self.obs = {
            "bus_voltages": {
                n: list(v.values()) for n, v in self.physical_state.voltages_pu.items()
            },
            "cap_statuses": {
                n: self.circuit.capacitor_state(n)[0] for n in self.cap_names
            },
            "reg_statuses": {
                n: (self.circuit.regulator_tap(n) - r.mintap) / (r.maxtap - r.mintap)
                for n, r in self.circuit.regulators.items()
            },
            "bat_statuses": {
                n: [b.soc, powers["storage." + n][0] / b.max_kw]
                for n, b in self.circuit.batteries.items()
            },
            "pv_statuses": {
                n: [
                    powers["pvsystem." + n][0] / pv.pmpp,
                    powers["pvsystem." + n][1] / pv.kva,
                ]
                for n, pv in self.circuit.pvs.items()
            },
            "svg_statuses": {
                n: [
                    powers["generator." + n][0] / dev.kva,
                    powers["generator." + n][1] / dev.kva,
                ]
                for n, dev in self.circuit.svgs.items()
            },
            "power_loss_ratio": self.physical_state.losses[0] / self.base_load_kw,
            "time": self.t,
        }

    def wrap_obs(self, obs):
        values = []
        for key in (
            "bus_voltages",
            "cap_statuses",
            "reg_statuses",
            "bat_statuses",
            "pv_statuses",
            "svg_statuses",
        ):
            for value in obs[key].values():
                values.extend(np.asarray(value).reshape(-1))
        values.extend(
            [
                self.current_row[4],
                self.current_row[5] / 100,
                *self.current_row[1:4],
                self.t / self.horizon,
            ]
        )
        if self.config.use_lagrangian_reward and self.lagrangian_updater:
            values.append(self.lagrangian_updater.lmbda)
        result = np.asarray(values, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("观测出现非有限值")
        return result

    def get_physics_manifest(self):
        elements = (
            ["Capacitor." + n for n in self.cap_names]
            + ["RegControl." + n for n in self.reg_names]
            + ["Storage." + n for n in self.bat_names]
            + (
                ["PVSystem." + n for n in self.pv_names]
                if self.pv_control_enabled
                else []
            )
        )
        elements += ["Generator." + n for n in self.svg_names]
        schema = [
            {"element": element, "space": str(space)}
            for element, space in zip(elements, self.agent_spaces)
        ]
        digest = sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        return {
            **self.circuit.manifest(),
            "action_schema": schema,
            "action_schema_sha256": digest,
        }

    def _info(self):
        phases = [v for values in self.obs["bus_voltages"].values() for v in values]
        count = sum(
            v < self.config.voltage_min or v > self.config.voltage_max for v in phases
        )
        buses = sum(
            any(
                v < self.config.voltage_min or v > self.config.voltage_max
                for v in values
            )
            for values in self.obs["bus_voltages"].values()
        )
        pv_powers = [-self.circuit.powers("PVSystem." + n) for n in self.pv_names]
        pv_kw = sum(p[0] for p in pv_powers)
        rated = sum(pv.pmpp for pv in self.circuit.pvs.values())
        storage = [-self.circuit.powers("Storage." + n)[0] for n in self.bat_names]
        loadings = [
            b["loading_pu"]
            for b in self.physical_state.branches.values()
            if b["loading_pu"] is not None
        ]
        return {
            "max_branch_loading_pu": max(loadings) if loadings else None,
            "branch_overload_count": sum(x > 1 for x in loadings),
            "unrated_branch_count": len(self.physical_state.branches) - len(loadings),
            "dataset_sha256": self.dataset.sha256,
            "physics_model_sha256": self.circuit.model_sha256,
            "control_trace": {
                k: v for k, v in self.circuit.trace.items() if not k.startswith("Load.")
            },
            "active_balance_error_kw": self.physical_state.active_balance_error_kw,
            "svg_reactive_power_kvar": float(
                sum(-self.circuit.powers("Generator." + n)[1] for n in self.svg_names)
            ),
            "scenario_id": self.entry["id"],
            "scenario": self.entry["scenario"],
            "step": self.t,
            "time_s": float(self.circuit.native.Solution.dblHour * 3600),
            "irradiance_kw_m2": float(self.current_row[4]),
            "array_temperature_c": float(self.current_row[5]),
            "power_loss_kw": float(self.circuit.total_loss()[0]),
            "total_load_kw": float(self.circuit.total_load_power()[0]),
            "pv_power_kw": float(pv_kw),
            "pv_output_kw": float(pv_kw),
            "pv_utilization": float(pv_kw / rated) if rated else 0.0,
            "pv_avg_power_factor": (
                float(
                    np.mean([abs(p) / max(np.hypot(p, q), 1e-9) for p, q in pv_powers])
                )
                if pv_powers
                else 0.0
            ),
            "power_loss_kvar": float(self.circuit.total_loss()[1]),
            "total_power_kw": float(-self.circuit.total_power()[0]),
            "total_power_kvar": float(-self.circuit.total_power()[1]),
            "voltage_violation_count": count,
            "voltage_violations_per_bus": buses / max(len(self.all_bus_names), 1),
            "voltage_violation_rate": count / max(len(phases), 1),
            "voltage_min_pu": float(min(phases)),
            "voltage_max_pu": float(max(phases)),
            "battery_charge_kw": sum(max(-p, 0) for p in storage),
            "battery_discharge_kw": sum(max(p, 0) for p in storage),
            "power_loss_ratio": self.obs["power_loss_ratio"],
            "battery_avg_soc": (
                float(np.mean([b.soc for b in self.circuit.batteries.values()]))
                if self.bat_num
                else 0
            ),
            "dss_converged": bool(self.circuit.native.Solution.Converged),
        }

    def reset(self, load_profile_idx=None, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            if self.lagrangian_updater:
                self.lagrangian_updater.reset(self.config.lambda_init)
        index = self._next_episode if load_profile_idx is None else load_profile_idx
        if not 0 <= index < len(self.dataset.episodes):
            raise ValueError(f"场景编号 {index} 越界")
        self._next_episode = (index + 1) % len(self.dataset.episodes)
        self.entry, self.profile = self.dataset.episodes[index]
        self.t, self.episode_costs = 0, []
        self.reward_func.reset_episode()
        self.circuit.reset(seconds=float(self.profile[0, 0]))
        self._apply_data(0)
        self.available_pv_kw = self.circuit.available_pv_power()
        self._observe()
        return self.wrap_obs(self.obs)

    def get_obs(self, wrapped=True):
        return self.wrap_obs(self.obs) if wrapped else self.obs

    def close(self):
        if not self._closed:
            self.circuit.close()
            self._closed = True
