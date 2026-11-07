"""顺序电价→用户响应→一次原生工程区间；逐主体元收益，不是团队MARL。"""

from itertools import product
from pathlib import Path
import numpy as np
from gymnasium import spaces
from envs.powergrid.session import DSSSession
from envs.powergrid.scenario import prepare_dataset, apply_operating_row
from envs.stackelberg.stackelberg_config import StackelbergConfig

ROOT = Path(__file__).resolve().parents[2]


class SequentialPricingEnv:
    def __init__(self, config):
        from utils.config_validation import build_dataclass

        self.config = cfg = (
            config
            if isinstance(config, StackelbergConfig)
            else build_dataclass(StackelbergConfig, config)
        )
        cfg.validate()
        self.dataset = prepare_dataset(cfg.data, cfg.max_episode_steps)
        folder = Path(cfg.dss_folder or f"node_systems/{cfg.system_name}")
        self.circuit = DSSSession(
            folder if folder.is_absolute() else ROOT / folder,
            cfg.dss_file,
            cfg.overlays,
            cfg.data.step_seconds,
        )
        try:
            self._bind()
            self.horizon = cfg.max_episode_steps
            self.agent_names = ["operator"] + [c.name for c in cfg.consumers]
            self.action_spaces = [
                spaces.Discrete(len(cfg.retail_prices_yuan_per_kwh))
            ] + [spaces.Discrete(len(cfg.load_fractions)) for c in cfg.consumers]
            self.seed(cfg.seed)
            leader = self.reset(load_profile_idx=0)
            self.observation_spaces = [
                spaces.Box(-np.inf, np.inf, leader.shape, dtype=np.float32)
            ] + [
                spaces.Box(-np.inf, np.inf, (len(leader) + 3,), dtype=np.float32)
                for c in cfg.consumers
            ]
        except Exception:
            self.close()
            raise

    def _bind(self):
        cfg = self.config
        from envs.powergrid.topology import energized_phases, terminal_nodes

        reached = energized_phases(self.circuit)
        for name, load in self.circuit.loads.items():
            if (
                load.kw < 0
                or not set(terminal_nodes(self.circuit.connection("Load." + name), 0))
                <= reached
            ):
                raise ValueError("定价任务需要非负真实负荷且全部相别连接有效主网源")
        self.groups = [
            [n.lower().removeprefix("load.") for n in c.loads] for c in cfg.consumers
        ]
        owned = [n for group in self.groups for n in group]
        if len(set(owned)) != len(owned) or set(owned) != set(self.circuit.loads):
            raise ValueError("每个真实Load必须恰好归属一个消费者，不能重复、遗漏或虚构")
        if set(cfg.data.load_groups) - set(self.circuit.loads):
            raise ValueError("data.load_groups包含未知Load")
        if (
            self.circuit.pvs
            or self.circuit.svgs
            or self.circuit.native.Generators.Count
        ):
            raise ValueError("首版定价只支持真实负荷需求响应，不支持占位DER/PV售电")
        if any(len(s) != 1 for s in self.circuit.capacitors.values()):
            raise ValueError("首版只支持单级电容器")
        for name in self.circuit.loads:
            if not self.circuit.connection("Load." + name)["enabled"]:
                raise ValueError("定价模型不能包含停用的消费者负荷")
        self.group_base_kw = np.array(
            [sum(self.circuit.loads[n].kw for n in group) for group in self.groups]
        )
        self.base_kw = float(self.group_base_kw.sum())
        if (self.group_base_kw <= 0).any():
            raise ValueError("每个消费者需要正的DSS负荷基准")

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self._next_episode = int(self.rng.integers(len(self.dataset.episodes)))
        for i, space in enumerate(self.action_spaces):
            space.seed((seed + i) % 2**32)

    def reset(self, load_profile_idx=None, *, seed=None):
        if seed is not None:
            self.seed(seed)
        index = self._next_episode if load_profile_idx is None else load_profile_idx
        if not 0 <= index < len(self.dataset.episodes):
            raise ValueError("场景编号越界")
        self._next_episode = (index + 1) % len(self.dataset.episodes)
        self.entry, self.profile = self.dataset.episodes[index]
        self.t, self.phase, self.price_index = 0, "leader", None
        self.previous_price = 0.0
        self.previous_fractions = np.ones(len(self.groups))
        self.circuit.reset(float(self.profile[0, 0]))
        # 机械设备由回合开始的原生控制确定，之后冻结；Storage固定原生空闲。
        apply_operating_row(
            self.circuit, self.profile[0], self.dataset, self.config.data.load_groups
        )
        self.circuit.apply(
            {"Storage." + n: {"kw": 0.0} for n in self.circuit.batteries}
        )
        self.circuit.solve(native_controls=True)
        self._prepare_row()
        return self.leader_observation()

    def _prepare_row(self):
        self.current_row = self.profile[self.t]
        apply_operating_row(
            self.circuit, self.current_row, self.dataset, self.config.data.load_groups
        )
        self.circuit.solve()
        self.requested = {
            n: (
                d.kw
                * float(
                    self.current_row[
                        1 + self.dataset.group_for(n, self.config.data.load_groups)
                    ]
                ),
                d.kvar
                * float(
                    self.current_row[
                        1 + self.dataset.group_for(n, self.config.data.load_groups)
                    ]
                ),
            )
            for n, d in self.circuit.loads.items()
        }
        self.baseline_kw = self._delivered()
        self.response = tuple([len(self.config.load_fractions) - 1] * len(self.groups))
        self.baseline_state = self.circuit.read_state()
        self.wholesale_price = self.config.wholesale_prices_yuan_per_kwh[
            int(self.current_row[0] // 3600) % 24
        ]

    def _delivered(self):
        values = np.array(
            [
                sum(float(self.circuit.powers("Load." + n)[0]) for n in group)
                for group in self.groups
            ]
        )
        if not np.isfinite(values).all() or (values < -1e-6).any():
            raise RuntimeError("消费者原生交付功率必须非负有限")
        return np.maximum(values, 0)

    def leader_observation(self):
        if self.phase != "leader":
            raise RuntimeError("当前不是领导者阶段")
        state = self.baseline_state
        v = [v for bus in state.voltages_pu.values() for v in bus.values()]
        if not v:
            raise RuntimeError("定价任务需要有效源与原生电压")
        loading = max(b["loading_pu"] for b in state.branches.values())
        return np.asarray(
            [
                self.t / self.horizon,
                np.sin(2 * np.pi * self.current_row[0] / 86400),
                np.cos(2 * np.pi * self.current_row[0] / 86400),
                self.wholesale_price,
                self.previous_price,
                min(v),
                max(v),
                loading,
                state.grid_import[0] / self.base_kw,
                *self.baseline_kw / self.group_base_kw,
                *self.previous_fractions,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _index(value, space):
        a = np.asarray(value)
        if (
            a.size != 1
            or not np.isfinite(a).all()
            or float(a.item()) != int(a.item())
            or not space.contains(int(a.item()))
        ):
            raise ValueError("动作必须是空间内的单个有限整数")
        return int(a.item())

    def publish_price(self, action):
        if self.phase != "leader" or self.t >= self.horizon:
            raise RuntimeError("须在领导者阶段公布一次本轮价格")
        index = self._index(action, self.action_spaces[0])
        self._leader_obs = self.leader_observation()
        self.price_index, self.phase = index, "followers"
        price = self.config.retail_prices_yuan_per_kwh[index]
        return [
            np.asarray(
                [
                    *self._leader_obs,
                    price,
                    self.baseline_kw[i] / self.group_base_kw[i],
                    c.discomfort_yuan_per_kwh,
                ],
                dtype=np.float32,
            )
            for i, c in enumerate(self.config.consumers)
        ]

    def _validate_response(self, actions):
        if len(actions) != len(self.groups):
            raise ValueError("跟随者动作数量不符")
        return tuple(self._index(a, s) for a, s in zip(actions, self.action_spaces[1:]))

    def _apply_response(self, response):
        self.circuit.apply(
            {
                "Load."
                + n: {
                    "kw": self.requested[n][0] * self.config.load_fractions[a],
                    "kvar": self.requested[n][1] * self.config.load_fractions[a],
                }
                for group, a in zip(self.groups, response)
                for n in group
            }
        )

    def _ledger(self, price_index, response):
        state = self.circuit.read_state()
        dt = self.config.data.step_seconds / 3600
        delivered = self._delivered()
        energy = delivered * dt
        price = self.config.retail_prices_yuan_per_kwh[price_index]
        bills = energy * price
        reduction = np.maximum(0, self.baseline_kw - delivered)
        beta = np.array([c.discomfort_yuan_per_kwh for c in self.config.consumers])
        discomfort = 0.5 * beta * reduction**2 / np.maximum(self.baseline_kw, 1e-9) * dt
        voltages = [v for bus in state.voltages_pu.values() for v in bus.values()]
        violation = sum(
            max(0, self.config.voltage_min - v, v - self.config.voltage_max)
            for v in voltages
        )
        overload = sum(max(0, b["loading_pu"] - 1) for b in state.branches.values())
        network_cost = (
            self.config.network_cost_yuan_per_hour * (violation + overload) * dt
        )
        # 首版不支持售电资源；负荷可出现数值级负进口，不能据此生成补贴。
        if state.grid_import[0] < -1e-6:
            raise RuntimeError("首版定价任务不支持向主网售电")
        wholesale_cost = max(0, state.grid_import[0]) * dt * self.wholesale_price
        leader = float(bills.sum() - wholesale_cost - network_cost)
        utilities = [leader, *(-bills - discomfort).tolist()]
        info = dict(
            utilities_yuan=utilities,
            retail_price_yuan_per_kwh=price,
            wholesale_price_yuan_per_kwh=self.wholesale_price,
            consumer_energy_kwh=energy.tolist(),
            consumer_bills_yuan=bills.tolist(),
            consumer_discomfort_yuan=discomfort.tolist(),
            consumer_baseline_kw=self.baseline_kw.tolist(),
            consumer_delivered_kw=delivered.tolist(),
            response=list(response),
            retail_revenue_yuan=float(bills.sum()),
            wholesale_cost_yuan=float(wholesale_cost),
            network_cost_yuan=float(network_cost),
            grid_import_kw=float(state.grid_import[0]),
            losses_kw=float(state.losses[0]),
            voltage_min_pu=float(min(voltages)),
            voltage_max_pu=float(max(voltages)),
            max_branch_loading_pu=float(
                max(b["loading_pu"] for b in state.branches.values())
            ),
            balance_error_kw=float(state.active_balance_error_kw),
            interval_start_s=float(self.current_row[0]),
            step_seconds=self.config.data.step_seconds,
            scenario=self.entry["id"],
            split=self.config.data.split,
        )
        if not np.isfinite(utilities).all():
            raise RuntimeError("效用不是有限值")
        return np.asarray(utilities, dtype=np.float64), info

    def respond(self, actions):
        if self.phase != "followers":
            raise RuntimeError("须先公布价格，再采样跟随者动作")
        response = self._validate_response(actions)
        self.circuit.begin_interval(float(self.current_row[0]))
        self._apply_response(response)
        self.circuit.solve()
        utilities, info = self._ledger(self.price_index, response)
        self.state = self.circuit.read_state()
        self.circuit.finish_interval()
        info["time_s"] = self.circuit.time_s
        info["dss_converged"] = True
        self.previous_price = self.config.retail_prices_yuan_per_kwh[self.price_index]
        self.previous_fractions = np.array(
            [self.config.load_fractions[a] for a in response]
        )
        self.t += 1
        done = self.t == self.horizon
        self.phase = "done" if done else "leader"
        if not done:
            self._prepare_row()
        return (None if done else self.leader_observation()), utilities, done, info

    def evaluate_response(self, price_index, actions):
        """当前区间的零时间反事实；恢复原生负荷、阶段、时间和能量，不当作交付。"""
        if self.phase not in ("leader", "followers"):
            raise RuntimeError("回合已结束")
        p = self._index(price_index, self.action_spaces[0])
        response = self._validate_response(actions)
        try:
            self._apply_response(response)
            self.circuit.solve()
            return self._ledger(p, response)
        finally:
            self._apply_response(self.response)
            self.circuit.solve()

    def exact_response_table(self, price_index):
        count = int(np.prod([s.n for s in self.action_spaces[1:]]))
        if count > 4096:
            raise ValueError(
                "精确参考最多4096个联合响应；大算例请单独实现并声明近似范围"
            )
        table = {
            a: self.evaluate_response(price_index, a)[0]
            for a in product(*[range(s.n) for s in self.action_spaces[1:]])
        }
        equilibria = []
        for a, u in table.items():
            regrets = [
                max(
                    table[tuple([*a[:i], b, *a[i + 1 :]])][i + 1] - u[i + 1]
                    for b in range(s.n)
                )
                for i, s in enumerate(self.action_spaces[1:])
            ]
            if max(regrets) <= 1e-7:
                equilibria.append(a)
        if not equilibria:
            raise RuntimeError("当前离散阶段博弈没有纯策略均衡；不伪造响应")
        return table, sorted(equilibria)

    def exact_reference(self):
        """每个价格重求纯响应，按动作字典序选均衡；领导者收益相等时选较低价格。"""
        results = []
        for p in range(self.action_spaces[0].n):
            table, equilibria = self.exact_response_table(p)
            selected = equilibria[0]
            results.append(
                dict(
                    price_index=p,
                    response=selected,
                    utilities_yuan=table[selected].tolist(),
                    pure_equilibria=len(equilibria),
                )
            )
        best = max(results, key=lambda r: r["utilities_yuan"][0])
        return dict(
            selection="lexicographic_pure_response_then_lowest_price_tie",
            prices=results,
            best=best,
        )

    def unilateral_gains(self, price_index, actions):
        response = self._validate_response(actions)
        current = self.evaluate_response(price_index, response)[0]
        return [
            max(
                0.0,
                max(
                    self.evaluate_response(
                        price_index, tuple([*response[:i], b, *response[i + 1 :]])
                    )[0][i + 1]
                    - current[i + 1]
                    for b in range(s.n)
                ),
            )
            for i, s in enumerate(self.action_spaces[1:])
        ]

    def step(self, *args):
        raise RuntimeError(
            "顺序任务禁止同步联合step；使用publish_price→respond或sequential_ppo入口"
        )

    def physics_manifest(self):
        return dict(
            task="sequential_pricing_v1",
            interaction="leader_price_followers_response_one_interval",
            native=self.circuit.manifest(),
            agents=self.agent_names,
            consumers=[
                dict(
                    name=c.name,
                    loads=g,
                    discomfort_yuan_per_kwh=c.discomfort_yuan_per_kwh,
                )
                for c, g in zip(self.config.consumers, self.groups)
            ],
            retail_prices_yuan_per_kwh=self.config.retail_prices_yuan_per_kwh,
            load_fractions=self.config.load_fractions,
            economics=dict(
                wholesale_prices_yuan_per_kwh=self.config.wholesale_prices_yuan_per_kwh,
                voltage_min=self.config.voltage_min,
                voltage_max=self.config.voltage_max,
                network_cost_yuan_per_hour=self.config.network_cost_yuan_per_hour,
            ),
            observation_shapes=[list(s.shape) for s in self.observation_spaces],
            observation_schema="leader:progress,sin_hour,cos_hour,wholesale,previous_retail,vmin,vmax,loading,grid/base,baseline/group_base,previous_fractions; follower:leader,current_retail,own_baseline/base,discomfort",
            actions=[int(s.n) for s in self.action_spaces],
        )

    def close(self):
        self.circuit.close()
