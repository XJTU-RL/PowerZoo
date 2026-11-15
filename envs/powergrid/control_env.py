"""同步设备运行流程，供独立VVC与SmartGrid任务共用；不包含任务奖励。"""

from pathlib import Path
from hashlib import sha256
import json
import math
import numpy as np
import gymnasium as gym
from envs.powergrid.session import DSSSession
from envs.powergrid.scenario import prepare_dataset, apply_operating_row


def summarize_branch_stats(branches):
    """支路过载统计的三态契约（复审 4d8af3b5：未采集曾被记成 0 过载）。

    - branches 为 None（light 读取未采集）：collected=False，最大值与
      计数均为 None——未采集不等于无过载；
    - 已采集但无可评级支路（loading_pu 全 None）：collected=True、
      计数为事实值（0 条可评级过载），unrated 记录未评级数——不得
      据此宣称整个网络没有过载。
    """
    if branches is None:
        return {
            "branch_stats_collected": False,
            "max_branch_loading_pu": None,
            "branch_overload_count": None,
            "unrated_branch_count": None,
        }
    loadings = [
        branch["loading_pu"]
        for branch in branches.values()
        if branch["loading_pu"] is not None
    ]
    return {
        "branch_stats_collected": True,
        "max_branch_loading_pu": max(loadings) if loadings else None,
        "branch_overload_count": sum(x > 1 for x in loadings),
        "unrated_branch_count": len(branches) - len(loadings),
    }


class DeviceControlEnv(gym.Env):
    def __init__(
        self,
        folder_path,
        config,
        overlays=(),
        *,
        pv_control_enabled=False,
        pv_act_num=float("inf"),
        svg_elements=(),
        pv_control_mode="legacy",
        pv_control_devices=None,
        pv_action_mode="pq",
        light_state_read=False,
        observation_mode="global",
    ):
        self.config = cfg = config
        smartds_spec = getattr(cfg, "smartds_series", None)
        if smartds_spec:
            # SMART-DS 真实逐设备时序：episode=天，训练/评估按天留出。
            from utils.external_cases import locate_entry, manifest_path
            from envs.powergrid.smartds_timeseries import SmartDSEpisodeDataset

            folder, master = locate_entry(
                smartds_spec["case"], smartds_spec["entry"]
            )
            layout = folder.parents[4]
            expected_manifest = smartds_spec.get("manifest_sha256")
            if expected_manifest and sha256(
                Path(manifest_path(smartds_spec["case"])).read_bytes()
            ).hexdigest() != expected_manifest:
                raise ValueError("SMART-DS 外部清单摘要与训练快照不符")
            rel_master = str(folder.relative_to(layout)) + "/" + master
            digest_key = (
                "dataset_sha256_validation" if smartds_spec.get("split", "train") == "validation"
                else "dataset_sha256"
            )
            if any(smartds_spec.get(key) for key in ("dataset_sha256", "dataset_sha256_validation")):
                if not smartds_spec.get(digest_key):
                    raise ValueError(f"SMART-DS 缺少当前 split 的冻结摘要：{digest_key}")
            self.dataset = SmartDSEpisodeDataset(
                layout,
                rel_master,
                days={
                    "train": smartds_spec["train_days"],
                    "validation": smartds_spec.get("validation_days", []),
                },
                split=smartds_spec.get("split", "train"),
                # 快照恢复：prepare_run 按 split 冻结的对应摘要（无预填不校验）。
                # 摘要组成含 split（不同 split 是不同数据切片），评估 worker
                # 须取 validation 口径的期望值而非 train 的。
                content_sha256=smartds_spec.get(digest_key),
            )
            if smartds_spec.get(digest_key):
                # 恢复校验在构造期触发（访问惰性摘要并比对），数据版本
                # 不符立即失败而不是等到首个 info。
                _ = self.dataset.sha256
            if len(self.dataset.day_indices[self.dataset.split]) == 0:
                raise ValueError(
                    f"smartds_series split={self.dataset.split} 没有可用天"
                )
            if self.dataset.max_episode_steps_ok(cfg.max_episode_steps) is False:
                raise ValueError("episode 步数超过一天 96 步")
        else:
            self.dataset = prepare_dataset(cfg.data, cfg.max_episode_steps)
        system = layout if smartds_spec else Path(
            cfg.dss_folder or f"node_systems/{cfg.system_name}"
        )
        if not system.is_absolute():
            system = Path(folder_path) / system
        self.circuit = DSSSession(
            system, rel_master if smartds_spec else cfg.dss_file,
            overlays, cfg.data.step_seconds, svg_elements
        )
        if smartds_spec:
            if self.circuit.step_seconds != 900:
                self.circuit.close()
                raise ValueError("smartds_series 需要 data.step_seconds=900（15 分钟原始采样）")
            try:
                self.dataset.bind(self.circuit)
            except Exception:
                self.circuit.close()
                raise
        self._episode_index = 0
        if self.circuit.native.Generators.Count and any(
            n.lower().startswith("batt")
            for n in self.circuit.native.Generators.AllNames
        ):
            self.circuit.close()
            raise ValueError(
                "Generator.batt旧占位设备没有原生能量状态，请迁移为有额定值的Storage并使用network.dss"
            )
        unknown_loads = set(cfg.data.load_groups) - set(self.circuit.loads)
        if unknown_loads:
            self.circuit.close()
            raise ValueError(f"data.load_groups 包含不存在的 DSS 负荷：{unknown_loads}")
        self.dss_folder_path, self.dss_file = str(system.resolve()), cfg.dss_file
        self.horizon = cfg.max_episode_steps
        self.divergence_penalty = getattr(cfg, "divergence_penalty", 10000.0)
        self._diverged = False
        self.reg_act_num, self.bat_act_num, self.pv_act_num = (
            cfg.reg_act_num,
            cfg.bat_act_num,
            pv_act_num,
        )
        if any(
            reg.numtaps % (self.reg_act_num - 1)
            for reg in self.circuit.regulators.values()
        ):
            self.circuit.close()
            raise ValueError(
                "regulator.action_num 的间隔必须对齐 DSS NumTaps 的实际档位"
            )
        self.pv_control_enabled = pv_control_enabled
        # 精简状态读取（默认关）：训练步用批量电压+设备类注入，省略全网
        # 支路明细与负荷类注入（info 的支路统计与平衡校验为 None）；完整
        # 读取用于快照/评估。两种模式的观测量逐位一致（测试锁定）。
        self._light_state_read = bool(light_state_read)
        # 局部 actor 观测模式（CTDE）：global=actor 与 critic 同用全网 wrap_obs
        # （旧行为）；local=actor 用 agent_local_observation（自身母线/状态/
        # k 跳邻域摘要），critic 状态保持全网 wrap_obs 不变。
        if observation_mode not in ("global", "local"):
            self.circuit.close()
            raise ValueError(f"未知观测模式：{observation_mode}")
        self.observation_mode = observation_mode
        # PV 控制权三态（SMART-DS 引入；legacy 逐字节保持旧布尔路径）：
        # - legacy：pv_control_enabled 布尔语义，全部 PV 全量探测（探测写入
        #   %Pmpp=100 kvar=0 且保持——既有训练路径依赖此行为）。
        # - rl：只控制 pv_control_devices（缺省全部 PV），探测仅限接管名单且
        #   探测前后恢复设定；未接管 PV 不写不探测（保持模型 pf/kvar 设定）；
        #   绑定被接管 PV 的原生 InvControl 每回合重编译后显式停用并记录。
        # - fixed：PV 不进动作空间、不探测、不写（模型静态 pf 运行）。
        self.pv_control_mode = pv_control_mode
        self.pv_action_mode = pv_action_mode
        if pv_control_mode not in ("legacy", "rl", "fixed"):
            self.circuit.close()
            raise ValueError(f"未知 PV 控制模式：{pv_control_mode}")
        if pv_action_mode not in ("pq", "q_only"):
            self.circuit.close()
            raise ValueError(f"未知 PV 动作模式：{pv_action_mode}")
        if pv_action_mode == "q_only" and pv_act_num != float("inf"):
            self.circuit.close()
            raise ValueError("q_only 模式需要连续动作空间（pv_act_num=inf）")
        discovered_pvs = list(self.circuit.pvs)
        self.pv_control_list = list(discovered_pvs) if pv_control_enabled else []
        if pv_control_mode == "rl":
            requested = list(pv_control_devices) if pv_control_devices else list(discovered_pvs)
            unknown = set(requested) - set(discovered_pvs)
            if unknown:
                self.circuit.close()
                raise ValueError(f"pv_control_devices 包含不存在的 PV：{sorted(unknown)}")
            self.pv_control_list = sorted(set(requested))
            self.pv_control_enabled = True
        elif pv_control_mode == "fixed":
            self.pv_control_list = []
            self.pv_control_enabled = False
        self.disabled_invcontrols = {}
        if pv_control_mode == "rl":
            from envs.powergrid.devices import invcontrol_bindings

            controlled = set(self.pv_control_list)
            for name, members in invcontrol_bindings(self.circuit).items():
                if controlled.intersection(members):
                    self.disabled_invcontrols[name] = members
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
        self._obs_buses, self._obs_projection = self._resolve_observation_buses(
            getattr(cfg, "observation_buses", None),
            getattr(cfg, "observation_neighborhood", None),
        )
        self.useS, self.use_render = False, False
        self.wrap_observation, self.observe_load = True, False
        self.agent_spaces = (
            [gym.spaces.Discrete(2) for _ in self.cap_names]
            + [gym.spaces.Discrete(self.reg_act_num) for _ in self.reg_names]
            + [self._space(self.bat_act_num, 1) for _ in self.bat_names]
            # PV 动作维度：pq=2（有功分数+无功命令）；q_only=1（有功不控、
            # 保持当步可用功率，仅无功命令）。控制数量与顺序取自
            # pv_control_list（回合内稳定）。
            + [
                self._space(
                    self.pv_act_num, 1 if self.pv_action_mode == "q_only" else 2
                )
                for _ in self.pv_control_list
            ]
            + [self._space(float("inf"), 1) for _ in self.svg_names]
        )
        self.base_load_kw = sum(load.kw for load in self.circuit.loads.values())
        if self.base_load_kw <= 0:
            self.circuit.close()
            raise ValueError("运行控制需要正的原生负荷基准")
        if any(len(states) != 1 for states in self.circuit.capacitors.values()):
            self.circuit.close()
            raise ValueError("运行控制当前只支持单级电容器")
        self.action_space = gym.spaces.Tuple(self.agent_spaces)
        self._init_task()
        self._closed = False
        try:
            self.seed(cfg.seed)
            self.reset(load_profile_idx=0)
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=self.wrap_obs(self.obs).shape, dtype=np.float32
            )

        except Exception:
            self.close()
            raise

    @staticmethod
    def _space(n, dimension):
        return (
            gym.spaces.Box(-1, 1, (dimension,), dtype=np.float32)
            if n == float("inf")
            else gym.spaces.Discrete(int(n))
        )

    def _resolve_observation_buses(self, buses, neighborhood):
        """在首次 reset 前固定投影母线有序名单；跨步稳定是 schema 对齐的前提。

        名单模式：显式列表（大小写不敏感匹配 DSS 母线名）。
        邻域模式：可控设备母线 ∪ 电压源母线，经拓扑图 k 跳扩展。
        返回 (有序母线名列表或 None, 投影说明或 None)；None 表示全母线展开不变。
        """
        if buses is None and neighborhood is None:
            return None, None
        known = {b.lower(): b for b in self.all_bus_names}
        if buses is not None:
            missing = sorted(b for b in buses if b.lower() not in known)
            if missing:
                self.circuit.close()
                raise ValueError(f"observation_buses 包含不存在的母线：{missing}")
            projection = sorted({known[b.lower()] for b in buses})
            return projection, {"mode": "buses", "bus_count": len(projection)}
        groups = (
            ("capacitor.", self.cap_names),
            ("storage.", self.bat_names),
            ("pvsystem.", self.pv_names),
            ("generator.", self.svg_names),
        )
        seeds = set()
        for prefix, device_names in groups:
            for name in device_names:
                conn = self.circuit._connections.get(prefix + name.lower())
                if conn and conn["buses"]:
                    seeds.add(conn["buses"][0].split(".")[0].lower())
        for reg in self.circuit.regulators.values():
            conn = self.circuit._connections.get(
                "transformer." + reg.transformer.lower()
            )
            if conn and conn["buses"]:
                seeds.add(conn["buses"][0].split(".")[0].lower())
        for element, conn in self.circuit._connections.items():
            if element.startswith("vsource.") and conn["buses"]:
                seeds.add(conn["buses"][0].split(".")[0].lower())
        reached = {seed for seed in seeds if seed in known}
        frontier = {seed for seed in seeds if seed in self.topology}
        reached |= frontier
        for _ in range(neighborhood):
            nxt = set()
            for bus in frontier:
                nxt |= set(self.topology[bus])
            nxt -= reached
            reached |= nxt
            frontier = nxt
            if not frontier:
                break
        projection = sorted(known[b] for b in reached if b in known)
        return projection, {
            "mode": "neighborhood",
            "hops": neighborhood,
            "seed_count": len(seeds),
            "bus_count": len(projection),
        }

    def _voltage_statistics(self, voltages):
        """投影模式下保留全网电压量纲：min/max/mean 与越限比例取自全量 dict。"""
        values = np.concatenate(
            [np.asarray(v) for v in voltages.values() if len(v)]
        )
        if not values.size:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        below = values < self.config.voltage_min
        above = values > self.config.voltage_max
        return [
            float(values.min()),
            float(values.max()),
            float(values.mean()),
            float(below.mean()),
            float(above.mean()),
        ]

    # ------------------------------------------------------------------
    # 局部 actor 观测（CTDE）：每智能体只见自身母线/自身状态/k 跳邻域摘要；
    # 全网 wrap_obs 保留为集中 critic 的共享状态。信息边界由构造源限定，
    # 测试锁定"屏蔽邻域外电压不改局部观测"。
    # ------------------------------------------------------------------

    def _agent_devices(self):
        """动作空间构造序的 (设备类, 设备名) 列表；与 agent_spaces 对齐。"""
        return (
            [("capacitor", name) for name in self.cap_names]
            + [("regcontrol", name) for name in self.reg_names]
            + [("storage", name) for name in self.bat_names]
            + [("pvsystem", name) for name in self.pv_control_list]
            + [("generator", name) for name in self.svg_names]
        )

    def _local_neighborhoods(self):
        """每智能体的 k 跳邻域母线集合（拓扑静态，惰性构建一次）。"""
        cache = getattr(self, "_local_views", None)
        if cache is not None:
            return cache
        hops = int(getattr(self.config, "local_observation_neighborhood", 1))
        views = []
        for kind, name in self._agent_devices():
            element = (
                "transformer." + self.circuit.regulators[name].transformer
                if kind == "regcontrol"
                else kind + "." + name
            )
            conn = self.circuit._connections.get(element.lower())
            seed = (
                conn["buses"][0].split(".")[0].lower()
                if conn and conn["buses"]
                else None
            )
            reached, frontier = set(), set()
            if seed:
                frontier = {seed} & set(self.topology)
                reached = {seed} | frontier
            for _ in range(hops):
                nxt = set()
                for bus in frontier:
                    nxt |= set(self.topology[bus])
                nxt -= reached
                reached |= nxt
                frontier = nxt
                if not frontier:
                    break
            views.append((seed, reached))
        self._local_views = views
        return views

    def _device_bus(self, kind, name):
        element = (
            "transformer." + self.circuit.regulators[name].transformer
            if kind == "regcontrol"
            else kind + "." + name
        )
        conn = self.circuit._connections.get(element.lower())
        if conn and conn["buses"]:
            return conn["buses"][0].split(".")[0].lower()
        return None

    def agent_local_observation(self, index):
        """第 index 个智能体的局部观测（float32 定长向量；组内同型同维）。

        组成（按序）：自身母线相电压（pad 到 3）+ 邻域电压 5 元组
        （min/max/mean/低越限比/高越限比）+ 日内时间 sin/cos + 回合进度
        + 设备自身特征（按类型，见 local_observation_schema）。
        """
        kind, name = self._agent_devices()[index]
        seed, neighborhood = self._local_neighborhoods()[index]
        voltages = self.physical_state.voltages_pu
        own = [voltages.get(seed, {}).get(phase, 0.0) for phase in (1, 2, 3)]
        values = np.concatenate(
            [
                np.asarray(list(voltages.get(bus, {}).values()))
                for bus in sorted(neighborhood)
                if voltages.get(bus)
            ]
            or [np.zeros(0)]
        )
        if not values.size:
            values = np.zeros(1)
        low, high = self.config.voltage_min, self.config.voltage_max
        summary = [
            float(values.min()),
            float(values.max()),
            float(values.mean()),
            float((values < low).mean()),
            float((values > high).mean()),
        ]
        hour = (self.circuit.time_s / 3600.0) % 24.0
        clock = [math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]
        progress = [self.t / max(self.horizon, 1)]
        # 设备特征：全部类型定长 7 槽（不足补 0），配合类型 one-hot 使所有
        # 智能体的局部观测同维——并行环境按 (n_envs, n_agents, dim) 堆叠
        # 要求统一维度；类型身份由 one-hot 段携带，分组共享仍按动作空间分。
        if kind == "capacitor":
            device = [float(self.obs["cap_statuses"][name])]
        elif kind == "regcontrol":
            device = [float(self.obs["reg_statuses"][name])]
        elif kind == "storage":
            soc, power = self.obs["bat_statuses"][name]
            device = [float(soc), float(power)]
        elif kind == "pvsystem":
            pv = self.circuit.pvs[name]
            p, q = self.obs["pv_statuses"][name]
            available = self.available_pv_kw.get(name, 0.0)
            kw = available * pv.fraction
            headroom = math.sqrt(max(0.0, pv.kva**2 - kw**2))
            device = [
                float(p),
                float(q),
                float(available / pv.pmpp),
                float(min(pv.kvar_max, headroom) / pv.kva),
                float(min(pv.kvar_max_abs, headroom) / pv.kva),
                float(pv.fraction),
                float(pv.q_fraction),
            ]
        else:  # generator(SVG)
            p, q = self.obs["svg_statuses"][name]
            device = [float(p), float(q)]
        device = device[:7] + [0.0] * (7 - len(device))
        kinds = ("capacitor", "regcontrol", "storage", "pvsystem", "generator")
        onehot = [1.0 if kind == k else 0.0 for k in kinds]
        vector = np.asarray(
            onehot + own + summary + clock + progress + device, dtype=np.float32
        )
        if not np.isfinite(vector).all():
            raise RuntimeError(f"智能体{index}局部观测出现非有限值")
        return vector

    def local_observation_dim(self, index):
        # 统一维度：one-hot(5)+母线电压(3)+邻域摘要(5)+时间(2)+进度(1)+设备(7)。
        return 23

    def local_observation_schema(self):
        """逐智能体的局部观测语义清单（进入 manifest 与 checkpoint 绑定）。"""
        schema = []
        for index, (kind, name) in enumerate(self._agent_devices()):
            schema.append(
                {
                    "agent": index,
                    "element": f"{kind}.{name}",
                    "dim": self.local_observation_dim(index),
                    "fields": [
                        "type_onehot[5]",
                        "own_bus_voltage_pu[3]",
                        "neighborhood.v_min/v_max/v_mean/v_low_ratio/v_high_ratio",
                        "time_sin/cos",
                        "episode_progress",
                        "device_state[7]",
                    ],
                    "neighborhood_hops": int(
                        getattr(self.config, "local_observation_neighborhood", 1)
                    ),
                }
            )
        return schema

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self._next_episode = int(self.rng.integers(len(self.dataset.episodes)))
        for i, space in enumerate(self.agent_spaces):
            space.seed((seed + i) % 2**32)

    def _apply_data(self, index):
        from envs.powergrid.smartds_timeseries import SmartDSEpisodeDataset

        if isinstance(self.dataset, SmartDSEpisodeDataset):
            day = self.entry["day"]
            self.dataset.apply_step(self.circuit, day, index)
            self.current_row = self.profile[index]  # 观测上下文（全网均值）
            return
        self.current_row = row = self.profile[index]
        apply_operating_row(self.circuit, row, self.dataset, self.config.data.load_groups)

    def _pv_action(self, name, action):
        from envs.powergrid.devices import pv_command_request

        pv = self.circuit.pvs[name]
        if self.pv_act_num == float("inf"):
            if self.pv_action_mode == "q_only":
                # Q-only：有功不控（=当步可用功率满发），仅下发无功命令。
                fraction, qcommand = 1.0, float(action[0])
            else:
                fraction, qcommand = (float(action[0]) + 1) / 2, float(action[1])
        else:
            fraction, qcommand = float(action[0]) / (self.pv_act_num - 1), 0.0
        request = pv_command_request(
            pv, fraction, qcommand, self.available_pv_kw[name]
        )
        diff = abs(pv.fraction - fraction) + abs(pv.q_fraction - qcommand)
        pv.fraction, pv.q_fraction = fraction, qcommand
        return request, diff

    def _storage_action(self, name, action):
        fraction = (
            float(action[0])
            if self.bat_act_num == float("inf")
            else 2 * float(action[0]) / (self.bat_act_num - 1) - 1
        )
        return {"kw": fraction * self.circuit.batteries[name].max_kw}

    def _actions(self, action):
        from envs.powergrid.adapters import parse_actions

        return parse_actions(action, self.agent_spaces)

    def step(self, action):
        if self.t >= self.horizon or self._diverged:
            raise RuntimeError("回合已结束，请先 reset")
        try:
            return self._step(action)
        except RuntimeError as exc:
            if not self.circuit._failed:
                raise
            self._diverged = True
            return self._divergence_transition(exc)

    def _divergence_transition(self, exc):
        """物理求解失败转移：恒功率负荷电压崩溃等真实发散工况终止回合并重罚。

        大规模系统（如 9500-Node）在极端分接/电容组合下潮流无解属真实物理；
        惩罚必须超过剩余步数的最好累计奖励，避免智能体学会主动制造崩溃
        逃避损耗惩罚。观测冻结为上一步值（失败会话不能读取状态），下一次
        reset 经 circuit.reset 自动恢复（既有失败后须 reset 契约不变）。
        """
        self._on_divergence()
        self.t += 1
        reward = -float(self.divergence_penalty)
        info = {
            "physical_failure": True,
            "failure_reason": str(exc),
            "reward_total": reward,
            "episode_progress": self.t / self.horizon,
            "terminated": True,
            "truncated": False,
            "TimeLimit.truncated": False,
        }
        return self.wrap_obs(self.obs), reward, True, info

    def _step(self, action):
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
        # legacy=全量探测且保持探测设定（旧行为）；rl=仅接管名单且探测后恢复；
        # fixed=完全不探测（PV 保持模型静态设定）。
        if self.pv_control_mode == "fixed":
            self.available_pv_kw = {}
        elif self.pv_control_mode == "legacy":
            self.available_pv_kw = self.circuit.available_pv_power()
        else:
            self.available_pv_kw = self.circuit.available_pv_power(
                self.pv_control_list
            )
        requests = {}
        for name in self.pv_control_list:
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
        done = self.t == self.horizon
        reward, info = self._reward(capdiff, regdiff, soc_err, dis_err, pvdiff)
        info.update(self._info())
        info.update(
            reward_total=float(reward),
            discharge_ctrl=float(sum(dis_err)),
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
                if name not in self.pv_control_list:
                    continue
                applied = trace["PVSystem." + name]["applied"]
                if self.pv_action_mode == "q_only":
                    values.append(applied["kvar"] / pv.kva)
                else:
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
        self.physical_state = self.circuit.read_state(light=self._light_state_read)
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
        voltages = obs["bus_voltages"]
        if self._obs_buses is None:
            for value in voltages.values():
                values.extend(np.asarray(value).reshape(-1))
        else:
            for name in self._obs_buses:
                values.extend(np.asarray(voltages[name]).reshape(-1))
            values.extend(self._voltage_statistics(voltages))
        for key in (
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
        values.extend(self._task_observation())
        result = np.asarray(values, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("观测出现非有限值")
        return result

    def get_physics_manifest(self):
        elements = (
            ["Capacitor." + n for n in self.cap_names]
            + ["RegControl." + n for n in self.reg_names]
            + ["Storage." + n for n in self.bat_names]
            + ["PVSystem." + n for n in self.pv_control_list]
        )
        elements += ["Generator." + n for n in self.svg_names]
        schema = [
            {"element": element, "space": str(space)}
            for element, space in zip(elements, self.agent_spaces)
        ]
        digest = sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        observation_schema = []
        if self._obs_buses is None:
            for bus, phases in self.physical_state.voltages_pu.items():
                observation_schema.extend(
                    f"bus.{bus}.phase{phase}.voltage_pu" for phase in phases
                )
        else:
            for bus in self._obs_buses:
                phases = self.physical_state.voltages_pu.get(bus, {})
                observation_schema.extend(
                    f"bus.{bus}.phase{phase}.voltage_pu" for phase in phases
                )
            observation_schema += [
                "bus.voltage_stat.min_pu",
                "bus.voltage_stat.max_pu",
                "bus.voltage_stat.mean_pu",
                "bus.voltage_stat.violation_low_ratio",
                "bus.voltage_stat.violation_high_ratio",
            ]
        for key in (
            "cap_statuses",
            "reg_statuses",
            "bat_statuses",
            "pv_statuses",
            "svg_statuses",
        ):
            for name, value in self.obs[key].items():
                observation_schema.extend(
                    f"{key}.{name}[{i}]" for i in range(np.asarray(value).size)
                )
        observation_schema += [
            "irradiance_kw_m2",
            "temperature_c_div100",
            "load.residential",
            "load.commercial",
            "load.industrial",
            "episode_progress",
        ]
        observation_schema += self._task_observation_names()
        if len(observation_schema) != len(self.wrap_obs(self.obs)):
            raise RuntimeError("观测值与语义清单长度不一致")
        return {
            **self.circuit.manifest(),
            "smartds_dataset": (
                {
                    "split_sha256": {
                        split: self.dataset.sha256_for_split(split)
                        for split, days in self.dataset.day_indices.items() if days
                    },
                    "days": self.dataset.day_indices,
                    "mapping": self.dataset._ensure_series().mapping(),
                } if getattr(self.config, "smartds_series", None) else None
            ),
            "task": type(self.config).__name__,
            "observation_mode": self.observation_mode,
            "local_observation_schema": (
                self.local_observation_schema()
                if self.observation_mode == "local"
                else None
            ),
            "pv_control": {
                "mode": self.pv_control_mode,
                "action_mode": self.pv_action_mode,
                "controlled": list(self.pv_control_list),
                "uncontrolled": [
                    name for name in self.pv_names if name not in self.pv_control_list
                ],
                "disabled_invcontrols": {
                    name: members
                    for name, members in self.disabled_invcontrols.items()
                },
            },
            "observation_projection": self._obs_projection,
            "observation_schema": observation_schema,
            "observation_schema_sha256": sha256(
                json.dumps(observation_schema).encode()
            ).hexdigest(),
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
        return {
            **summarize_branch_stats(self.physical_state.branches),
            "step_seconds": self.circuit.step_seconds,
            "dataset_sha256": self.dataset.sha256,
            "smartds_split": (
                self.dataset.split if getattr(self.config, "smartds_series", None) else None
            ),
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
        self._reset_task(seed)
        index = self._next_episode if load_profile_idx is None else load_profile_idx
        if not 0 <= index < len(self.dataset.episodes):
            raise ValueError(f"场景编号 {index} 越界")
        self._next_episode = (index + 1) % len(self.dataset.episodes)
        self.entry, self.profile = self.dataset.episodes[index]
        self._episode_index = index
        self.t = 0
        self._diverged = False
        self.circuit.reset(seconds=float(self.profile[0, 0]))
        if self.disabled_invcontrols:
            # 每次重编译后重放 InvControl 停用（RL 接管设备的唯一写入方声明）。
            for name in self.disabled_invcontrols:
                self.circuit.command(f"Edit InvControl.{name} enabled=no")
        self._apply_data(0)
        if self.pv_control_mode == "fixed":
            # fixed：不探测不写 PV；_apply_data 的写入使量测栅栏失效，
            # 需求解一次才允许读取首步观测。
            self.available_pv_kw = {}
            self.circuit.solve()
        elif self.pv_control_mode == "legacy":
            self.available_pv_kw = self.circuit.available_pv_power()
        else:
            self.available_pv_kw = self.circuit.available_pv_power(
                self.pv_control_list
            )
        self._observe()
        return self.wrap_obs(self.obs)

    def get_obs(self, wrapped=True):
        return self.wrap_obs(self.obs) if wrapped else self.obs

    def close(self):
        if not self._closed:
            self.circuit.close()
            self._closed = True

    def _init_task(self):
        pass

    def _on_divergence(self):
        """物理失败结算钩子：任务层在提前终止前结算回合级状态（默认无）。"""

    def _reset_task(self, seed):
        pass

    def _task_observation(self):
        return []

    def _reward(self, capdiff, regdiff, soc_err, dis_err, pvdiff):
        raise NotImplementedError("具体任务必须实现奖励")

    def _task_observation_names(self):
        return []
