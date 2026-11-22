"""静态训练能力声明。导入本模块不会加载训练器或仿真引擎。"""

from dataclasses import asdict, dataclass
from math import inf
from pathlib import Path
from types import MappingProxyType

from utils.config_validation import ConfigError, environment_family
from utils.discrete_marl_config import DISCRETE_VVC_ALGORITHMS, validate_discrete_vvc
from utils.environment_registry import ENVIRONMENTS

# 与 CBESS info["costs"] 和 LDCPPO 约束 critic 的通道对应。
_COST_KEYS = ("VD", "VN", "LO")


@dataclass(frozen=True)
class AlgorithmCapability:
    entrypoint: str
    actions: tuple[str, ...]
    observations: tuple[str, ...]
    state: str
    sharing: tuple[str, ...]
    required_mask: str = "none"
    required_costs: tuple[str, ...] = ()
    termination: str = "terminated_and_truncated"
    checkpoint: tuple[str, ...] = ("policy_reload",)
    evaluation: str = "validation_worker"
    unavailable: str = ""


@dataclass(frozen=True)
class EnvironmentCapability:
    entrypoint: str
    actions: tuple[str, ...]
    device_actions: tuple[str, ...]
    device_ids: str
    observations: tuple[str, ...]
    state: str
    reward: str
    mask: str
    physical_limits: str
    costs: tuple[str, ...]
    termination: str
    evaluation: str
    checkpoint_binding: str
    action_encoding: str


@dataclass(frozen=True)
class DeviceLayout:
    device_ids: tuple[str, ...]
    actions: tuple[str, ...]
    homogeneous: bool

    def __post_init__(self):
        if (
            not self.device_ids
            or len(self.device_ids) != len(self.actions)
            or any(
                not isinstance(name, str) or not name.strip()
                for name in self.device_ids
            )
            or len({name.casefold() for name in self.device_ids})
            != len(self.device_ids)
        ):
            raise ConfigError("动作清单必须包含非空且唯一的有序设备 ID")
        if set(self.actions) - {"discrete", "continuous"}:
            raise ConfigError("设备动作空间必须是 Discrete 或 Box")

    @property
    def kind(self):
        return self.actions[0] if len(set(self.actions)) == 1 else "mixed"


def device_layout(wrapper):
    """已构造同步适配器的动作身份来自当前物理清单，空间来自实际接口。"""
    from gymnasium.spaces import Box, Discrete

    schema = wrapper.get_physics_manifest()["action_schema"]
    spaces = wrapper.action_space
    observations = wrapper.observation_space
    if not spaces or len(spaces) != len(schema) or len(spaces) != len(observations):
        raise ConfigError("设备、动作空间和观测空间数量不一致")
    kinds, ids = [], []
    for entry, space in zip(schema, spaces):
        if "group" in entry:
            ids.append(entry["group"])
            matches = isinstance(space, Discrete) and space.n == len(entry["choices"])
        else:
            ids.append(entry["element"])
            matches = entry["space"] == str(space)
        if not matches:
            raise ConfigError("物理清单空间与适配器空间不一致")
        if isinstance(space, Discrete):
            kinds.append("discrete")
        elif isinstance(space, Box):
            kinds.append("continuous")
        else:
            raise ConfigError(f"未知设备动作空间 {type(space).__name__}")
    return DeviceLayout(
        tuple(ids),
        tuple(kinds),
        all(s == spaces[0] for s in spaces)
        and all(s == observations[0] for s in observations),
    )


_MARL_OBS = ("global", "local")
_HETEROGENEOUS = ("discrete", "continuous", "mixed")
_ON_POLICY = AlgorithmCapability(
    "marl",
    _HETEROGENEOUS,
    _MARL_OBS,
    "team_global",
    ("independent", "group", "all"),
)
_CONTINUOUS = AlgorithmCapability(
    "marl",
    ("continuous",),
    _MARL_OBS,
    "team_global",
    ("independent", "all"),
)
_DISCRETE_VVC = AlgorithmCapability(
    "marl",
    ("discrete",),
    ("local",),
    "team_global",
    ("independent",),
    required_mask="structural",
    termination="terminated_and_truncated_if_enabled",
)
_ALGORITHMS = MappingProxyType(
    {
        **{name: _ON_POLICY for name in ("happo", "hatrpo", "haa2c", "mappo", "shom")},
        **{name: _CONTINUOUS for name in ("haddpg", "hatd3", "maddpg", "matd3")},
        "sb3_ppo": AlgorithmCapability(
            "centralized",
            ("continuous", "multidiscrete"),
            ("global",),
            "none",
            ("single",),
            evaluation="policy_predict",
        ),
        "ldcppo": AlgorithmCapability(
            "centralized",
            ("continuous",),
            ("global",),
            "none",
            ("single",),
            required_costs=_COST_KEYS,
            termination="legacy_done_only",
            checkpoint=("training_resume",),
            evaluation="unprofiled",
        ),
        **{name: _DISCRETE_VVC for name in sorted(DISCRETE_VVC_ALGORITHMS)},
    }
)

# 仅声明已核对消费者的形态。其它已注册家族会返回环境未声明错误。
_VVC = EnvironmentCapability(
    "marl",
    _HETEROGENEOUS,
    _HETEROGENEOUS,
    "physics_manifest.action_schema",
    _MARL_OBS,
    "team_global",
    "team",
    "structural",
    "native_readback",
    (),
    "finite_horizon_or_physical_failure",
    "validation_split",
    "physics_manifest",
    "per_device",
)
_ENVIRONMENT_CAPABILITIES = MappingProxyType(
    {
        ("vvc", "marl"): _VVC,
        ("vvc", "centralized"): EnvironmentCapability(
            "centralized",
            ("continuous",),
            _HETEROGENEOUS,
            _VVC.device_ids,
            ("global",),
            "none",
            "single",
            "none",
            "native_readback",
            (),
            _VVC.termination,
            _VVC.evaluation,
            _VVC.checkpoint_binding,
            "box_quantizes_discrete_devices",
        ),
        ("dsr", "marl"): EnvironmentCapability(
            "marl",
            ("discrete",),
            ("discrete",),
            "physics_manifest.action_schema.group",
            ("global",),
            "team_global",
            "team",
            "feasible",
            "native_readback",
            (),
            "finite_horizon",
            "validation_split",
            "physics_manifest",
            "per_group",
        ),
        ("dsr", "centralized"): EnvironmentCapability(
            "centralized",
            ("multidiscrete",),
            ("discrete",),
            "physics_manifest.action_schema.group",
            ("global",),
            "none",
            "single",
            "none",
            "native_readback",
            (),
            "finite_horizon",
            "validation_split",
            "physics_manifest",
            "per_group",
        ),
        ("cbess", "centralized"): EnvironmentCapability(
            "centralized",
            ("continuous",),
            ("continuous",),
            "backend_layout",
            ("global",),
            "none",
            "single",
            "none",
            "native_readback",
            _COST_KEYS,
            "finite_horizon",
            "train_only_reset",
            "unavailable",
            "native_box",
        ),
    }
)


def discover_capabilities():
    """返回可序列化声明，包含未声明的已注册环境。"""
    return {
        "schema_version": 1,
        "algorithms": {name: asdict(spec) for name, spec in _ALGORITHMS.items()},
        "environments": {
            family: {
                "interaction": registered.interaction,
                "profiles": {
                    entrypoint: asdict(spec)
                    for (name, entrypoint), spec in _ENVIRONMENT_CAPABILITIES.items()
                    if name == family
                },
            }
            for family, registered in ENVIRONMENTS.items()
        },
    }


@dataclass(frozen=True)
class Preflight:
    algorithm: str
    environment: str
    system: str
    entrypoint: str
    action_kinds: tuple[str, ...] = ()
    device_action_kinds: tuple[str, ...] = ()
    observation: str = ""
    action_encoding: str = ""
    device_ids: tuple[str, ...] = ()
    sharing: str = ""
    deferred: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def compatible(self):
        return not self.issues

    def to_dict(self):
        return {**asdict(self), "compatible": self.compatible}


_STANDARD_VVC_SYSTEMS = {"13Bus", "34Bus", "123Bus"}


def _vvc_device_actions(values):
    if values.get("continuous_storage_only"):
        return ("continuous",)
    if values["bat_act_num"] != inf:
        return ("discrete",)
    system = values["system_name"]
    folder = values.get("dss_folder")
    bundled = Path("node_systems") / system
    root = Path(__file__).resolve().parents[1]
    if (
        system in _STANDARD_VVC_SYSTEMS
        and (not folder or Path(folder) in (bundled, root / bundled))
        and values.get("dss_file") == "network.dss"
        and not values.get("overlays")
        and not values.get("smartds_series")
    ):
        # 标准馈线保留原生机械设备，储能切到连续后为混合动作。
        return ("mixed",)
    return _HETEROGENEOUS


def preflight(
    config,
    *,
    backend=None,
    layout=None,
    require_mask="none",
    checkpoint=None,
    independent_replay=False,
):
    """校验已解析的 ConfigLoader 配置，可选使用物理环境绑定结果复核。"""
    if require_mask not in ("none", "structural", "feasible"):
        raise ConfigError(f"未知掩码要求 {require_mask!r}")
    if checkpoint not in (None, "policy_reload", "training_resume"):
        raise ConfigError(f"未知 checkpoint 要求 {checkpoint!r}")
    algorithm = backend if backend is not None else config.main_args["algo"]
    entrypoint = "centralized" if backend is not None else "marl"
    family = environment_family(config.main_args["env"])
    values = config.env_args
    system = values.get("system_name", values.get("system", family))
    selection = (algorithm, family, system, entrypoint)
    algo = _ALGORITHMS.get(algorithm)
    env = _ENVIRONMENT_CAPABILITIES.get((family, entrypoint))
    issues, deferred = [], []
    if algo is None or algo.entrypoint != entrypoint:
        issues.append(f"未知或未声明的{entrypoint}算法/后端 {algorithm!r}")
    elif algo.unavailable:
        issues.append(algo.unavailable)
    if env is None:
        issues.append(f"{family!r} 没有公开的 {entrypoint} 能力声明")
    if algorithm in DISCRETE_VVC_ALGORITHMS and entrypoint == "marl":
        try:
            validate_discrete_vvc(algorithm, family, config.algo_args, values)
        except ConfigError as exc:
            issues.append(str(exc))
    if (
        family == "vvc"
        and entrypoint == "marl"
        and values.get("continuous_storage_only")
        and algorithm in ("haddpg", "hatd3")
    ):
        issues.append(
            f"{algorithm} 尚未支持 VVC 原生连续储能模式（仅 MADDPG/MATD3 已接入）"
        )
    if issues:
        return Preflight(*selection, issues=tuple(issues))

    device_actions = _vvc_device_actions(values) if family == "vvc" else env.device_actions
    if layout is not None:
        if layout.kind not in device_actions:
            issues.append(
                f"实际设备动作 {layout.kind} 与静态声明 {device_actions} 不符"
            )
        device_actions = (layout.kind,)
    elif env.device_ids.startswith("physics_manifest"):
        deferred.append("按物理清单绑定设备 ID 及动作/观测形状")
    action_encoding = env.action_encoding
    if family == "vvc" and backend is not None and device_actions == ("continuous",):
        action_encoding = "native_box"
    actions = device_actions if entrypoint == "marl" else env.actions
    if not set(actions).intersection(algo.actions):
        issues.append(f"{algorithm} 动作支持 {algo.actions}，环境提供 {actions}")
    elif not set(actions).issubset(algo.actions):
        deferred.append(f"复核实际动作空间是否属于 {algo.actions}")

    observation = (
        values.get("observation_mode", "global") if entrypoint == "marl" else "global"
    )
    configured_sharing = config.algo_args.get("algo", {}).get("share_param", False)
    sharing = (
        "single"
        if entrypoint == "centralized"
        else {False: "independent", True: "all", "group": "group"}[configured_sharing]
    )
    if observation not in algo.observations or observation not in env.observations:
        issues.append(f"不支持观测模式 {observation!r}")
    if algo.state != env.state or (
        env.state == "team_global" and values.get("state_type") != "EP"
    ):
        issues.append("集中团队状态需要 env.state_type=EP")
    if sharing not in algo.sharing:
        issues.append(f"{algorithm} 不支持策略共享方式 {sharing}")
    if sharing == "all":
        if layout is None:
            deferred.append("全共享要求实际动作和观测空间同构")
        elif not layout.homogeneous:
            issues.append("全共享要求实际动作和观测空间同构")

    mask_order = ("none", "structural", "feasible")
    mask = "none" if family == "vvc" and device_actions == ("continuous",) else env.mask
    if mask_order.index(mask) < max(
        mask_order.index(require_mask), mask_order.index(algo.required_mask)
    ):
        issues.append(f"动作掩码能力不足：{family}/{entrypoint} 提供 {mask}")
    missing_costs = set(algo.required_costs) - set(env.costs)
    if missing_costs:
        issues.append(f"环境缺少约束成本通道 {sorted(missing_costs)}")
    if checkpoint and checkpoint not in algo.checkpoint:
        issues.append(f"{algorithm} 不支持 {checkpoint}")
    if (
        checkpoint == "policy_reload" or independent_replay
    ) and env.checkpoint_binding == "unavailable":
        issues.append(f"{family} 缺少公开的 checkpoint 物理绑定")
    smartds = values.get("smartds_series") or {}
    if independent_replay and (
        env.evaluation != "validation_split"
        or algo.evaluation not in ("validation_worker", "policy_predict")
        or values.get("data", {}).get("split") != "train"
        or smartds.get("split", "train") != "train"
        or (smartds and not smartds.get("validation_days"))
    ):
        issues.append("独立回放要求公开验证集合入口和 train 训练配置")

    return Preflight(
        *selection,
        actions,
        device_actions,
        observation=observation,
        action_encoding=action_encoding,
        device_ids=layout.device_ids if layout is not None else (),
        sharing=sharing,
        deferred=tuple(deferred),
        issues=tuple(issues),
    )
