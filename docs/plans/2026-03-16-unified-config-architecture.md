# PowerZoo 统一配置架构重构 Design Spec

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create an implementation plan from this spec, then use superpowers:executing-plans to implement it.

**Date:** 2026-03-16
**Status:** Approved
**Scope:** 全部 4 个环境（VVC, SmartGrid, Stackelberg, DSR）的配置-环境-数据加载管线统一重构
**Supersedes:** `docs/plans/2026-03-08-34bus-pv-consolidation.md`（34Bus PV 合并作为 Phase 0 子任务嵌入本方案）

---

## 1. 问题陈述

PowerZoo 的配置加载体系存在 **4 套并行加载系统**，每个环境走完全不同的路径，导致：

| 环境 | 配置来源 | 核心问题 |
|------|---------|---------|
| VVC | `env_register._ENV_INFO` 硬编码字典 | YAML 配置完全失效 |
| SmartGrid | 私有 `ConfigLoader` + `PRESET_CONFIGS` + JSON | 三层配置源，优先级混乱 |
| Stackelberg | `_parse_config()` 手工拼装 | 键名不匹配（`num_steps` vs `max_episode_steps`），大量死参数 |
| DSR | `DEFAULT_DSR_CONFIG` + 选择性 `update_keys` | 手工列表遗漏导致死参数 |

附加问题：
- `node_systems/` 下 34Bus PV 有 4 个冗余目录（44MB，90%+ 重复）
- 系统路径解析各环境自行拼接，无共享逻辑
- `configs/sys_cfgs/` 下的 JSON 配置文件与 YAML 系统并存

## 2. 设计目标

1. **单一配置管线**：YAML → `unified_config_loader` → `env_args` dict → `{Env}Config` dataclass → 环境实例
2. **消灭所有硬编码配置**：`_ENV_INFO`、`PRESET_CONFIGS`、`DEFAULT_DSR_CONFIG`、`default_dss_files` 全部迁移到 YAML
3. **类型安全**：每个环境通过 typed dataclass 接收配置，替代 raw dict
4. **YAML 是唯一真相源**：修改 YAML 即生效，不存在被硬编码覆盖的情况
5. **向后兼容**：旧训练脚本接口 `get_defaults_yaml_args(algo, env)` 不变
6. **包含 34Bus PV 合并**：作为基础设施阶段自然嵌入

## 3. 架构设计

### 3.1 统一配置管线

```
configs/envs_cfgs/{env}.yaml          ← YAML 是唯一真相源
         ↓
configs_tools.py                      ← 保留（v1/v2 路由，极薄）
         ↓
unified_config_loader.py              ← 唯一加载器：解析 system_ref → 合并 → 返回 env_args
         ↓
env_args: dict                        ← 扁平化完整参数字典（管线中唯一的 dict 传递点）
         ↓
envs_tools.py:make_train_env()        ← 环境路由（保留 if/elif）
         ↓
┌────────────────────────────────────────────────────────┐
│  VVCConfig.from_env_args(env_args)                     │  ← 新增
│  SmartGridConfig.from_env_args(env_args)               │  ← 已有，加 from_env_args
│  StackelbergConfig.from_env_args(env_args)             │  ← 新增
│  DSRConfig.from_env_args(env_args)                     │  ← 已有，加 from_env_args
└────────────────────────────────────────────────────────┘
         ↓
{Env}.__init__(config, worker_idx)    ← 接收 typed Config，不再接收 raw dict
         ↓
OpenDSS backend (Circuits / DSRCoreEnv / StackelbergBaseEnv)
```

### 3.2 设计决策

**D1: `env_args` dict 是统一的传输协议。** `unified_config_loader` 输出扁平化 dict，之后 `Config.from_env_args()` 转为类型安全的 dataclass。

**D2: 消灭所有"二次加载"。** 每个环境不再重新读取 YAML/JSON/硬编码字典。配置从 `env_args` 一次性转换。

**D3: `configs_tools.py` 保持极薄。** 仅做 v1/v2 路由，不做合并逻辑。

**D4: `envs_tools.py` 的 if/elif 路由保持不变。** 不引入 factory pattern。每个分支内部新增 `Config.from_env_args()` 调用。

**D5: 每个环境拥有自己的 typed Config，不创建共同基类。** VVC/SmartGrid/Stackelberg/DSR 的配置参数差异太大，共同基类会沦为空壳或充满 Optional 字段。

**D6: Worker 文件管理暂不统一。** 各环境的 worker 文件需求不同（SmartGrid 有 PV irradiation/temperature，VVC 没有），抽象收益低。

## 4. Config Dataclass 设计

### 4.1 VVCConfig（新增）

```python
# envs/vvc/vvc/vvc_config.py

@dataclass
class VVCConfig:
    # 系统
    system_name: str = '13Bus'
    dss_file: str = 'IEEE13Nodeckt_daily.dss'
    source_bus: str = 'sourcebus'
    max_episode_steps: int = 24
    seed: int = 123456

    # 设备动作维度
    reg_act_num: int = 33
    bat_act_num: Union[int, float] = 33
    pv_act_num: Union[int, float] = 33  # 默认离散；PV 环境通过 YAML 设为 inf
    pv_control_enabled: bool = False

    # 奖励权重
    power_w: float = 10.0
    cap_w: float = 0.0303
    reg_w: float = 0.0303
    soc_w: float = 0.0
    dis_w: float = 0.1818

    # 显示
    node_size: int = 500
    shift: int = 10
    show_node_labels: bool = True
    load_noise: bool = True

    # 运行时
    scale: float = 1.0
    use_render: bool = False
    useS: bool = False
    record_node: bool = False
    dss_act: bool = False

    @classmethod
    def from_env_args(cls, env_args: dict) -> 'VVCConfig':
        """字段反射 + 键名别名映射"""
        ...
```

替代 `env_register.py` 的 `_ENV_INFO`（约 24 个硬编码条目，含 VVC/DSR/Stackelberg 交叉条目）和 `_SYS_INFO`（4 个系统固定信息）。旧变体名通过 `unified_config_loader` 的 legacy mapping 自动展开为参数组合。注意：`_ENV_INFO` 中的 DSR/Stackelberg 条目（如 `dsr_13bus`、`stackelberg_34bus`）需在 Phase 4 中审计是否为死代码——这些环境已有独立的配置路径，这些条目可能是历史遗留。

### 4.2 SmartGridConfig（已有，修复加载链）

```python
# envs/smartgrid/base_env/env_config.py（修改）

@dataclass
class SmartGridConfig:
    # 现有字段全部保留
    # ...

    # 新增
    pv_plan: Optional[str] = None  # PV 方案选择

    @classmethod
    def from_env_args(cls, env_args: dict) -> 'SmartGridConfig':
        """替代原有的 ConfigLoader.get_config()"""
        ...
```

删除 SmartGrid 私有的 `config_loader.py`、`PRESET_CONFIGS`。

### 4.3 StackelbergConfig（新增）

```python
# envs/stackelberg/stackelberg_config.py

@dataclass
class StackelbergConfig:
    # 系统
    system_name: str = '13Bus'
    dss_file: str = 'IEEE13Nodeckt_daily.dss'
    max_episode_steps: int = 24
    seed: int = 123456

    # 智能体
    n_consumer_agents: int = 8

    # 子配置（dict 类型，结构灵活）
    tou_config: Optional[dict] = None
    tier_config: Optional[dict] = None
    reward_weights: Optional[dict] = None
    load_aggregation: Optional[dict] = None
    async_config: Optional[dict] = None
    monitoring_config: Optional[dict] = None
    n1_security: Optional[dict] = None

    # 运行时
    worker_idx: Optional[int] = None
    use_render: bool = False
    scale: float = 1.0

    @classmethod
    def from_env_args(cls, env_args: dict) -> 'StackelbergConfig':
        """字段反射 + dict 子配置直传"""
        ...
```

修复键名不匹配（`num_steps` → `max_episode_steps`），消灭 `_parse_config()` 中的 `default_dss_files` 硬编码。

### 4.4 DSRConfig（已有，加 `from_env_args`）

```python
# envs/dsr/core/config.py（修改）

@dataclass
class DSRConfig:
    # 现有 50+ 字段全部保留
    # ...

    @classmethod
    def from_env_args(cls, env_args: dict) -> 'DSRConfig':
        """字段反射，替代手工 update_keys 列表"""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in env_args.items() if k in valid_fields and v is not None}
        return cls(**kwargs)
```

删除 `DEFAULT_DSR_CONFIG` 全局实例和 `_parse_config()` 中的 `update_keys` 列表。

### 4.5 构造函数签名变化

| 环境 | 当前 | 改造后 |
|------|------|--------|
| VVCEnv | `__init__(self, args: dict, rank)` | `__init__(self, config: VVCConfig, rank)` |
| SmartGrid Env | `__init__(self, info, dss_act, mode, worker_idx)` | `__init__(self, config: SmartGridConfig, worker_idx)` |
| SmartGrid VVCEnv | `__init__(self, base_env, env_args, rank)` | `__init__(self, base_env, config: SmartGridConfig, rank)` |
| StackelbergVVCEnv | `__init__(self, args: dict)` | `__init__(self, config: StackelbergConfig, rank)` |
| DSREnv | `__init__(self, args: dict, rank)` | `__init__(self, config: DSRConfig, rank)` |

## 5. YAML 配置迁移

### 5.1 消灭清单

| 硬编码 | 文件 | 迁移目标 |
|--------|------|---------|
| `_ENV_INFO` (~24 条目) | `envs/vvc/vvc/env_register.py:62-200` | `configs/systems/*.yaml` + `configs/envs_cfgs/vvc.yaml` |
| `_SYS_INFO` (4 系统) | `envs/vvc/vvc/env_register.py:27-58` | `configs/systems/*.yaml` |
| `PRESET_CONFIGS` | `envs/smartgrid/base_env/env_config.py` | `configs/systems/*.yaml` |
| `environments_info.json` | `configs/sys_cfgs/environments_info.json` | `configs/systems/*.yaml` |
| `system_info.json` | `configs/sys_cfgs/system_info.json` | `configs/systems/*.yaml` |
| `DEFAULT_DSR_CONFIG` | `envs/dsr/core/config.py` | `DSRConfig` dataclass 默认值 |
| `default_dss_files` | `envs/stackelberg/stackelberg_vvc_env.py:112-116` | `configs/systems/*.yaml` |

### 5.2 `configs/systems/*.yaml` 统一格式

```yaml
system:
  name: 13Bus
  dss_folder: node_systems/13Bus
  dss_file: IEEE13Nodeckt_daily.dss
  source_bus: sourcebus

episode:
  max_steps: 24

devices:
  capacitors: {count: 2, action_space: discrete}
  regulators: {count: 2, action_num: 33}
  batteries: {count: 2, action_num: 33}
  pv_systems: {count: 0, control_enabled: false}

constraints:
  voltage_min: 0.95
  voltage_max: 1.05

default_rewards:
  power_loss: 10.0
  capacitor: 0.0303
  regulator: 0.0303
  battery_soc: 0.0
  battery_discharge: 0.1818

display:
  node_size: 500
  shift: 10
  show_labels: true
  load_noise: true
```

### 5.3 VVC 旧变体兼容

`_ENV_INFO` 的 13 个变体本质是系统 × 设备配置 × 奖励权重的组合。迁移策略：

- 系统信息 → `configs/systems/{System}.yaml`
- 变体参数 → `configs/envs_cfgs/vvc.yaml` 中直接设置

保留 legacy mapping 支持旧 `env_name`：

```python
# unified_config_loader.py
VVC_LEGACY_VARIANTS = {
    '13Bus_cbat':     {'system_ref': '13Bus', 'bat_act_num': float('inf')},
    '13Bus_soc':      {'system_ref': '13Bus', 'soc_w': 0.606},
    '13Bus_cbat_soc': {'system_ref': '13Bus', 'bat_act_num': float('inf'), 'soc_w': 0.606},
    '34Bus_pv':       {'system_ref': '34Bus_PV', 'pv_control_enabled': True},
    # ...
}
```

### 5.4 34Bus PV 合并

作为 Phase 0 基础设施的一部分：

- 4 个冗余目录 → 1 个 `node_systems/34Bus_PV/` + `pv_plans/` 子目录
- `configs/systems/34Bus_PV.yaml` 新增 `pv_plans` 段
- `configs/envs_cfgs/smartgrid.yaml` 新增 `pv_plan: aggressive`
- `circuit.py:_create_temp_compile_file()` 根据 `pv_plan` 参数选择 DSS 文件
- 删除 `34Bus_PV_Aggressive.yaml`、`34Bus_PV_Conservative.yaml`、`34Bus_PV_Optimized.yaml`

### 5.5 系统路径解析统一

```python
# utils/path_utils.py（扩展）

def resolve_system_path(system_name: str) -> Path:
    """将 system_name 解析为 node_systems 下的绝对路径"""
    project_root = get_project_root()
    if 'node_systems' in str(system_name):
        candidate = project_root / system_name
        if candidate.exists():
            return candidate
    system_dir = project_root / 'node_systems' / system_name
    if system_dir.exists():
        return system_dir
    raise FileNotFoundError(f"System '{system_name}' not found at {system_dir}")
```

所有环境统一使用此函数替代各自的路径拼接。

## 6. `envs_tools.py` 改造

`envs_tools.py` 是管线中 dict → typed Config 的分界线。改造后每个分支新增 `Config.from_env_args()` 调用：

```python
def make_train_env(env_name, seed, n_threads, env_args):
    def get_env_fn(rank):
        def init_env():
            if env_name in ("vvc", "powerzoo"):
                config = VVCConfig.from_env_args(env_args)
                env = VVCEnv(config, rank)
            elif env_name == "smartgrid":
                config = SmartGridConfig.from_env_args(env_args)
                base_env = Env(config, worker_idx=rank)
                env = VVCEnv(base_env, config, rank)
            elif env_name == "dsr":
                config = DSRConfig.from_env_args(env_args)
                env = DSREnv(config, rank)
            elif env_name.startswith("stackelberg"):
                config = StackelbergConfig.from_env_args({**env_args, 'env_name': env_name})
                env = StackelbergVVCEnv(config, rank)
            # ... 其他环境保持不变 ...
            env.seed(seed + rank * 1000)
            return env
        return init_env
    # SubprocVecEnv / DummyVecEnv 逻辑不变
```

`make_eval_env()` 和 `make_render_env()` 做同样改造。辅助函数（`get_num_agents` 等）不变。

## 7. 文件变更总结

### 新增

| 文件 | 说明 |
|------|------|
| `envs/vvc/vvc/vvc_config.py` | VVCConfig dataclass |
| `envs/stackelberg/stackelberg_config.py` | StackelbergConfig dataclass |
| `node_systems/34Bus_PV/pv_plans/*.dss` | 4 个 PV 方案 DSS 文件 |

### 修改

| 文件 | 变更 |
|------|------|
| `envs/smartgrid/base_env/env_config.py` | 加 `from_env_args()`、`pv_plan` 字段；删 `PRESET_CONFIGS` |
| `envs/dsr/core/config.py` | 加 `from_env_args()`；删 `DEFAULT_DSR_CONFIG` |
| `envs/vvc/vvc/env_register.py` | 删 `_ENV_INFO`/`_SYS_INFO`；`make_base_env` 改接 `VVCConfig` |
| `envs/vvc/vvc_env.py` | `__init__` 改接 `VVCConfig` |
| `envs/stackelberg/stackelberg_vvc_env.py` | 删 `_parse_config()`；`__init__` 改接 `StackelbergConfig` |
| `envs/dsr/dsr_env.py` | 删 `_parse_config()`；`__init__` 改接 `DSRConfig` |
| `envs/smartgrid/base_env/env.py` | `__init__` 改接 `SmartGridConfig`；`folder_path` 由 `resolve_system_path(config.system_name)` 提供 |
| `envs/smartgrid/base_env/vvc_env.py` | `__init__` 改接 `SmartGridConfig` 替代 raw dict |
| `envs/smartgrid/base_env/env_register.py` | `make_base_env` 改接 `SmartGridConfig`；删除 `config_loader` 导入 |
| `envs/smartgrid/circuit_system/circuit.py` | 新增 `pv_plan` 参数 |
| `utils/envs_tools.py` | 各分支加 `Config.from_env_args()` |
| `utils/unified_config_loader.py` | 加 VVC legacy variant 映射 |
| `utils/path_utils.py` | 加 `resolve_system_path()` |
| `configs/envs_cfgs/vvc.yaml` | 加 `system_ref` |
| `configs/envs_cfgs/smartgrid.yaml` | 简化；加 `pv_plan` |
| `configs/envs_cfgs/stackelberg_*.yaml` | 确保 `system_ref` |
| `configs/envs_cfgs/dsr*.yaml` | 确保 `system_ref` |
| `configs/systems/13Bus.yaml` | 补全 VVC 所需字段 |
| `configs/systems/34Bus.yaml` | 补全 VVC 所需字段 |
| `configs/systems/34Bus_PV.yaml` | 加 `pv_plans` 段 |
| `configs/systems/_registry.yaml` | 清理变体条目 |

### 删除

| 文件 | 理由 |
|------|------|
| `envs/smartgrid/base_env/config_loader.py` | 私有 ConfigLoader → 统一走 unified_config_loader |
| `configs/sys_cfgs/environments_info.json` | JSON → YAML |
| `configs/sys_cfgs/system_info.json` | JSON → YAML |
| `configs/systems/34Bus_PV_Aggressive.yaml` | 合并入 34Bus_PV + pv_plan |
| `configs/systems/34Bus_PV_Conservative.yaml` | 同上 |
| `configs/systems/34Bus_PV_Optimized.yaml` | 同上 |

### 归档

| 目录 | 处理 |
|------|------|
| `node_systems/34Bus_PV_Aggressive/` | 移入 `node_systems/_archived/` |
| `node_systems/34Bus_PV_Conservative/` | 同上 |
| `node_systems/34Bus_PV_Optimized/` | 同上 |

## 8. 迁移顺序

```
Phase 0: 基础设施
  - configs/systems/*.yaml 补全（13Bus, 34Bus 等）
  - path_utils.py 扩展 resolve_system_path()
  - unified_config_loader 加 legacy variant 映射
  - 34Bus PV 目录合并 + pv_plans/

Phase 1: SmartGrid（最容易）
  - SmartGridConfig.from_env_args()（新增，替代 from_dict）
  - 删除 config_loader.py / PRESET_CONFIGS / JSON
  - 更新 env_register.py 移除 config_loader 导入，改接 SmartGridConfig
  - Env(config) + vvc_env.py VVCEnv(base_env, config) 改造
  - circuit.py pv_plan 注入
  - **同步更新 envs_tools.py 的 smartgrid 分支**

Phase 2: DSR（次容易）
  - DSRConfig.from_env_args()（字段反射替代 update_keys）
  - 删除 DEFAULT_DSR_CONFIG 及 DSR_*BUS_CONFIG 预设常量
  - DSREnv(config) 改造
  - **同步更新 envs_tools.py 的 dsr 分支**

Phase 3: Stackelberg（中等）
  - 新建 StackelbergConfig
  - 修复键名不匹配
  - StackelbergVVCEnv(config, rank) 改造
  - **同步更新 envs_tools.py 的 stackelberg 分支**

Phase 4: VVC（最难）
  - 新建 VVCConfig
  - _ENV_INFO/_SYS_INFO → YAML 迁移 + legacy mapping
  - **审计 _ENV_INFO 全部条目**（含 DSR/Stackelberg 交叉引用条目，确认是否为死代码）
  - env_register.py 重构（make_base_env 改接 VVCConfig，Env 构造 folder_path 改由 resolve_system_path 提供）
  - VVCEnv(config, rank) 改造
  - **同步更新 envs_tools.py 的 vvc/powerzoo 分支**

Phase 5: 清理
  - envs_tools.py 最终审查（所有分支已在 Phase 1-4 逐步改造完毕）
  - configs_tools.py 清理
  - 删除死文件、归档旧目录
  - 更新 CLAUDE.md / _registry.yaml
```

## 9. 测试策略

每个 Phase 通过 3 层测试：

**Layer 1 — Config 单元测试**
- `from_env_args()` 正确转换
- 未知键静默忽略
- 默认值回退
- 键名别名映射

**Layer 2 — 管线集成测试**
- YAML → unified_config_loader → env_args → Config → env.reset() → env.step()
- 验证 obs/action 形状与基线一致

**Layer 3 — 短训练回归测试**
- 100 步 HAPPO 训练，验证无 crash
- reward 在合理范围

### 基线快照

每个 Phase 开始前采集环境指纹（n_agents, obs_shape, action_shape, reward_range）。Phase 完成后重新采集对比。obs/action 形状必须完全一致。

## 10. 风险与回滚

| 风险 | 缓解 |
|------|------|
| VVC _ENV_INFO 迁移遗漏某变体 | legacy mapping + 测试覆盖所有 13 个变体名 |
| SmartGrid 删 config_loader.py 后有隐含依赖 | grep 全项目搜索引用 |
| from_env_args() 丢失嵌套 dict 参数 | Layer 2 测试覆盖嵌套参数路径 |
| 34Bus PV sinterval 不一致 | Phase 0 中对齐并验证 |
| 多 worker 并行 worker 文件路径出错 | n_threads=4 集成测试 |
| `_ENV_INFO` 含 DSR/Stackelberg 交叉引用条目 | Phase 4 审计 `get_info_and_folder()` 全部调用点 |
| SmartGrid `env_register.py` 导入 `config_loader.py` | Phase 1 必须同步移除导入并替换调用 |
| DSR 存在 `DSR_13BUS_CONFIG` 等预设常量 | Phase 2 审计并清理所有预设常量引用 |

### 回滚策略

每个 Phase 独立 feature branch，通过 3 层测试后才 merge。失败时 revert 该 branch 的 merge commit。

## 11. 明确排除

| 排除项 | 理由 |
|--------|------|
| 统一 worker 文件管理 | 各环境需求不同 |
| Config 共同基类 | 配置差异太大，基类空壳 |
| 重构 runner/buffer 层 | 与配置架构无关 |
| 修改训练脚本接口 | `get_defaults_yaml_args()` 接口保持不变 |
| env registry / factory pattern | 环境数少，if/elif 更清晰 |
| `district_dispatch` / `lag` 环境改造 | 非核心电力仿真环境，保持现状（envs_tools.py 中分支不动） |
