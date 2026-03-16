# PowerZoo 统一配置架构 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify PowerZoo's 4 parallel config loading systems into a single pipeline: YAML → unified_config_loader → typed Config dataclass → environment constructor.

**Architecture:** Each Phase independently migrates one environment. Phase 0 builds shared infrastructure (system YAML completion, path utils, 34Bus PV consolidation). Phases 1-4 each add a `from_env_args()` classmethod to the environment's Config dataclass, update the env constructor to accept typed Config, update the corresponding `envs_tools.py` branch, and delete the old loading code. Phase 5 cleans up.

**Tech Stack:** Python 3.10+, OpenDSS (dss-python), YAML config, pytest, dataclasses

**Spec:** `docs/plans/2026-03-16-unified-config-architecture.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `envs/vvc/vvc/vvc_config.py` | VVCConfig dataclass with `from_env_args()` |
| `envs/stackelberg/stackelberg_config.py` | StackelbergConfig dataclass with `from_env_args()` |
| `node_systems/34Bus_PV/pv_plans/none.dss` | PV plan: no PV systems |
| `node_systems/34Bus_PV/pv_plans/conservative.dss` | PV plan: 5 PV, 720kW |
| `node_systems/34Bus_PV/pv_plans/optimized.dss` | PV plan: 7 PV, 900kW |
| `node_systems/34Bus_PV/pv_plans/aggressive.dss` | PV plan: 9 PV, 1080kW |
| `tests/test_unified_config.py` | Config unit tests for all environments |
| `tests/test_config_pipeline.py` | Integration tests: YAML → Config → env |

### Modified Files
| File | Change |
|------|--------|
| `utils/path_utils.py` | Add `resolve_system_path()` |
| `utils/unified_config_loader.py` | Add VVC legacy variant mapping; add `pv_plan` to system config extraction |
| `utils/envs_tools.py` | Each branch: `Config.from_env_args()` before env construction |
| `configs/systems/13Bus.yaml` | Add `display.load_noise`, `display.show_node_labels` |
| `configs/systems/34Bus.yaml` | Add `display.load_noise`, `display.show_node_labels` |
| `configs/systems/34Bus_PV.yaml` | Add `pv_plans` metadata section |
| `configs/systems/_registry.yaml` | Remove PV variant entries, simplify |
| `configs/envs_cfgs/smartgrid.yaml` | Change `system_ref: 34Bus_PV`, add `pv_plan: aggressive`, remove `environment_specific.system_name` duplication |
| `configs/envs_cfgs/vvc.yaml` | Already has `system_ref: 13Bus` — OK |
| `envs/smartgrid/base_env/env_config.py` | Add `SmartGridConfig.from_env_args()`, `pv_plan` field; delete `PRESET_CONFIGS` |
| `envs/smartgrid/base_env/env.py` | Constructor accepts `SmartGridConfig`; use `resolve_system_path()` |
| `envs/smartgrid/base_env/vvc_env.py` | Constructor accepts `SmartGridConfig` instead of raw dict |
| `envs/smartgrid/base_env/env_register.py` | Remove `config_loader` import; `make_base_env` accepts `SmartGridConfig` |
| `envs/smartgrid/circuit_system/circuit.py` | Add `pv_plan` parameter to `__init__` and `_create_temp_compile_file` |
| `envs/dsr/core/config.py` | Add `DSRConfig.from_env_args()` |
| `envs/dsr/dsr_env.py` | Constructor accepts `DSRConfig`; delete `_parse_config()` |
| `envs/stackelberg/stackelberg_vvc_env.py` | Constructor accepts `StackelbergConfig`; delete `_parse_config()` |
| `envs/vvc/vvc/env_register.py` | Delete `_ENV_INFO`/`_SYS_INFO`; `make_base_env` accepts `VVCConfig` |
| `envs/vvc/vvc_env.py` | Constructor accepts `VVCConfig` |

### Deleted Files
| File | Reason |
|------|--------|
| `envs/smartgrid/base_env/config_loader.py` | Replaced by unified_config_loader |
| `configs/systems/34Bus_PV_Aggressive.yaml` | Merged into 34Bus_PV + pv_plan |
| `configs/systems/34Bus_PV_Conservative.yaml` | Same |
| `configs/systems/34Bus_PV_Optimized.yaml` | Same |

---

## Phase 0: Infrastructure

### Task 0.1: Add `resolve_system_path()` to path_utils

**Files:**
- Modify: `utils/path_utils.py`
- Test: `tests/test_unified_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_unified_config.py
"""Unified config architecture tests"""
import pytest
from pathlib import Path


class TestResolveSystemPath:
    """Test resolve_system_path utility"""

    def test_resolves_13bus(self):
        from utils.path_utils import resolve_system_path
        path = resolve_system_path('13Bus')
        assert path.exists()
        assert path.name == '13Bus'
        assert 'node_systems' in str(path)

    def test_resolves_34bus_pv(self):
        from utils.path_utils import resolve_system_path
        path = resolve_system_path('34Bus_PV')
        assert path.exists()

    def test_resolves_with_prefix(self):
        from utils.path_utils import resolve_system_path
        path = resolve_system_path('node_systems/13Bus')
        assert path.exists()
        assert path.name == '13Bus'

    def test_raises_for_nonexistent(self):
        from utils.path_utils import resolve_system_path
        with pytest.raises(FileNotFoundError):
            resolve_system_path('NonExistent999Bus')
```

- [ ] **Step 2: Run test — verify failure**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestResolveSystemPath -v --tb=short 2>&1 | tail -15`
Expected: FAIL — `resolve_system_path` not found

- [ ] **Step 3: Implement `resolve_system_path`**

Read `utils/path_utils.py` first. Then add at the end:

```python
def resolve_system_path(system_name: str) -> Path:
    """将 system_name 解析为 node_systems 下的绝对路径

    Args:
        system_name: 系统名称（如 '13Bus', '34Bus_PV'）
                     或已含 'node_systems/' 前缀的路径

    Returns:
        node_systems/{system_name} 的绝对路径

    Raises:
        FileNotFoundError: 系统目录不存在
    """
    project_root = get_project_root()

    # 已含路径前缀
    if 'node_systems' in str(system_name):
        candidate = project_root / system_name
        if candidate.exists():
            return candidate

    # 标准解析
    system_dir = project_root / 'node_systems' / system_name
    if system_dir.exists():
        return system_dir

    raise FileNotFoundError(f"System '{system_name}' not found at {system_dir}")
```

- [ ] **Step 4: Run test — verify pass**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestResolveSystemPath -v --tb=short`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add utils/path_utils.py tests/test_unified_config.py
git commit -m "feat: add resolve_system_path utility for unified system directory resolution"
```

---

### Task 0.2: Add VVC legacy variant mapping to unified_config_loader

**Files:**
- Modify: `utils/unified_config_loader.py`
- Test: `tests/test_unified_config.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_unified_config.py`:

```python
class TestLegacyVariantMapping:
    """Test VVC legacy env_name → config expansion"""

    def test_13bus_cbat_expands(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('13Bus_cbat')
        assert result['system_ref'] == '13Bus'
        assert result['bat_act_num'] == float('inf')

    def test_13bus_soc_expands(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('13Bus_soc')
        assert result['system_ref'] == '13Bus'
        assert result['soc_w'] == pytest.approx(20.0 / 33)

    def test_34bus_pv_expands(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('34Bus_pv')
        assert result['system_ref'] == '34Bus'
        assert result['pv_control_enabled'] is True

    def test_unknown_returns_none(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('totally_unknown')
        assert result is None

    def test_base_system_returns_none(self):
        """Base systems (13Bus, 34Bus) are not legacy variants"""
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('13Bus')
        assert result is None
```

- [ ] **Step 2: Run test — verify failure**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestLegacyVariantMapping -v --tb=short 2>&1 | tail -15`
Expected: FAIL — `expand_legacy_variant` not found

- [ ] **Step 3: Implement `expand_legacy_variant`**

Add to `utils/unified_config_loader.py` before the `ConfigLoader` class:

```python
# VVC legacy variant mapping
# Maps old env_name variants to {system_ref + parameter overrides}
# Used for backward compatibility when env_name like '13Bus_cbat' is detected
_VVC_LEGACY_VARIANTS = {
    # 13Bus variants
    '13Bus_cbat':     {'system_ref': '13Bus', 'bat_act_num': float('inf')},
    '13Bus_soc':      {'system_ref': '13Bus', 'soc_w': 20.0 / 33},
    '13Bus_cbat_soc': {'system_ref': '13Bus', 'bat_act_num': float('inf'), 'soc_w': 20.0 / 33},
    # 34Bus variants
    '34Bus_pv':       {'system_ref': '34Bus', 'pv_control_enabled': True, 'irrad_dss': 'irrad_up_down.dss'},
    '34Bus_cbat':     {'system_ref': '34Bus', 'bat_act_num': float('inf'), 'power_w': 1.0},
    '34Bus_soc':      {'system_ref': '34Bus', 'power_w': 1.0, 'soc_w': 500.0 / 33, 'dis_w': 4.0 / 33},
    '34Bus_cbat_soc': {'system_ref': '34Bus', 'bat_act_num': float('inf'), 'power_w': 1.0, 'soc_w': 500.0 / 33, 'dis_w': 4.0 / 33},
    # 123Bus variants
    '123Bus_cbat':     {'system_ref': '123Bus', 'bat_act_num': float('inf')},
    '123Bus_soc':      {'system_ref': '123Bus', 'soc_w': 500.0 / 33, 'dis_w': 5.0 / 33},
    '123Bus_cbat_soc': {'system_ref': '123Bus', 'bat_act_num': float('inf'), 'soc_w': 500.0 / 33, 'dis_w': 5.0 / 33},
    # 8500Node variants
    '8500Node_cbat':     {'system_ref': '8500Node', 'bat_act_num': float('inf')},
    '8500Node_soc':      {'system_ref': '8500Node', 'soc_w': 10000.0 / 33, 'dis_w': 100.0 / 33},
    '8500Node_cbat_soc': {'system_ref': '8500Node', 'bat_act_num': float('inf'), 'soc_w': 10000.0 / 33, 'dis_w': 100.0 / 33},
}


def expand_legacy_variant(env_name: str) -> dict | None:
    """Expand a legacy VVC variant name to config parameters.

    Args:
        env_name: Legacy environment name (e.g., '13Bus_cbat')

    Returns:
        Dict of {system_ref + parameter overrides}, or None if not a legacy variant
    """
    return _VVC_LEGACY_VARIANTS.get(env_name)
```

- [ ] **Step 4: Run test — verify pass**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestLegacyVariantMapping -v --tb=short`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add utils/unified_config_loader.py tests/test_unified_config.py
git commit -m "feat: add VVC legacy variant mapping for backward compatibility"
```

---

### Task 0.3: Supplement system YAML files

**Files:**
- Modify: `configs/systems/13Bus.yaml`
- Modify: `configs/systems/34Bus.yaml`
- Modify: `configs/systems/34Bus_PV.yaml`

The existing system YAMLs are missing some fields that `_SYS_INFO` and `_ENV_INFO` provide. Add them.

- [ ] **Step 1: Add missing fields to 13Bus.yaml**

Add under `display:` section:

```yaml
display:
  node_size: 500
  shift: 10
  show_labels: true
  show_node_labels: true  # alias for show_labels (VVC compat)
  load_noise: true
```

- [ ] **Step 2: Add missing fields to 34Bus.yaml**

```yaml
display:
  node_size: 500
  shift: 80
  show_labels: true
  show_node_labels: true
  load_noise: false
```

- [ ] **Step 3: Add pv_plans to 34Bus_PV.yaml**

Append after `display` section:

```yaml
# PV 方案清单 (通过 pv_plan 参数选择)
pv_plans:
  none:
    description: "无PV系统（纯基础测试）"
    pv_count: 0
    total_capacity_kw: 0
  conservative:
    description: "保守方案（720kW, 40.7% 渗透率）"
    pv_count: 5
    total_capacity_kw: 720
    penetration_rate: 0.407
  optimized:
    description: "优化方案（900kW, 50.8% 渗透率）"
    pv_count: 7
    total_capacity_kw: 900
    penetration_rate: 0.508
  aggressive:
    description: "高渗透率方案（1080kW, 61% 渗透率）"
    pv_count: 9
    total_capacity_kw: 1080
    penetration_rate: 0.61
```

- [ ] **Step 4: Commit**

```bash
git add configs/systems/13Bus.yaml configs/systems/34Bus.yaml configs/systems/34Bus_PV.yaml
git commit -m "feat: supplement system YAMLs with display/load_noise fields and pv_plans"
```

---

### Task 0.4: 34Bus PV directory consolidation

**Files:**
- Create: `node_systems/34Bus_PV/pv_plans/` (4 DSS files)
- Modify: `node_systems/34Bus_PV/pv_systems_base.dss` (symlink)

- [ ] **Step 1: Create pv_plans directory and copy DSS files**

```bash
cd /home/zhengxiaodong/exps/PowerZoo
mkdir -p node_systems/34Bus_PV/pv_plans
cp node_systems/34Bus_PV/pv_systems_base.dss node_systems/34Bus_PV/pv_plans/none.dss
cp node_systems/34Bus_PV_Conservative/pv_systems_base.dss node_systems/34Bus_PV/pv_plans/conservative.dss
cp node_systems/34Bus_PV_Optimized/pv_systems_base.dss node_systems/34Bus_PV/pv_plans/optimized.dss
cp node_systems/34Bus_PV_Aggressive/pv_systems_base.dss node_systems/34Bus_PV/pv_plans/aggressive.dss
```

- [ ] **Step 2: Verify files**

Run: `ls -la node_systems/34Bus_PV/pv_plans/`
Expected: 4 files, `aggressive.dss` largest

- [ ] **Step 3: Replace pv_systems_base.dss with symlink**

```bash
cd /home/zhengxiaodong/exps/PowerZoo/node_systems/34Bus_PV
mv pv_systems_base.dss pv_systems_base.dss.bak
ln -s pv_plans/aggressive.dss pv_systems_base.dss
```

- [ ] **Step 4: Verify symlink**

Run: `ls -la node_systems/34Bus_PV/pv_systems_base.dss`
Expected: symlink → pv_plans/aggressive.dss

- [ ] **Step 5: Update smartgrid.yaml**

Change `system_ref: 34Bus_PV_Aggressive` to `system_ref: 34Bus_PV` and add `pv_plan: aggressive`.
Change `environment_specific.system_name` to `34Bus_PV`.

- [ ] **Step 6: Remove variant system configs**

```bash
git rm configs/systems/34Bus_PV_Aggressive.yaml
git rm configs/systems/34Bus_PV_Conservative.yaml
git rm configs/systems/34Bus_PV_Optimized.yaml
```

- [ ] **Step 7: Archive variant directories**

```bash
mkdir -p node_systems/_archived
mv node_systems/34Bus_PV_Aggressive node_systems/_archived/
mv node_systems/34Bus_PV_Conservative node_systems/_archived/
mv node_systems/34Bus_PV_Optimized node_systems/_archived/
echo "node_systems/_archived/" >> .gitignore
```

- [ ] **Step 8: Clean up backup**

```bash
rm -f node_systems/34Bus_PV/pv_systems_base.dss.bak
```

- [ ] **Step 9: Commit**

```bash
git add node_systems/34Bus_PV/pv_plans/ node_systems/34Bus_PV/pv_systems_base.dss
git add configs/envs_cfgs/smartgrid.yaml configs/systems/ .gitignore
git rm -r --cached node_systems/34Bus_PV_Aggressive/ node_systems/34Bus_PV_Conservative/ node_systems/34Bus_PV_Optimized/ 2>/dev/null || true
git commit -m "refactor: consolidate 34Bus PV variants into pv_plans/ directory"
```

---

## Phase 1: SmartGrid Migration

### Task 1.1: Add `SmartGridConfig.from_env_args()` and `pv_plan` field

**Files:**
- Modify: `envs/smartgrid/base_env/env_config.py`
- Test: `tests/test_unified_config.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_unified_config.py`:

```python
class TestSmartGridConfigFromEnvArgs:
    """Test SmartGridConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        env_args = {
            'system_name': '34Bus_PV',
            'dss_file': 'ieee34Mod1_duty.dss',
            'max_episode_steps': 360,
            'seed': 42,
            'pv_plan': 'aggressive',
        }
        config = SmartGridConfig.from_env_args(env_args)
        assert config.system_name == '34Bus_PV'
        assert config.max_episode_steps == 360
        assert config.pv_plan == 'aggressive'

    def test_device_config_from_env_specific(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        env_args = {
            'env_specific_config': {
                'devices': {
                    'batteries': {'action_num': 33, 'action_space': 'continuous'},
                    'pv_systems': {'control_enabled': True, 'action_space': 'continuous'},
                },
                'reward_weights': {
                    'power_loss': 0.4,
                    'pv_control': 0.4,
                },
            },
        }
        config = SmartGridConfig.from_env_args(env_args)
        assert config.battery.is_continuous
        assert config.pv.control_enabled
        assert config.reward_weights.power_loss == 0.4

    def test_unknown_keys_ignored(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        env_args = {'totally_unknown_key': 'should_not_crash'}
        config = SmartGridConfig.from_env_args(env_args)
        assert not hasattr(config, 'totally_unknown_key')

    def test_defaults_used_when_empty(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        config = SmartGridConfig.from_env_args({})
        assert config.max_episode_steps == 360  # SmartGridConfig default
        assert config.pv_plan is None
```

- [ ] **Step 2: Run test — verify failure**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestSmartGridConfigFromEnvArgs -v --tb=short 2>&1 | tail -15`
Expected: FAIL — `SmartGridConfig.from_env_args` not found

- [ ] **Step 3: Add `pv_plan` field and `from_env_args` to SmartGridConfig**

Read `envs/smartgrid/base_env/env_config.py` fully. Then:

1. Add `pv_plan: Optional[str] = None` field after `worker_idx`
2. Add `from_env_args` classmethod:

```python
@classmethod
def from_env_args(cls, env_args: dict) -> 'SmartGridConfig':
    """从 unified_config_loader 的 env_args 构建配置

    替代原有的 ConfigLoader.get_config()。
    """
    config = cls()

    # 直接标量字段
    direct_fields = {
        'system_name': 'system_name',
        'dss_file': 'dss_file',
        'env_name': 'env_name',
        'max_episode_steps': 'max_episode_steps',
        'episode_length': 'max_episode_steps',  # alias
        'seed': 'seed',
        'pv_plan': 'pv_plan',
        'dss_act': 'dss_act',
        'voltage_min': 'voltage_min',
        'voltage_max': 'voltage_max',
        'source_bus': 'source_bus',
    }
    for src_key, dst_attr in direct_fields.items():
        if src_key in env_args and env_args[src_key] is not None:
            setattr(config, dst_attr, env_args[src_key])

    # 设备配置 (从 env_specific_config.devices)
    devices = env_args.get('env_specific_config', {}).get('devices', {})
    if 'regulators' in devices:
        reg_cfg = devices['regulators']
        config.regulator.action_num = reg_cfg.get('action_num', config.regulator.action_num)
    if 'batteries' in devices:
        bat_cfg = devices['batteries']
        if bat_cfg.get('action_space') == 'continuous':
            config.battery.action_num = float('inf')
        elif 'action_num' in bat_cfg:
            config.battery.action_num = bat_cfg['action_num']
    if 'pv_systems' in devices:
        pv_cfg = devices['pv_systems']
        config.pv.control_enabled = pv_cfg.get('control_enabled', config.pv.control_enabled)
        if pv_cfg.get('action_space') == 'continuous':
            config.pv.action_num = float('inf')
        elif 'action_num' in pv_cfg:
            config.pv.action_num = pv_cfg['action_num']

    # 奖励权重 (从 env_specific_config.reward_weights)
    weights = env_args.get('env_specific_config', {}).get('reward_weights', {})
    if weights:
        for attr in ['power_loss', 'capacitor', 'regulator', 'battery_soc', 'battery_discharge', 'pv_control']:
            if attr in weights:
                setattr(config.reward_weights, attr, weights[attr])

    # 约束覆盖
    constraints = env_args.get('env_specific_config', {}).get('constraints', {})
    if constraints:
        config.voltage_min = constraints.get('voltage_min', config.voltage_min)
        config.voltage_max = constraints.get('voltage_max', config.voltage_max)

    # 显示配置 (从 system config)
    for display_field in ['node_size', 'shift', 'show_node_labels']:
        if display_field in env_args and env_args[display_field] is not None:
            setattr(config, display_field, env_args[display_field])

    return config
```

- [ ] **Step 4: Run test — verify pass**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestSmartGridConfigFromEnvArgs -v --tb=short`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add envs/smartgrid/base_env/env_config.py tests/test_unified_config.py
git commit -m "feat: add SmartGridConfig.from_env_args() and pv_plan field"
```

---

### Task 1.2: Add pv_plan to Circuits class

**Files:**
- Modify: `envs/smartgrid/circuit_system/circuit.py:49-53, 225-242`
- Test: `tests/test_unified_config.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_unified_config.py`:

```python
class TestCircuitPvPlan:
    """Test pv_plan injection into Circuits"""

    def test_circuits_accepts_pv_plan(self):
        from envs.smartgrid.circuit_system.circuit import Circuits
        import inspect
        sig = inspect.signature(Circuits.__init__)
        assert 'pv_plan' in sig.parameters

    def test_temp_compile_uses_pv_plan(self, tmp_path):
        """_create_temp_compile_file should use pv_plans/{plan}.dss"""
        dss_dir = tmp_path / "test_system"
        dss_dir.mkdir()
        (dss_dir / "main.dss").write_text("! dummy\n")
        (dss_dir / "loadshape.dss").write_text("! dummy\n")
        (dss_dir / "pv_data.dss").write_text("! no PVSystem here\n")
        pv_plans = dss_dir / "pv_plans"
        pv_plans.mkdir()
        (pv_plans / "aggressive.dss").write_text("New PVSystem.PV1 phases=3\n")

        import os
        original_cwd = os.getcwd()
        os.chdir(str(dss_dir))
        try:
            from envs.smartgrid.circuit_system.circuit import Circuits
            c = object.__new__(Circuits)
            c.worker_idx = None
            c.pv_plan = "aggressive"
            temp_file = c._create_temp_compile_file("main.dss")
            with open(temp_file, 'r') as f:
                content = f.read()
            assert "pv_plans/aggressive.dss" in content
            assert "pv_systems_base.dss" not in content
            os.remove(temp_file)
        finally:
            os.chdir(original_cwd)
```

- [ ] **Step 2: Run test — verify failure**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestCircuitPvPlan -v --tb=short 2>&1 | tail -15`
Expected: FAIL

- [ ] **Step 3: Add pv_plan to Circuits.__init__**

In `envs/smartgrid/circuit_system/circuit.py:49-53`, add `pv_plan=None` parameter.
After `self.worker_idx = worker_idx` add:
```python
self.pv_plan = pv_plan
```

- [ ] **Step 4: Modify _create_temp_compile_file PV loading logic**

In `envs/smartgrid/circuit_system/circuit.py:236-238`, replace:

```python
if not pv_has_system_def and os.path.exists("pv_systems_base.dss"):
    temp_content.append("redirect pv_systems_base.dss\n")
    logger.info(f"Worker {self.worker_idx}: 加载PV基础定义 pv_systems_base.dss")
```

With:

```python
if not pv_has_system_def:
    if self.pv_plan is not None:
        pv_plan_file = f"pv_plans/{self.pv_plan}.dss"
        if os.path.exists(pv_plan_file):
            temp_content.append(f"redirect {pv_plan_file}\n")
            logger.info(f"Worker {self.worker_idx}: 加载PV方案 {pv_plan_file}")
        else:
            logger.warning(f"Worker {self.worker_idx}: PV方案 {pv_plan_file} 不存在，回退")
            if os.path.exists("pv_systems_base.dss"):
                temp_content.append("redirect pv_systems_base.dss\n")
    elif os.path.exists("pv_systems_base.dss"):
        temp_content.append("redirect pv_systems_base.dss\n")
        logger.info(f"Worker {self.worker_idx}: 加载PV基础定义 pv_systems_base.dss")
```

- [ ] **Step 5: Run test — verify pass**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestCircuitPvPlan -v --tb=short`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add envs/smartgrid/circuit_system/circuit.py tests/test_unified_config.py
git commit -m "feat: add pv_plan parameter to Circuits for config-driven PV scheme selection"
```

---

### Task 1.3: Rewire SmartGrid env.py to accept SmartGridConfig

**Files:**
- Modify: `envs/smartgrid/base_env/env.py`
- Modify: `envs/smartgrid/base_env/env_register.py`
- Modify: `envs/smartgrid/base_env/vvc_env.py`

This is the biggest single task. The goal: `Env.__init__` takes `SmartGridConfig` + `worker_idx` instead of `(info, dss_act, mode, worker_idx)`.

- [ ] **Step 1: Read current Env.__init__ to map all info dict field usage**

Read `envs/smartgrid/base_env/env.py` lines 180-270 to see every `info[...]` and `info.get(...)` call.

- [ ] **Step 2: Modify Env.__init__ signature**

Change:
```python
def __init__(self, info, dss_act=False, mode='single', worker_idx=None):
```
To:
```python
def __init__(self, config_or_info, dss_act=False, mode='single', worker_idx=None):
```

At the top of `__init__`, add adapter logic:
```python
# Support both SmartGridConfig and legacy dict
from envs.smartgrid.base_env.env_config import SmartGridConfig
if isinstance(config_or_info, SmartGridConfig):
    config = config_or_info
    info = config.to_info_dict()
    dss_act = config.dss_act
    worker_idx = config.worker_idx if config.worker_idx is not None else worker_idx
else:
    info = config_or_info
    config = None

# Store config for downstream use
self._config = config
```

Add after circuit creation: pass `pv_plan` to Circuits:
```python
pv_plan = config.pv_plan if config else info.get('pv_plan')
self.circuit = Circuits(
    os.path.join(self.dss_folder_path, self.dss_file),
    RBP_act_num=(self.reg_act_num, self.bat_act_num, self.pv_act_num),
    dss_act=dss_act,
    worker_idx=worker_idx,
    pv_plan=pv_plan
)
```

- [ ] **Step 3: Update env_register.py — remove config_loader import**

In `envs/smartgrid/base_env/env_register.py`, change the `make_base_env` function to accept optional `SmartGridConfig`:

```python
def make_base_env(env_name_or_config, dss_act=False, worker_idx=None, config_dict=None):
    """创建基础环境

    支持两种调用方式:
    1. make_base_env(config: SmartGridConfig, worker_idx=rank)  # 新方式
    2. make_base_env(env_name, dss_act, worker_idx, config_dict)  # 旧方式
    """
    from envs.smartgrid.base_env.env_config import SmartGridConfig

    if isinstance(env_name_or_config, SmartGridConfig):
        config = env_name_or_config
        # ... create Env from config ...
    else:
        # Legacy path — keep existing logic for backward compat
        env_name = env_name_or_config
        # ... existing logic ...
```

Remove the import of `config_loader`:
```python
# DELETE: from envs.smartgrid.base_env.config_loader import load_config, get_env_config
```

- [ ] **Step 4: Update vvc_env.py wrapper**

Change `VVCEnv.__init__` in `envs/smartgrid/base_env/vvc_env.py` to accept SmartGridConfig:

Read the file first, then change the constructor to store `config` instead of `env_args` dict for the fields it needs (`useS`, `use_render`, `record_node`, etc.).

- [ ] **Step 5: Update envs_tools.py SmartGrid branch**

In `utils/envs_tools.py`, change the smartgrid branch in `make_train_env`:

```python
elif env_name == "smartgrid":
    from envs.smartgrid.base_env.env_config import SmartGridConfig
    from envs.smartgrid.base_env.vvc_env import VVCEnv
    from envs.smartgrid.base_env.env_register import make_base_env

    config = SmartGridConfig.from_env_args(env_args)
    config.worker_idx = rank
    base_env = make_base_env(config, worker_idx=rank)
    env = VVCEnv(base_env, config, rank)
```

Do the same for `make_eval_env` and `make_render_env`.

- [ ] **Step 6: Run SmartGrid tests**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/envs/smartgrid/ -v --tb=short 2>&1 | tail -30`
Expected: All existing tests pass

- [ ] **Step 7: Commit**

```bash
git add envs/smartgrid/base_env/env.py envs/smartgrid/base_env/env_register.py
git add envs/smartgrid/base_env/vvc_env.py utils/envs_tools.py
git commit -m "refactor: SmartGrid accepts SmartGridConfig, remove config_loader dependency"
```

---

### Task 1.4: Delete SmartGrid config_loader.py and PRESET_CONFIGS

**Files:**
- Delete: `envs/smartgrid/base_env/config_loader.py`
- Modify: `envs/smartgrid/base_env/env_config.py` (remove PRESET_CONFIGS)

- [ ] **Step 1: Grep for all references to config_loader**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && grep -r "config_loader" envs/smartgrid/ --include="*.py" -l`
Expected: list of files. All should have been updated in Task 1.3.

- [ ] **Step 2: Grep for PRESET_CONFIGS references**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && grep -r "PRESET_CONFIGS\|get_preset" envs/ --include="*.py" -l`
Expected: only `env_config.py`. If others found, update them first.

- [ ] **Step 3: Delete config_loader.py**

```bash
git rm envs/smartgrid/base_env/config_loader.py
```

- [ ] **Step 4: Remove PRESET_CONFIGS from env_config.py**

Read `envs/smartgrid/base_env/env_config.py` to find the PRESET_CONFIGS section and `get_preset()` function. Delete both.

- [ ] **Step 5: Run SmartGrid tests again**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/envs/smartgrid/ -v --tb=short 2>&1 | tail -20`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add envs/smartgrid/base_env/env_config.py
git commit -m "refactor: delete SmartGrid config_loader.py and PRESET_CONFIGS"
```

---

## Phase 2: DSR Migration

### Task 2.1: Add `DSRConfig.from_env_args()`

**Files:**
- Modify: `envs/dsr/core/config.py`
- Test: `tests/test_unified_config.py`

- [ ] **Step 1: Write failing test**

```python
class TestDSRConfigFromEnvArgs:
    """Test DSRConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.dsr.core.config import DSRConfig
        env_args = {
            'system_name': '123Bus',
            'max_episode_steps': 15,
            'n_dg': 7,
            'reward_restore': 20.0,
        }
        config = DSRConfig.from_env_args(env_args)
        assert config.system_name == '123Bus'
        assert config.n_dg == 7

    def test_unknown_keys_ignored(self):
        from envs.dsr.core.config import DSRConfig
        config = DSRConfig.from_env_args({'unknown_key': 'ignored'})
        assert config.system_name == '123Bus'  # default

    def test_none_values_use_default(self):
        from envs.dsr.core.config import DSRConfig
        config = DSRConfig.from_env_args({'system_name': None})
        assert config.system_name == '123Bus'  # default, not None
```

- [ ] **Step 2: Run test — verify failure**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py::TestDSRConfigFromEnvArgs -v --tb=short 2>&1 | tail -15`

- [ ] **Step 3: Implement from_env_args**

Add to `envs/dsr/core/config.py` DSRConfig class:

```python
@classmethod
def from_env_args(cls, env_args: dict) -> 'DSRConfig':
    """从 unified_config_loader 的 env_args 自动构建配置

    用 dataclass 字段反射替代手工 update_keys 列表。
    """
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in env_args.items() if k in valid_fields and v is not None}
    return cls(**kwargs)
```

- [ ] **Step 4: Run test — verify pass**

- [ ] **Step 5: Commit**

```bash
git add envs/dsr/core/config.py tests/test_unified_config.py
git commit -m "feat: add DSRConfig.from_env_args() with field reflection"
```

---

### Task 2.2: Rewire DSREnv to accept DSRConfig

**Files:**
- Modify: `envs/dsr/dsr_env.py`
- Modify: `utils/envs_tools.py` (dsr branch)

- [ ] **Step 1: Modify DSREnv.__init__ to accept Config**

Read `envs/dsr/dsr_env.py`. Change `__init__`:

```python
def __init__(self, config_or_args, rank=None):
    from envs.dsr.core.config import DSRConfig

    if isinstance(config_or_args, DSRConfig):
        self.config = config_or_args
        self.args = {}  # no raw dict needed
    else:
        # Legacy path: raw dict
        self.args = copy.deepcopy(config_or_args)
        self.config = DSRConfig.from_env_args(self.args)

    self.rank = rank
    self.core_env = DSRCoreEnv(self.config, worker_idx=rank)
    # ... rest of __init__ reads from self.config instead of args ...
```

Replace all `args.get(...)` calls with `self.config.{field}` attribute access.

- [ ] **Step 2: Delete `_parse_config` method and `DEFAULT_DSR_CONFIG` usage**

The `_parse_config` method and its `update_keys` list are now replaced by `DSRConfig.from_env_args()`.

- [ ] **Step 3: Update envs_tools.py dsr branch**

```python
elif env_name == "dsr":
    from envs.dsr.core.config import DSRConfig
    from envs.dsr.dsr_env import DSREnv
    config = DSRConfig.from_env_args(env_args)
    env = DSREnv(config, rank)
```

Same for `make_eval_env` and `make_render_env`.

- [ ] **Step 4: Run DSR tests**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/ -k "dsr" -v --tb=short 2>&1 | tail -20`

- [ ] **Step 5: Commit**

```bash
git add envs/dsr/dsr_env.py envs/dsr/core/config.py utils/envs_tools.py
git commit -m "refactor: DSREnv accepts DSRConfig, delete _parse_config and update_keys"
```

---

## Phase 3: Stackelberg Migration

### Task 3.1: Create StackelbergConfig

**Files:**
- Create: `envs/stackelberg/stackelberg_config.py`
- Test: `tests/test_unified_config.py`

- [ ] **Step 1: Write failing test**

```python
class TestStackelbergConfigFromEnvArgs:
    """Test StackelbergConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        env_args = {
            'system_name': '13Bus',
            'max_episode_steps': 24,
            'n_consumer_agents': 8,
        }
        config = StackelbergConfig.from_env_args(env_args)
        assert config.system_name == '13Bus'
        assert config.n_consumer_agents == 8

    def test_extracts_system_from_env_name(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        env_args = {'env_name': 'stackelberg_34Bus'}
        config = StackelbergConfig.from_env_args(env_args)
        assert config.system_name == '34Bus'

    def test_max_episode_steps_alias(self):
        """num_steps alias should work (backward compat)"""
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        env_args = {'num_steps': 48}
        config = StackelbergConfig.from_env_args(env_args)
        assert config.max_episode_steps == 48

    def test_dict_sub_configs_passthrough(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        tou = {'peak_hours': [8, 9, 10]}
        env_args = {'tou_config': tou}
        config = StackelbergConfig.from_env_args(env_args)
        assert config.tou_config == tou
```

- [ ] **Step 2: Run test — verify failure**

- [ ] **Step 3: Create stackelberg_config.py**

```python
"""Stackelberg 博弈环境配置"""
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class StackelbergConfig:
    """Stackelberg VVC 环境配置 — 唯一数据源"""

    # 系统配置
    system_name: str = '13Bus'
    dss_file: str = 'IEEE13Nodeckt_daily.dss'
    max_episode_steps: int = 24
    seed: int = 123456

    # 智能体配置
    n_consumer_agents: int = 8

    # 子配置 (dict 类型，结构灵活)
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
    use_load_noise: bool = False
    scale: float = 1.0
    exp_name: Optional[str] = None

    # DSS 文件映射 (由 system YAML 提供)
    dss_folder: Optional[str] = None

    @classmethod
    def from_env_args(cls, env_args: dict) -> 'StackelbergConfig':
        """从 env_args 构建配置

        修复历史键名不匹配问题。
        """
        config = cls()

        # 系统名：优先 system_name，回退到 env_name 去前缀
        config.system_name = env_args.get(
            'system_name',
            env_args.get('env_name', 'stackelberg_13Bus').replace('stackelberg_', '')
        )

        # 统一键名 (修复 num_steps vs max_episode_steps)
        config.max_episode_steps = env_args.get(
            'max_episode_steps',
            env_args.get('num_steps', config.max_episode_steps)
        )

        # DSS 文件
        config.dss_file = env_args.get('dss_file', config.dss_file)
        config.dss_folder = env_args.get('dss_folder', config.dss_folder)

        # dict 子配置直传
        dict_fields = [
            'tou_config', 'tier_config', 'reward_weights',
            'load_aggregation', 'async_config', 'monitoring_config', 'n1_security',
        ]
        for f in dict_fields:
            if f in env_args:
                setattr(config, f, env_args[f])

        # 标量参数
        scalar_fields = [
            'n_consumer_agents', 'seed', 'use_render', 'use_load_noise',
            'scale', 'worker_idx', 'exp_name',
        ]
        for f in scalar_fields:
            if f in env_args and env_args[f] is not None:
                setattr(config, f, env_args[f])

        return config
```

- [ ] **Step 4: Run test — verify pass**

- [ ] **Step 5: Commit**

```bash
git add envs/stackelberg/stackelberg_config.py tests/test_unified_config.py
git commit -m "feat: create StackelbergConfig dataclass with from_env_args"
```

---

### Task 3.2: Rewire StackelbergVVCEnv to accept StackelbergConfig

**Files:**
- Modify: `envs/stackelberg/stackelberg_vvc_env.py`
- Modify: `utils/envs_tools.py` (stackelberg branch)

- [ ] **Step 1: Read stackelberg_vvc_env.py __init__ and _parse_config fully**

- [ ] **Step 2: Modify __init__ to accept Config**

```python
def __init__(self, config_or_args, rank=None):
    from envs.stackelberg.stackelberg_config import StackelbergConfig

    if isinstance(config_or_args, StackelbergConfig):
        self.config = config_or_args
        self.config.worker_idx = rank
        self.args = {}
    else:
        self.args = copy.deepcopy(config_or_args)
        self.config = StackelbergConfig.from_env_args(self.args)
        self.config.worker_idx = rank if rank is not None else self.config.worker_idx

    self.logger = logging.getLogger('StackelbergPowerZoo')
    self.env_config = self._build_env_config()
    # ... rest stays the same but reads from self.config ...
```

- [ ] **Step 3: Replace `_parse_config` with `_build_env_config` that reads from self.config**

The new method simply translates `StackelbergConfig` fields into the `env_config` dict that `StackelbergBaseEnv` expects:

```python
def _build_env_config(self) -> dict:
    """Build env_config dict from typed StackelbergConfig"""
    cfg = self.config
    env_config = {
        'system_name': cfg.system_name,
        'dss_file': cfg.dss_file,
        'dss_folder': cfg.dss_folder,
        'max_episode_steps': cfg.max_episode_steps,
        'seed': cfg.seed,
        'worker_idx': cfg.worker_idx,
        'n_consumer_agents': cfg.n_consumer_agents,
        'use_render': cfg.use_render,
        'scale': cfg.scale,
    }
    # Pass through dict sub-configs
    for key in ['tou_config', 'tier_config', 'reward_weights',
                'load_aggregation', 'async_config', 'monitoring_config', 'n1_security']:
        val = getattr(cfg, key, None)
        if val is not None:
            env_config[key] = val
    return env_config
```

- [ ] **Step 4: Update envs_tools.py stackelberg branch**

```python
elif env_name.startswith("stackelberg"):
    from envs.stackelberg.stackelberg_config import StackelbergConfig
    from envs.stackelberg.stackelberg_vvc_env import StackelbergVVCEnv
    config = StackelbergConfig.from_env_args({**env_args, 'env_name': env_name})
    env = StackelbergVVCEnv(config, rank)
```

Same for `make_eval_env` and `make_render_env`.

- [ ] **Step 5: Run Stackelberg tests**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/ -k "stackelberg" -v --tb=short 2>&1 | tail -20`

- [ ] **Step 6: Commit**

```bash
git add envs/stackelberg/stackelberg_vvc_env.py utils/envs_tools.py
git commit -m "refactor: StackelbergVVCEnv accepts StackelbergConfig, fix key name mismatch"
```

---

## Phase 4: VVC Migration

### Task 4.1: Create VVCConfig

**Files:**
- Create: `envs/vvc/vvc/vvc_config.py`
- Test: `tests/test_unified_config.py`

- [ ] **Step 1: Write failing test**

```python
class TestVVCConfigFromEnvArgs:
    """Test VVCConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.vvc.vvc.vvc_config import VVCConfig
        env_args = {
            'system_name': '13Bus',
            'dss_file': 'IEEE13Nodeckt_daily.dss',
            'max_episode_steps': 24,
            'power_w': 10.0,
            'reg_act_num': 33,
        }
        config = VVCConfig.from_env_args(env_args)
        assert config.system_name == '13Bus'
        assert config.power_w == 10.0

    def test_episode_length_alias(self):
        from envs.vvc.vvc.vvc_config import VVCConfig
        config = VVCConfig.from_env_args({'episode_length': 48})
        assert config.max_episode_steps == 48

    def test_defaults(self):
        from envs.vvc.vvc.vvc_config import VVCConfig
        config = VVCConfig.from_env_args({})
        assert config.system_name == '13Bus'
        assert config.reg_act_num == 33
        assert config.pv_act_num == 33
```

- [ ] **Step 2: Run test — verify failure**

- [ ] **Step 3: Create vvc_config.py**

```python
"""VVC 环境配置 — 替代 env_register._ENV_INFO 硬编码字典"""
import dataclasses
from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class VVCConfig:
    """VVC 环境配置 — 唯一数据源"""

    # 系统配置
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

    # 显示配置
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
    for_LLM: bool = False

    @classmethod
    def from_env_args(cls, env_args: dict) -> 'VVCConfig':
        """从 env_args 构建配置

        支持字段反射 + 键名别名映射。
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}

        # 提取已知字段
        kwargs = {k: v for k, v in env_args.items() if k in valid_fields and v is not None}

        # 键名别名
        aliases = {
            'episode_length': 'max_episode_steps',
            'num_steps': 'max_episode_steps',
            'pv_control': 'pv_control_enabled',
        }
        for old_key, new_key in aliases.items():
            if old_key in env_args and new_key not in kwargs and env_args[old_key] is not None:
                kwargs[new_key] = env_args[old_key]

        # 从 system config 提取显示配置
        display_aliases = {
            'show_labels': 'show_node_labels',
        }
        for old_key, new_key in display_aliases.items():
            if old_key in env_args and new_key not in kwargs:
                kwargs[new_key] = env_args[old_key]

        # 从 default_rewards 提取奖励权重（system YAML 格式）
        if 'env_specific_config' in env_args:
            esc = env_args['env_specific_config']
            rw = esc.get('reward_weights', {})
            reward_map = {
                'power_loss': 'power_w',
                'capacitor': 'cap_w',
                'regulator': 'reg_w',
                'battery_soc': 'soc_w',
                'battery_discharge': 'dis_w',
            }
            for yaml_key, config_key in reward_map.items():
                if yaml_key in rw and config_key not in kwargs:
                    kwargs[config_key] = rw[yaml_key]

        return cls(**kwargs)
```

- [ ] **Step 4: Run test — verify pass**

- [ ] **Step 5: Commit**

```bash
git add envs/vvc/vvc/vvc_config.py tests/test_unified_config.py
git commit -m "feat: create VVCConfig dataclass replacing _ENV_INFO hardcoded dict"
```

---

### Task 4.2: Rewire VVC env_register.py and VVCEnv

**Files:**
- Modify: `envs/vvc/vvc/env_register.py`
- Modify: `envs/vvc/vvc_env.py`
- Modify: `utils/envs_tools.py` (vvc branch)

This is the highest-risk task. The key change: `make_base_env(env_name)` → `make_base_env(config: VVCConfig)`.

- [ ] **Step 1: Audit _ENV_INFO cross-references**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && grep -rn "make_base_env\|get_info_and_folder" envs/ --include="*.py" | grep -v "__pycache__"`

Identify all callers. Only VVC's own code should call these. If DSR/Stackelberg call them, those paths are dead code (they have their own env creation).

- [ ] **Step 2: Modify make_base_env to accept VVCConfig**

Read `envs/vvc/vvc/env_register.py` fully (especially `make_base_env` and `get_info_and_folder`).

Add a new code path for `VVCConfig`:

```python
def make_base_env(config_or_name, dss_act=False, worker_idx=None):
    """创建 VVC 基础环境

    支持两种调用方式:
    1. make_base_env(config: VVCConfig, worker_idx=rank)  # 新方式
    2. make_base_env(env_name: str, dss_act, worker_idx)  # 旧方式
    """
    from envs.vvc.vvc.vvc_config import VVCConfig
    from utils.path_utils import resolve_system_path

    if isinstance(config_or_name, VVCConfig):
        config = config_or_name
        system_dir = str(resolve_system_path(config.system_name))
        info = {
            'system_name': config.system_name,
            'dss_file': config.dss_file,
            'max_episode_steps': config.max_episode_steps,
            'reg_act_num': config.reg_act_num,
            'bat_act_num': config.bat_act_num,
            'pv_act_num': config.pv_act_num if config.pv_control_enabled else float('inf'),
            'pv_control': config.pv_control_enabled,
            'power_w': config.power_w,
            'cap_w': config.cap_w,
            'reg_w': config.reg_w,
            'soc_w': config.soc_w,
            'dis_w': config.dis_w,
            'source_bus': config.source_bus,
            'node_size': config.node_size,
            'shift': config.shift,
            'show_node_labels': config.show_node_labels,
            'load_noise': config.load_noise,
            'for_LLM': config.for_LLM,
            'scale': config.scale,
        }
        dss_act = config.dss_act
    else:
        # Legacy path: keep _ENV_INFO lookup for backward compat during transition
        env_name = config_or_name
        info, system_dir = get_info_and_folder(env_name)

    # ... rest of make_base_env (worker file setup, Env creation) ...
```

- [ ] **Step 3: Modify VVCEnv.__init__ to accept VVCConfig**

```python
class VVCEnv:
    def __init__(self, config_or_args, rank=None):
        from envs.vvc.vvc.vvc_config import VVCConfig

        if isinstance(config_or_args, VVCConfig):
            self.config = config_or_args
            self.args = {}
            self.env = make_base_env(config_or_args, worker_idx=rank)
            env_name = config_or_args.system_name
        else:
            self.args = copy.deepcopy(config_or_args)
            self.config = None
            self.env = make_base_env(config_or_args['env_name'], worker_idx=rank)
            env_name = config_or_args['env_name']

        # ... rest reads from self.env (which is the base Env object) ...
        # Runtime flags from config or args
        self.env.use_render = (self.config.use_render if self.config else self.args.get('use_render', False))
        self.env.useS = (self.config.useS if self.config else self.args.get('useS', False))
        self.env.record_node = (self.config.record_node if self.config else self.args.get('record_node', False))
```

- [ ] **Step 4: Update envs_tools.py vvc branch**

```python
if env_name in ("vvc", "powerzoo"):
    from envs.vvc.vvc.vvc_config import VVCConfig
    from envs.vvc.vvc_env import VVCEnv
    config = VVCConfig.from_env_args(env_args)
    env = VVCEnv(config, rank)
```

Same for `make_eval_env` and `make_render_env`.

- [ ] **Step 5: Run VVC tests**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/envs/vvc/ -v --tb=short 2>&1 | tail -20`

- [ ] **Step 6: Commit**

```bash
git add envs/vvc/vvc/env_register.py envs/vvc/vvc_env.py utils/envs_tools.py
git commit -m "refactor: VVC accepts VVCConfig, _ENV_INFO retained as legacy fallback"
```

---

## Phase 5: Cleanup

### Task 5.1: Integration pipeline tests

**Files:**
- Create: `tests/test_config_pipeline.py`

- [ ] **Step 1: Write integration tests**

```python
"""End-to-end integration tests: YAML → Config → env.reset() → env.step()"""
import pytest
from utils.configs_tools import get_defaults_yaml_args


class TestSmartGridPipeline:
    def test_yaml_to_config_to_env(self):
        algo_args, env_args = get_defaults_yaml_args('happo', 'smartgrid')
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        config = SmartGridConfig.from_env_args(env_args)
        assert config.system_name == '34Bus_PV'
        assert config.pv_plan == 'aggressive'


class TestVVCPipeline:
    def test_yaml_to_config_to_env(self):
        algo_args, env_args = get_defaults_yaml_args('happo', 'vvc')
        from envs.vvc.vvc.vvc_config import VVCConfig
        config = VVCConfig.from_env_args(env_args)
        assert config.system_name == '13Bus'
        assert config.max_episode_steps == 24


class TestDSRPipeline:
    def test_yaml_to_config(self):
        algo_args, env_args = get_defaults_yaml_args('happo', 'dsr')
        from envs.dsr.core.config import DSRConfig
        config = DSRConfig.from_env_args(env_args)
        assert config.system_name == '123Bus'


class TestStackelbergPipeline:
    def test_yaml_to_config(self):
        algo_args, env_args = get_defaults_yaml_args('happo', 'stackelberg_13bus')
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        config = StackelbergConfig.from_env_args(env_args)
        assert config.system_name == '13Bus'
        assert config.max_episode_steps == 24  # NOT the old default from num_steps bug
```

- [ ] **Step 2: Run all integration tests**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_config_pipeline.py -v --tb=short`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_config_pipeline.py
git commit -m "test: add end-to-end config pipeline integration tests"
```

---

### Task 5.2: Delete dead JSON configs and update registry

**Files:**
- Delete: `configs/sys_cfgs/environments_info.json`
- Delete: `configs/sys_cfgs/system_info.json`
- Modify: `configs/systems/_registry.yaml`

- [ ] **Step 1: Grep for JSON file references**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && grep -r "environments_info\|system_info" --include="*.py" -l`

If only `config_loader.py` (already deleted) references them, safe to delete.

- [ ] **Step 2: Delete JSON files**

```bash
git rm configs/sys_cfgs/environments_info.json configs/sys_cfgs/system_info.json
```

- [ ] **Step 3: Update _registry.yaml**

Remove individual PV variant entries. Update 34Bus_PV metadata to reflect pv_plans.

- [ ] **Step 4: Run all tests**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_unified_config.py tests/test_config_pipeline.py -v --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add configs/sys_cfgs/ configs/systems/_registry.yaml
git commit -m "chore: delete dead JSON configs, update system registry"
```

---

### Task 5.3: Final cleanup and CLAUDE.md update

- [ ] **Step 1: Grep for remaining dead imports**

```bash
cd /home/zhengxiaodong/exps/PowerZoo
grep -rn "from.*config_loader import\|PRESET_CONFIGS\|DEFAULT_DSR_CONFIG\|_ENV_INFO\|_SYS_INFO" envs/ --include="*.py" | grep -v "__pycache__" | grep -v "_old\|_archived"
```

Fix any remaining references.

- [ ] **Step 2: Run full test suite**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/ -v --tb=short 2>&1 | tail -40`

- [ ] **Step 3: Update CLAUDE.md config-related sections**

Update the "配置体系" section to reflect unified pipeline.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup for unified config architecture"
```
