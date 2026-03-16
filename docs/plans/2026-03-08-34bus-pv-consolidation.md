# 34Bus PV 变体合并与配置驱动化 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 4 个 34Bus PV 变体目录（44MB、90%+ 重复文件）合并为单一目录 + `pv_plans/` 配置驱动方案，实现通过 YAML 参数 `pv_plan` 切换 PV 方案。

**Architecture:** 核心思路是"Python 动态注入"——`circuit.py` 根据 `pv_plan` 参数选择对应的 DSS 文件（`pv_plans/{plan}.dss`），替代当前硬编码的 `pv_systems_base.dss` 查找。配置链路：`smartgrid.yaml` → `unified_config_loader` → `env.py` → `Circuits.__init__()` → `_create_temp_compile_file()`。

**Tech Stack:** Python 3.10+, OpenDSS (dss-python), YAML config, pytest

---

## 变更影响分析

### 当前 4 个变体的唯一差异

| 变体 | pv_systems_base.dss 内容 | PV 数量 | 总容量 |
|------|--------------------------|---------|--------|
| 34Bus_PV | 空模板（无 PVSystem） | 0 | 0kW |
| 34Bus_PV_Conservative | 5 组变压器+PV | 5 | 720kW |
| 34Bus_PV_Optimized | 7 组变压器+PV | 7 | 900kW |
| 34Bus_PV_Aggressive | 9 组变压器+PV | 9 | 1080kW |

其余文件（`ieee34Mod1_duty.dss`, `IEEELineCodes.dss`, loadshape/, irradiation/, temperature/）完全相同或仅 `sinterval` 不同。

### 代码触点

| 文件 | 行号 | 变更内容 |
|------|------|----------|
| `envs/smartgrid/circuit_system/circuit.py` | L49-53, L182-248 | `__init__` 新增 `pv_plan` 参数；`_create_temp_compile_file()` 改用 `pv_plans/` |
| `envs/smartgrid/base_env/env.py` | L248-253 | 传递 `pv_plan` 给 `Circuits` |
| `utils/unified_config_loader.py` | L209-215 | 提取 `pv_plan` 到 `env_args` |
| `configs/envs_cfgs/smartgrid.yaml` | L12 | 新增 `pv_plan` 字段 |
| `configs/systems/34Bus_PV.yaml` | L4-8 | 统一为 `dss_folder: node_systems/34Bus_PV` |
| `configs/systems/_registry.yaml` | 全文 | 删除独立变体条目，改为 `pv_plans` 子列表 |
| `node_systems/34Bus_PV/pv_plans/` | 新目录 | 4 个 PV 方案 DSS 文件 |

---

### Task 1: 创建 pv_plans 目录与 DSS 文件

**Files:**
- Create: `node_systems/34Bus_PV/pv_plans/none.dss`
- Create: `node_systems/34Bus_PV/pv_plans/conservative.dss`
- Create: `node_systems/34Bus_PV/pv_plans/optimized.dss`
- Create: `node_systems/34Bus_PV/pv_plans/aggressive.dss`

**Step 1: 从各变体提取 pv_systems_base.dss 内容**

读取以下文件内容：
- `node_systems/34Bus_PV/pv_systems_base.dss` → 复制为 `pv_plans/none.dss`
- `node_systems/34Bus_PV_Conservative/pv_systems_base.dss` → 复制为 `pv_plans/conservative.dss`
- `node_systems/34Bus_PV_Optimized/pv_systems_base.dss` → 复制为 `pv_plans/optimized.dss`
- `node_systems/34Bus_PV_Aggressive/pv_systems_base.dss` → 复制为 `pv_plans/aggressive.dss`

```bash
mkdir -p node_systems/34Bus_PV/pv_plans
cp node_systems/34Bus_PV/pv_systems_base.dss node_systems/34Bus_PV/pv_plans/none.dss
cp node_systems/34Bus_PV_Conservative/pv_systems_base.dss node_systems/34Bus_PV/pv_plans/conservative.dss
cp node_systems/34Bus_PV_Optimized/pv_systems_base.dss node_systems/34Bus_PV/pv_plans/optimized.dss
cp node_systems/34Bus_PV_Aggressive/pv_systems_base.dss node_systems/34Bus_PV/pv_plans/aggressive.dss
```

**Step 2: 验证文件创建成功**

Run: `ls -la node_systems/34Bus_PV/pv_plans/`
Expected: 4 个 DSS 文件，`aggressive.dss` 最大（9 组 PV），`none.dss` 最小

**Step 3: 更新 pv_systems_base.dss 为软链接（向后兼容）**

保留 `pv_systems_base.dss` 作为默认方案（指向 `pv_plans/aggressive.dss`，因为当前 `smartgrid.yaml` 使用 Aggressive），确保未升级的代码路径不会崩溃。

```bash
cd node_systems/34Bus_PV
# 备份原 pv_systems_base.dss
mv pv_systems_base.dss pv_systems_base.dss.bak
# 创建软链接
ln -s pv_plans/aggressive.dss pv_systems_base.dss
```

**Step 4: Commit**

```bash
git add node_systems/34Bus_PV/pv_plans/
git add node_systems/34Bus_PV/pv_systems_base.dss
git commit -m "feat: create pv_plans/ directory with 4 PV scheme DSS files"
```

---

### Task 2: 修改 Circuits 类接受 pv_plan 参数

**Files:**
- Modify: `envs/smartgrid/circuit_system/circuit.py:49-53` (`__init__`)
- Modify: `envs/smartgrid/circuit_system/circuit.py:182-248` (`_create_temp_compile_file`)
- Test: `tests/test_circuit_pv_plan.py`

**Step 1: 写 failing test**

```python
# tests/test_circuit_pv_plan.py
"""测试 pv_plan 参数注入到 Circuits 类"""
import os
import pytest


class TestCircuitPvPlan:
    """测试 Circuits 类的 pv_plan 参数传递"""

    def test_circuits_accepts_pv_plan_param(self):
        """Circuits.__init__ 应该接受 pv_plan 参数并存储为属性"""
        from envs.smartgrid.circuit_system.circuit import Circuits
        import inspect
        sig = inspect.signature(Circuits.__init__)
        assert 'pv_plan' in sig.parameters, "Circuits.__init__ 缺少 pv_plan 参数"

    def test_default_pv_plan_is_none(self):
        """pv_plan 参数默认值应为 None（向后兼容）"""
        from envs.smartgrid.circuit_system.circuit import Circuits
        import inspect
        sig = inspect.signature(Circuits.__init__)
        default = sig.parameters['pv_plan'].default
        assert default is None, f"pv_plan 默认值应为 None，实际为 {default}"

    def test_temp_compile_uses_pv_plan(self, tmp_path):
        """_create_temp_compile_file 应根据 pv_plan 注入对应的 pv_plans/{plan}.dss"""
        # 创建模拟的 DSS 目录结构
        dss_dir = tmp_path / "34Bus_PV"
        dss_dir.mkdir()

        # 创建主 DSS 文件
        main_dss = dss_dir / "ieee34Mod1_duty.dss"
        main_dss.write_text("! dummy main circuit\n")

        # 创建 pv_plans 目录
        pv_plans_dir = dss_dir / "pv_plans"
        pv_plans_dir.mkdir()
        (pv_plans_dir / "aggressive.dss").write_text(
            "New PVSystem.PV834 phases=3 bus1=trafo_pv834 kV=0.48\n"
        )

        # 创建 loadshape 和 pv_data 文件
        (dss_dir / "loadshape.dss").write_text("! dummy loadshape\n")
        (dss_dir / "pv_data.dss").write_text("! dummy pv curves only, no PVSystem\n")

        # 切换到 DSS 目录（OpenDSS 以 DSS 文件所在目录为工作目录）
        original_cwd = os.getcwd()
        os.chdir(str(dss_dir))
        try:
            # 直接调用 _create_temp_compile_file（不实际编译 DSS）
            from envs.smartgrid.circuit_system.circuit import Circuits
            # 创建一个不初始化 DSS 的 mock 实例
            c = object.__new__(Circuits)
            c.worker_idx = None
            c.pv_plan = "aggressive"

            temp_file = c._create_temp_compile_file("ieee34Mod1_duty.dss")
            with open(temp_file, 'r') as f:
                content = f.read()

            assert "pv_plans/aggressive.dss" in content, \
                f"临时编译文件应包含 pv_plans/aggressive.dss，实际内容:\n{content}"
            assert "pv_systems_base.dss" not in content, \
                "使用 pv_plan 参数时不应引用 pv_systems_base.dss"

            # 清理
            if os.path.exists(temp_file):
                os.remove(temp_file)
        finally:
            os.chdir(original_cwd)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_circuit_pv_plan.py -v --tb=short 2>&1 | head -40`
Expected: FAIL — `Circuits.__init__` 缺少 `pv_plan` 参数

**Step 3: 修改 `Circuits.__init__` 新增 pv_plan 参数**

在 `envs/smartgrid/circuit_system/circuit.py:49-53`：

```python
# 修改前：
def __init__(self, dss_file,
            batt_file='Battery.csv',
            RBP_act_num=(33, 33, float('inf')),
            dss_act=False,
            worker_idx=None):

# 修改后：
def __init__(self, dss_file,
            batt_file='Battery.csv',
            RBP_act_num=(33, 33, float('inf')),
            dss_act=False,
            worker_idx=None,
            pv_plan=None):
```

在 `self.worker_idx = worker_idx` 之后添加：
```python
self.pv_plan = pv_plan  # PV 方案名称（如 'aggressive'），None 时使用默认行为
```

**Step 4: 修改 `_create_temp_compile_file` 使用 pv_plan**

在 `envs/smartgrid/circuit_system/circuit.py:225-242`，替换 PV 加载逻辑：

```python
# 修改前 (L225-242)：
# 添加PV数据文件（如果存在）
# 检查pv_data文件是否包含完整的PVSystem定义
# 如果只有曲线定义（无PVSystem），需要先加载pv_systems_base.dss提供PV系统和变压器
if os.path.exists(pv_data_file):
    pv_has_system_def = False
    with open(pv_data_file, 'r') as f:
        for line in f:
            if 'pvsystem' in line.lower() and line.strip().lower().startswith('new'):
                pv_has_system_def = True
                break

    if not pv_has_system_def and os.path.exists("pv_systems_base.dss"):
        # 先加载PV系统和变压器的基础定义，再加载worker的曲线文件
        temp_content.append("redirect pv_systems_base.dss\n")
        logger.info(f"Worker {self.worker_idx}: 加载PV基础定义 pv_systems_base.dss")

    temp_content.append(f"redirect {pv_data_file}\n")
    logger.info(f"Worker {self.worker_idx}: 加载PV文件 {pv_data_file}")

# 修改后：
# 添加PV数据文件（如果存在）
# 检查pv_data文件是否包含完整的PVSystem定义
# 如果只有曲线定义（无PVSystem），需要先加载PV系统基础定义
if os.path.exists(pv_data_file):
    pv_has_system_def = False
    with open(pv_data_file, 'r') as f:
        for line in f:
            if 'pvsystem' in line.lower() and line.strip().lower().startswith('new'):
                pv_has_system_def = True
                break

    if not pv_has_system_def:
        # 优先使用 pv_plan 参数指定的方案文件
        if self.pv_plan is not None:
            pv_plan_file = f"pv_plans/{self.pv_plan}.dss"
            if os.path.exists(pv_plan_file):
                temp_content.append(f"redirect {pv_plan_file}\n")
                logger.info(f"Worker {self.worker_idx}: 加载PV方案 {pv_plan_file}")
            else:
                logger.warning(
                    f"Worker {self.worker_idx}: PV方案文件 {pv_plan_file} 不存在，"
                    f"回退到 pv_systems_base.dss"
                )
                if os.path.exists("pv_systems_base.dss"):
                    temp_content.append("redirect pv_systems_base.dss\n")
        elif os.path.exists("pv_systems_base.dss"):
            # 向后兼容：无 pv_plan 参数时使用 pv_systems_base.dss
            temp_content.append("redirect pv_systems_base.dss\n")
            logger.info(f"Worker {self.worker_idx}: 加载PV基础定义 pv_systems_base.dss")

    temp_content.append(f"redirect {pv_data_file}\n")
    logger.info(f"Worker {self.worker_idx}: 加载PV文件 {pv_data_file}")
```

**Step 5: Run test to verify it passes**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_circuit_pv_plan.py -v --tb=short 2>&1 | head -40`
Expected: 3 tests PASS

**Step 6: Commit**

```bash
git add envs/smartgrid/circuit_system/circuit.py tests/test_circuit_pv_plan.py
git commit -m "feat: add pv_plan parameter to Circuits for config-driven PV scheme selection"
```

---

### Task 3: 修改 env.py 传递 pv_plan 到 Circuits

**Files:**
- Modify: `envs/smartgrid/base_env/env.py:248-253`
- Test: `tests/test_circuit_pv_plan.py` (追加)

**Step 1: 写 failing test**

在 `tests/test_circuit_pv_plan.py` 追加：

```python
class TestEnvPvPlanPassthrough:
    """测试 env.py 将 pv_plan 传递给 Circuits"""

    def test_env_info_accepts_pv_plan(self):
        """SmartGridEnv 的 info dict 应该支持 pv_plan 键"""
        # 验证 env.py 代码中有 pv_plan 的提取逻辑
        import ast
        with open("envs/smartgrid/base_env/env.py", "r") as f:
            source = f.read()
        assert "pv_plan" in source, "env.py 应包含 pv_plan 参数处理逻辑"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_circuit_pv_plan.py::TestEnvPvPlanPassthrough -v --tb=short`
Expected: FAIL — `env.py` 中没有 `pv_plan`

**Step 3: 修改 env.py**

在 `envs/smartgrid/base_env/env.py` 中，找到 Circuits 实例化处（约 L248）：

```python
# 修改前：
self.circuit = Circuits(
    os.path.join(self.dss_folder_path, self.dss_file),
    RBP_act_num=(self.reg_act_num, self.bat_act_num, self.pv_act_num),
    dss_act=dss_act,
    worker_idx=worker_idx
)

# 修改后：
self.pv_plan = info.get('pv_plan', None)

self.circuit = Circuits(
    os.path.join(self.dss_folder_path, self.dss_file),
    RBP_act_num=(self.reg_act_num, self.bat_act_num, self.pv_act_num),
    dss_act=dss_act,
    worker_idx=worker_idx,
    pv_plan=self.pv_plan
)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_circuit_pv_plan.py -v --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add envs/smartgrid/base_env/env.py tests/test_circuit_pv_plan.py
git commit -m "feat: pass pv_plan from env config through to Circuits constructor"
```

---

### Task 4: 修改 unified_config_loader 提取 pv_plan

**Files:**
- Modify: `utils/unified_config_loader.py:209-215`
- Test: `tests/test_circuit_pv_plan.py` (追加)

**Step 1: 写 failing test**

```python
class TestConfigLoaderPvPlan:
    """测试 unified_config_loader 提取 pv_plan"""

    def test_pv_plan_extracted_from_env_config(self):
        """env_config 中的 pv_plan 应被提取到 env_args 中"""
        from utils.unified_config_loader import ConfigLoader
        loader = ConfigLoader()

        # 模拟 env_config 包含 pv_plan
        env_config = {
            'system_ref': '34Bus_PV',
            'pv_plan': 'aggressive',
            'env_name': 'test',
        }
        system_config = {
            'system': {
                'name': '34Bus_PV',
                'dss_folder': 'node_systems/34Bus_PV',
                'dss_file': 'ieee34Mod1_duty.dss',
            }
        }

        env_args = loader._build_env_args(env_config, system_config, None)
        assert env_args.get('pv_plan') == 'aggressive', \
            f"env_args 应包含 pv_plan='aggressive'，实际: {env_args.get('pv_plan')}"
```

**Step 2: Run test to verify it passes (or already passes)**

`pv_plan` 不在 `skip_keys` 中，所以 `_build_env_args` 的通用逻辑（L265-292）会自动将其传递到 `env_args`。这个测试**应该直接通过**，因为 `pv_plan` 是一个普通键。

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_circuit_pv_plan.py::TestConfigLoaderPvPlan -v --tb=short`
Expected: PASS（无需修改 unified_config_loader.py 代码）

> **NOTE**: `unified_config_loader._build_env_args()` 的通用处理逻辑（L265: `for key, value in env_config.items()`）会自动将 `pv_plan` 传入 `env_args`。无需特殊处理。如果测试失败（不应该），则需要在 `_build_env_args` 的 system_config 处理段新增 `pv_plan` 提取。

**Step 3: Commit**

```bash
git add tests/test_circuit_pv_plan.py
git commit -m "test: verify pv_plan passes through unified_config_loader"
```

---

### Task 5: 更新 YAML 配置文件

**Files:**
- Modify: `configs/envs_cfgs/smartgrid.yaml:12`
- Modify: `configs/systems/34Bus_PV.yaml`
- Modify: `configs/systems/_registry.yaml`
- Remove: `configs/systems/34Bus_PV_Aggressive.yaml`
- Remove: `configs/systems/34Bus_PV_Conservative.yaml`
- Remove: `configs/systems/34Bus_PV_Optimized.yaml`

**Step 1: 修改 smartgrid.yaml 添加 pv_plan 字段**

```yaml
# 修改前：
system_ref: 34Bus_PV_Aggressive

# 修改后：
system_ref: 34Bus_PV
pv_plan: aggressive       # PV方案: none | conservative | optimized | aggressive
```

同时更新 `environment_specific` 中的 `system_name`:
```yaml
# 修改前：
environment_specific:
  system_name: "34Bus_PV_Aggressive"

# 修改后：
environment_specific:
  system_name: "34Bus_PV"
```

**Step 2: 更新 34Bus_PV.yaml 系统配置**

在 `configs/systems/34Bus_PV.yaml` 中添加 `pv_plans` 元数据段，并更新设备计数为动态说明：

```yaml
# 在文件末尾 display 段之后新增：

# PV 方案清单
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

**Step 3: 更新 _registry.yaml**

```yaml
# 删除 pv_systems 中的独立变体：
# 修改前：
pv_systems:
  - 34Bus_PV
  - 34Bus_PV_Aggressive
  - 34Bus_PV_Conservative
  - 34Bus_PV_Optimized

# 修改后：
pv_systems:
  - name: 34Bus_PV
    pv_plans: [none, conservative, optimized, aggressive]

# 删除 metadata 中的 34Bus_PV_Aggressive、34Bus_PV_Conservative、34Bus_PV_Optimized 条目
# 更新 34Bus_PV metadata：
metadata:
  34Bus_PV:
    description: "IEEE 34-bus with configurable PV plans (pv_plan parameter)"
    node_count: 34
    pv_plans: [none, conservative, optimized, aggressive]
    typical_episode_length: 360
```

**Step 4: 删除独立变体的系统配置文件**

```bash
git rm configs/systems/34Bus_PV_Aggressive.yaml
git rm configs/systems/34Bus_PV_Conservative.yaml
git rm configs/systems/34Bus_PV_Optimized.yaml
```

**Step 5: 创建兼容映射（可选但推荐）**

为了向后兼容，在 `configs/systems/` 中保留轻量级转发文件，避免引用旧名称的代码报错：

> **决策**: 暂不创建转发文件。如果有外部代码引用旧名称，在测试阶段发现后再处理。

**Step 6: Commit**

```bash
git add configs/envs_cfgs/smartgrid.yaml configs/systems/34Bus_PV.yaml configs/systems/_registry.yaml
git commit -m "feat: config-driven PV plan selection, remove redundant variant configs"
```

---

### Task 6: 合并数据目录

**Files:**
- 修改: `node_systems/34Bus_PV/` 目录结构
- 删除: `node_systems/34Bus_PV_Aggressive/`, `34Bus_PV_Conservative/`, `34Bus_PV_Optimized/`

**Step 1: 比较 sinterval 差异**

关键发现：`34Bus_PV` 使用 `sinterval=120`（2分钟步长），而 `34Bus_PV_Aggressive` 使用 `sinterval=60`（1分钟步长）。loadshape 和 irradiation/temperature 数据的时间分辨率不同。

**需要确认的决策点**：合并后使用哪个 `sinterval`？
- 方案 A: 统一使用 `sinterval=60`（Aggressive 的数据，更精细）
- 方案 B: 保留两套 loadshape 数据，通过配置选择
- **推荐方案 A**：更精细的数据可以通过 OpenDSS 的内插机制兼容 `sinterval=120` 的场景

```bash
# 验证 sinterval 差异
grep -r "sinterval" node_systems/34Bus_PV/ieee34Mod1_duty.dss
grep -r "sinterval" node_systems/34Bus_PV_Aggressive/ieee34Mod1_duty.dss
```

**Step 2: 对齐 loadshape 数据**

如果 Aggressive 的 loadshape 数据更完整（更多 worker 目录），以其为基准：

```bash
# 检查 worker 目录数量
ls -d node_systems/34Bus_PV/loadshape/0*/ | wc -l
ls -d node_systems/34Bus_PV_Aggressive/loadshape/0*/ | wc -l
```

如果 Aggressive 有更多 worker 目录，复制缺失的到 34Bus_PV：

```bash
# 从 Aggressive 补充 loadshape worker 目录到 34Bus_PV
# （具体命令在执行时根据实际情况调整）
```

**Step 3: 对齐 irradiation/temperature 数据**

同理复制缺失的 worker 目录。

**Step 4: 更新 ieee34Mod1_duty.dss 中的 sinterval**

如果决定统一 `sinterval=60`，修改 `34Bus_PV/ieee34Mod1_duty.dss`。

**Step 5: 验证对齐结果**

```bash
diff <(ls node_systems/34Bus_PV/loadshape/) <(ls node_systems/34Bus_PV_Aggressive/loadshape/)
diff <(ls node_systems/34Bus_PV/irradiation/) <(ls node_systems/34Bus_PV_Aggressive/irradiation/)
diff <(ls node_systems/34Bus_PV/temperature/) <(ls node_systems/34Bus_PV_Aggressive/temperature/)
```

**Step 6: 归档旧变体目录**

```bash
# 移动到归档目录
mkdir -p node_systems/_archived
mv node_systems/34Bus_PV_Aggressive node_systems/_archived/
mv node_systems/34Bus_PV_Conservative node_systems/_archived/
mv node_systems/34Bus_PV_Optimized node_systems/_archived/

# 在 .gitignore 中排除归档目录（如果不需要追踪）
echo "node_systems/_archived/" >> .gitignore
```

**Step 7: Commit**

```bash
git add node_systems/34Bus_PV/ node_systems/_archived/ .gitignore
git rm -r --cached node_systems/34Bus_PV_Aggressive/ node_systems/34Bus_PV_Conservative/ node_systems/34Bus_PV_Optimized/
git commit -m "refactor: consolidate 34Bus PV data directories, archive old variants"
```

---

### Task 7: SmartGrid PV 专用配置文件更新

**Files:**
- Modify: `configs/envs_cfgs/smartgrid_pv_plans/` (如存在)
- Modify: `configs/envs_cfgs/smartgrid.yaml` 注释

**Step 1: 检查是否存在 smartgrid_pv_plans 目录**

```bash
ls configs/envs_cfgs/smartgrid_pv_plans/ 2>/dev/null
```

如果存在（之前的分析发现有 aggressive/conservative/optimized 子配置），需要更新这些文件的 `system_ref` 和添加 `pv_plan`。

**Step 2: 更新每个子配置（如存在）**

```yaml
# 每个子配置文件：
# 修改 system_ref: 34Bus_PV_{variant} → system_ref: 34Bus_PV
# 新增 pv_plan: {variant}
```

**Step 3: Commit**

```bash
git add configs/envs_cfgs/smartgrid_pv_plans/
git commit -m "feat: update smartgrid PV variant configs to use pv_plan parameter"
```

---

### Task 8: 端到端集成测试

**Files:**
- Test: `tests/test_circuit_pv_plan.py` (追加集成测试)

**Step 1: 写集成测试**

```python
class TestPvPlanIntegration:
    """端到端集成测试：从 YAML 配置到 DSS 编译"""

    def test_config_to_circuit_pipeline(self):
        """完整链路：YAML pv_plan → ConfigLoader → env_args → Circuits"""
        from utils.unified_config_loader import ConfigLoader
        loader = ConfigLoader()
        config = loader.load('happo', 'smartgrid')

        env_args = config.env_args
        # 验证 pv_plan 在 env_args 中
        assert 'pv_plan' in env_args, "env_args 应包含 pv_plan"
        assert env_args['pv_plan'] == 'aggressive', \
            f"smartgrid.yaml 中配置的 pv_plan 应为 'aggressive'，实际: {env_args.get('pv_plan')}"

    def test_pv_plan_file_exists_for_all_plans(self):
        """所有注册的 PV 方案对应的 DSS 文件都必须存在"""
        import os
        pv_plans_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "node_systems", "34Bus_PV", "pv_plans"
        )
        expected_plans = ['none', 'conservative', 'optimized', 'aggressive']
        for plan in expected_plans:
            plan_file = os.path.join(pv_plans_dir, f"{plan}.dss")
            assert os.path.exists(plan_file), f"PV 方案文件缺失: {plan_file}"

    def test_system_ref_points_to_consolidated_dir(self):
        """system_ref=34Bus_PV 应指向合并后的目录"""
        from utils.unified_config_loader import ConfigLoader
        loader = ConfigLoader()
        system_config = loader.load_system('34Bus_PV')
        dss_folder = system_config['system']['dss_folder']
        assert dss_folder == 'node_systems/34Bus_PV', \
            f"dss_folder 应为 node_systems/34Bus_PV，实际: {dss_folder}"
```

**Step 2: Run all tests**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/test_circuit_pv_plan.py -v --tb=short`
Expected: ALL PASS

**Step 3: Run existing SmartGrid tests (regression)**

Run: `cd /home/zhengxiaodong/exps/PowerZoo && python -m pytest tests/envs/smartgrid/ -v --tb=short 2>&1 | tail -20`
Expected: 无回归

**Step 4: Commit**

```bash
git add tests/test_circuit_pv_plan.py
git commit -m "test: add integration tests for pv_plan config-driven pipeline"
```

---

### Task 9: 清理与文档

**Files:**
- Remove: `tests/test_circuit_pv_plan.py` 中的临时/调试代码
- Remove: `node_systems/34Bus_PV/pv_systems_base.dss.bak`
- Update: CLAUDE.md 中相关引用（如有）

**Step 1: 清理备份文件**

```bash
rm -f node_systems/34Bus_PV/pv_systems_base.dss.bak
```

**Step 2: 验证最终目录结构**

```bash
# 期望结构：
# node_systems/34Bus_PV/
# ├── ieee34Mod1_duty.dss
# ├── IEEELineCodes.dss
# ├── pv_systems_base.dss → pv_plans/aggressive.dss (symlink)
# ├── pv_plans/
# │   ├── none.dss
# │   ├── conservative.dss
# │   ├── optimized.dss
# │   └── aggressive.dss
# ├── loadshape/
# ├── irradiation/
# ├── temperature/
# └── pv_data.dss

tree -L 2 node_systems/34Bus_PV/ | head -20
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: cleanup backup files after 34Bus PV consolidation"
```

---

## 风险与回滚

| 风险 | 缓解措施 |
|------|----------|
| 旧代码引用 `34Bus_PV_Aggressive` 等名称 | `pv_systems_base.dss` 软链接保持向后兼容 |
| sinterval 统一导致仿真结果变化 | 先比较两种 sinterval 的仿真结果差异 |
| 其他环境（VVC/Stackelberg/DSR）的 circuit.py | 仅影响 SmartGrid 的 `circuit.py`，其他环境独立 |
| worker 数据目录数量不一致 | Task 6 中明确对齐步骤 |
| `_archived/` 目录忘记加入 `.gitignore` | Task 6 Step 6 中明确处理 |

## 预期收益

- **存储**: 44MB → ~15MB（~70% 减少）
- **配置**: 4 个独立 system YAML → 1 个 + `pv_plan` 参数
- **可扩展性**: 新增 PV 方案只需在 `pv_plans/` 加一个 DSS 文件
- **代码路径**: 消除 4 条重复代码路径，统一为 1 条
