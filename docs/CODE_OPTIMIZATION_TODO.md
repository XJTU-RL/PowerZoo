# PowerZoo 代码优化 TODO 清单

**生成日期**: 2025-12-24
**审查范围**: 全项目代码审查（454个Python文件）
**审查专家**: RL算法专家、代码质量专家、环境兼容性专家、架构专家

---

## 📊 审查摘要

| 类别 | 问题数量 | 预计工作量 |
|------|----------|------------|
| P0 (阻塞性/高危) | 8 | 3-5天 |
| P1 (高优先级) | 12 | 1-2周 |
| P2 (中优先级) | 15 | 2-3周 |
| P3 (低优先级/技术债务) | 20+ | 持续改进 |

---

## 🚨 P0 - 阻塞性问题（必须立即修复）

### 1. [算法] MAPPO缺少NaN检查 - 可能导致训练崩溃
- **文件**: `algorithms/actors/mappo.py:133-138`
- **问题**: advantages归一化缺少NaN/Inf检查和回退机制
- **影响**: 当所有active_masks为0时，会导致advantages全部变为NaN，训练完全失败
- **修复方案**:
```python
if state_type == "EP":
    advantages_copy = advantages.copy()
    advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
    mean_advantages = np.nanmean(advantages_copy)
    std_advantages = np.nanstd(advantages_copy)

    epsilon = 1e-5
    if np.isnan(mean_advantages) or np.isnan(std_advantages):
        mean_advantages = np.mean(advantages)
        std_advantages = np.std(advantages)

    std_advantages = max(std_advantages, epsilon)
    advantages = (advantages - mean_advantages) / std_advantages
```
- [ ] 修复完成
- [ ] 添加单元测试验证

---

### 2. [环境] Done信号数据类型不符合HAPPO规范
- **文件**:
  - `envs/stackelberg/stackelberg_powerzoo_env.py:244`
  - `envs/dsr/dsr_env.py:242`
- **问题**: Done信号返回`List[bool]`而非`np.ndarray(dtype=bool)`
- **影响**: HAPPO训练可能崩溃或产生错误行为
- **修复方案**:
```python
# 统一Done信号格式
dones = np.array([bool(done)] * self.n_agents, dtype=bool)
assert isinstance(dones, np.ndarray)
assert dones.dtype == bool
```
- [ ] 修复stackelberg环境
- [ ] 修复dsr环境
- [ ] 运行HAPPO兼容性测试

---

### 3. [环境] step()返回格式不统一
- **文件**: `envs/powerzoo/powerzoo_env.py` (Legacy环境)
- **问题**: 使用单智能体OpenAI Gym格式`(obs, reward, done, info)`，而非MARL格式
- **影响**: 无法与HAPPO算法配合使用，HAPPO兼容性仅23%
- **修复方案**: 创建MARL适配包装器`PowerZooMARLWrapper`
- [ ] 创建包装器类
- [ ] 更新环境注册表
- [ ] 文档标记Legacy环境废弃

---

### 4. [环境] reset()返回值不一致
- **文件**: 各环境实现
- **问题**: 命名不一致（obs/local_obs/observations, state/share_obs/states）
- **影响**: 接口混乱，增加算法对接难度
- **修复方案**: 统一为`(local_obs, share_obs, avail_actions)`三元组
- [ ] 修复powerzoo环境
- [ ] 验证其他环境一致性

---

### 5. [架构] 清理历史遗留代码 - 释放360MB空间
- **目录/文件**:
  - `.yoyo/` (342MB) - 历史快照目录
  - `.trae/` (16KB) - 用途不明配置
  - `envs/powerzoo/powerzoo/._env.py` - Mac系统临时文件
  - 所有`__pycache__/`目录
- **执行命令**:
```bash
rm -rf .yoyo/ .trae/
find . -name "._*" -type f -delete
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```
- [ ] 执行清理
- [ ] 更新.gitignore

---

### 6. [代码质量] 删除死代码文件
- **文件**: `utils/sign.py` (382行)
- **问题**: 完全未被任何文件导入或使用
- **验证**: `grep -r "from utils.sign import" --include="*.py" .` 返回空
- [ ] 确认可删除后执行`rm utils/sign.py`

---

### 7. [代码质量] 删除隐藏备份文件
- **文件**: `utils/._envs_tools.py`
- **问题**: macOS系统生成的垃圾文件
- [ ] 执行`rm utils/._envs_tools.py`

---

### 8. [算法] HATRPO线搜索失败的静默回退
- **文件**: `algorithms/actors/hatrpo.py:209-212`
- **问题**: 线搜索失败时仅print警告，不记录到训练日志，导致"静默失败"
- **修复方案**:
```python
if not flag:
    params = flat_params(old_actor)
    update_model(self.actor, params)
    warning_msg = "TRPO line search failed - policy reverted to old parameters"
    print(f"[WARNING] {warning_msg}")
    # 返回loss_improve=0表示失败，便于train()方法记录
    return kl, 0.0, expected_improve, dist_entropy, ratio
```
- [ ] 修复并添加日志记录

---

## ⚠️ P1 - 高优先级（本周内修复）

### 9. [环境] 奖励数组格式不符合HAPPO规范
- **文件**:
  - `envs/stackelberg/stackelberg_powerzoo_env.py:244`
  - `envs/dsr/dsr_env.py:242`
- **问题**: 返回`List[List[float]]`而非`np.ndarray(n_agents, 1)`
- **修复方案**:
```python
rewards_formatted = np.array([[float(r)] for r in reward_values], dtype=np.float32)
assert rewards_formatted.shape == (self.n_agents, 1)
```
- [ ] 修复stackelberg环境
- [ ] 修复dsr环境

---

### 10. [环境] DSR动作空间异构问题
- **文件**: `envs/dsr/dsr_env.py:259`
- **问题**: 所有智能体使用相同的max动作空间大小，导致无效动作被采样
- **修复方案**: 按智能体类型设置不同动作空间
```python
def _setup_initial_action_spaces(self):
    self.action_space = []
    for i in range(self.n_agents):
        agent_type = self.core_env.agent_types[i]
        if agent_type == 'switch':
            n_actions = len(self.core_env.faultable_lines) + 1
            self.action_space.append(Discrete(n_actions))
        elif agent_type == 'pv':
            self.action_space.append(Discrete(self.config.pv_power_levels))
        elif agent_type == 'load':
            self.action_space.append(Discrete(self.config.load_action_levels))
```
- [ ] 修复完成

---

### 11. [环境] 统一get_avail_actions()返回值
- **文件**: `envs/dsr/dsr_env.py`
- **问题**: 连续动作空间返回List而非None
- **修复方案**: 连续动作空间统一返回None
- [ ] 修复完成

---

### 12. [算法] 删除冗余空实现类
- **文件**:
  - `algorithms/actors/maddpg.py` - MADDPG类完全继承HADDPG无新功能
  - `algorithms/actors/matd3.py` - MATD3类完全继承HATD3无新功能
- **修复方案A（推荐）**: 删除类文件，在注册表中使用别名
```python
ALGO_REGISTRY = {
    "haddpg": HADDPG,
    "maddpg": HADDPG,  # 别名
    "hatd3": HATD3,
    "matd3": HATD3,    # 别名
}
```
- [ ] 修改`algorithms/actors/__init__.py`
- [ ] 删除maddpg.py和matd3.py

---

### 13. [算法] HAPPO数值稳定性改进
- **文件**: `algorithms/actors/happo.py:158-172`
- **问题**: epsilon值不一致（1e-3 vs 1e-5），逻辑不一致
- **修复方案**: 统一epsilon值
- [ ] 修复完成

---

### 14. [架构] 修复62处相对导入违规
- **涉及范围**: 全项目（主要在envs/目录）
- **问题**: 违反项目规范"使用基于根目录的绝对导入，严禁使用相对导入"
- **修复方案**:
```python
# 转换前: from .circuit_system import Circuits
# 转换后: from envs.powerzoo_llm.circuit_system import Circuits
```
- [ ] 编写自动化转换脚本
- [ ] 执行批量转换
- [ ] 运行`tools/analyze_imports.py`验证
- [ ] 添加pre-commit hook防止新增相对导入

---

### 15. [代码质量] 解决类名冲突
- **文件**:
  - `models/base/act.py` - 定义`ACTLayer`
  - `models/base/qmix_act.py` - 也定义`ACTLayer`
- **问题**: 两个同名类功能不同，可能导致错误导入
- **修复方案**: 重命名qmix_act.py中的类为`QmixACTLayer`
- [ ] 修复完成

---

### 16. [算法] Twin Q-Critic添加no_grad上下文
- **文件**: `algorithms/critics/twin_continuous_q_critic.py:141-142`
- **问题**: 计算目标Q值时未使用`torch.no_grad()`，浪费内存
- **修复方案**:
```python
with torch.no_grad():
    next_q_values1 = self.target_critic(next_share_obs, next_actions)
    next_q_values2 = self.target_critic2(next_share_obs, next_actions)
    next_q_values = torch.min(next_q_values1, next_q_values2)
```
- [ ] 修复完成

---

### 17. [算法] OffPolicyBase抽象化
- **文件**: `algorithms/actors/off_policy_base.py`
- **问题**: 基类方法完全为空，未使用`abc.ABC`和`@abstractmethod`
- **修复方案**: 使用抽象基类模式
- [ ] 重构完成

---

### 18. [环境] 合并冗余README文档
- **文件**:
  - `envs/stackelberg/README.md`
  - `envs/stackelberg/README_STACKELBERG.md`
- **问题**: 内容重叠>60%
- [ ] 合并为单一README文件

---

### 19. [算法] 抽取公共advantages归一化方法
- **文件**: `algorithms/actors/on_policy_base.py`
- **问题**: HAPPO、MAPPO、HATRPO中重复相同逻辑
- **修复方案**: 抽取为`OnPolicyBase.normalize_advantages()`方法
- [ ] 创建公共方法
- [ ] 更新各算法调用

---

### 20. [架构] 为Legacy环境添加废弃警告
- **文件**: `envs/powerzoo/__init__.py`
- **修复方案**:
```python
import warnings
warnings.warn(
    "envs.powerzoo is deprecated, use envs.powerzoo_llm instead. "
    "This module will be removed in v2.0",
    DeprecationWarning,
    stacklevel=2
)
```
- [ ] 添加警告
- [ ] 更新文档

---

## 📋 P2 - 中优先级（下周内完成）

### 21. [架构] 合并common目录到utils
- **涉及范围**:
  - `common/base_logger.py` → `utils/base_logger.py`
  - `common/buffers/` → `utils/buffers/`
  - `common/valuenorm.py` → `utils/valuenorm.py`
- **工作量**: 1周
- [ ] 建立导入依赖图
- [ ] 迁移文件
- [ ] 全局替换导入路径
- [ ] 测试验证

---

### 22. [环境] 抽取公共电路适配器
- **文件**:
  - `envs/powerzoo/powerzoo/circuit.py` (~1400行)
  - `envs/powerzoo_llm/circuit_system/circuit.py` (~600行)
  - `envs/dsr/core/circuit.py` (~400行)
- **修复方案**: 创建`utils/circuit/base_circuit_adapter.py`基类
- **工作量**: 8小时
- [ ] 创建基类
- [ ] 迁移各环境电路代码

---

### 23. [环境] 合并日志记录器实现
- **文件** (4套独立实现):
  - `envs/powerzoo/powerzoo_logger.py`
  - `envs/powerzoo_llm/logging/powerzoo_llm_logger.py`
  - `envs/stackelberg/stackelberg_logger.py`
  - `envs/dsr/dsr_logger.py`
- **修复方案**: 创建`utils/logging/base_env_logger.py`统一基类
- **工作量**: 6小时
- [ ] 创建统一日志接口
- [ ] 迁移各环境logger

---

### 24. [环境] 抽取负载配置文件加载器
- **文件**:
  - `envs/powerzoo/powerzoo/loadprofile.py` (~300行)
  - `envs/powerzoo_llm/data_process/loadprofile*.py` (~800行)
  - `envs/dsr/core/loadprofile.py` (~200行)
- **修复方案**: 创建`utils/data_processing/base_load_profile.py`
- **工作量**: 5小时
- [ ] 创建基类
- [ ] 迁移重复代码

---

### 25. [架构] 统一配置格式
- **问题**: 混用Python配置文件和YAML配置
- **文件**: `configs/dan_happo_config.py`
- **修复方案**: 转换为YAML格式
- [ ] 转换配置文件
- [ ] 更新加载逻辑

---

### 26. [架构] 重组examples目录结构
- **当前问题**: `.sh`启动脚本和`.py`训练脚本混在一起
- **目标结构**:
```
examples/
├── scripts/          # 所有.py训练脚本
│   ├── train_multi_agent.py
│   └── train_single_agent.py
└── launchers/        # 所有.sh启动脚本
    ├── multi_agent/
    └── single_agent/
```
- [ ] 重组目录
- [ ] 更新脚本中的路径引用

---

### 27. [代码质量] 修复77个缺少类型提示的函数
- **涉及目录**: utils/, common/, models/, tools/
- **工作量**: 持续改进
- [ ] utils/目录函数类型提示
- [ ] common/目录函数类型提示
- [ ] models/目录函数类型提示
- [ ] tools/目录函数类型提示

---

### 28. [算法] 重构happo_diagnostics集成方式
- **文件**: `utils/happo_diagnostics.py`
- **问题**: 诊断代码侵入核心算法，违反关注点分离
- **修复方案**: 使用装饰器模式或回调机制
- [ ] 创建DiagnosticWrapper类
- [ ] 从算法核心代码中移除诊断调用

---

### 29. [算法] 批量数据转换优化
- **文件**: `algorithms/actors/happo.py:68-71`
- **问题**: 每个张量单独调用`.to(device)`，多次CUDA同步
- **修复方案**: 批量转换减少同步次数
- [ ] 优化完成

---

### 30. [环境] 创建统一环境工厂
- **问题**: 多个文件有重复的环境注册逻辑
- **修复方案**: 创建`utils/env_factory.py`
```python
class EnvironmentRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(env_class):
            cls._registry[name] = env_class
            return env_class
        return wrapper
```
- [ ] 创建环境工厂
- [ ] 迁移注册逻辑

---

### 31. [环境] 抽取智能体-设备映射逻辑
- **涉及文件**:
  - `envs/powerzoo_llm/base_env/powerzoo_env.py:_setup_agents()`
  - `envs/stackelberg/stackelberg_game/stackelberg_base_env.py:_init_agents()`
  - `envs/dsr/dsr_env.py:_setup_agents()`
- **修复方案**: 创建`utils/agent_mapping/device_mapper.py`
- [ ] 创建DeviceToAgentMapper类

---

### 32. [环境] 创建HAPPO兼容性测试套件
- **文件**: `tests/test_happo_compatibility.py`
- **内容**: 验证所有环境符合HAPPO接口规范
- [ ] 创建测试文件
- [ ] 添加到CI流程

---

### 33. [环境] 合并监控器实现
- **文件**:
  - `envs/stackelberg/stackelberg_game/stackelberg_monitor.py`
  - `envs/dsr/dsr_monitor.py`
- **修复方案**: 创建`utils/monitoring/base_monitor.py`
- [ ] 创建基类
- [ ] 迁移代码

---

### 34. [架构] 移动papers目录到docs/papers
- **当前位置**: `/papers/`
- **目标位置**: `/docs/papers/`
- [ ] 移动目录
- [ ] 更新任何引用

---

### 35. [架构] 配置系统升级
- **目标特性**:
  - 配置继承与组合
  - Pydantic运行时验证
  - 环境变量注入
- **工作量**: 3周
- [ ] 创建配置schema
- [ ] 实现配置继承

---

## 📝 P3 - 低优先级（技术债务，逐步优化）

### 36-40. [代码质量] 删除临时/调试文件
- [ ] 清理所有`*_old.py`, `*_backup.py`模式文件
- [ ] 清理`*_optimized.py`, `*_enhanced.py`模式文件（需确认是否已合并）
- [ ] 验证功能迁移完成后删除

### 41-45. [代码质量] 消除魔法数字
- [ ] `algorithms/actors/happo.py` - 提取1e-5, 1e-3等为类常量
- [ ] `algorithms/actors/mappo.py` - 同上
- [ ] `algorithms/critics/` - 同上

### 46-50. [代码质量] 重构超长函数
- [ ] 识别所有>100行的函数
- [ ] 分解为更小的函数

### 51-55. [代码质量] 减少深层嵌套
- [ ] 识别>4层嵌套的代码块
- [ ] 使用early return模式重构

### 56-60. [算法] 添加算法单元测试
- [ ] `tests/algorithms/test_happo.py`
- [ ] `tests/algorithms/test_mappo.py`
- [ ] `tests/algorithms/test_hatd3.py`
- [ ] `tests/algorithms/test_hasac.py`

### 61-65. [架构] 建立完整测试体系
- **目标覆盖率**: 从<5%提升到60%+
- [ ] Month 1: 核心环境测试（目标30%）
- [ ] Month 2: 工具模块测试（目标50%）
- [ ] Month 3: 端到端测试（目标60%）

### 66-70. [算法] 性能优化
- [ ] Twin Q-Critic JIT编译优化
- [ ] HATRPO共轭梯度法缓存
- [ ] 减少不必要的梯度计算

### 71-75. [文档] 新增文档
- [ ] `docs/MARL_Interface_Standard.md`
- [ ] `docs/Environment_Migration_Guide.md`
- [ ] `docs/HAPPO_Compatibility_Checklist.md`

---

## 📁 冗余文件/目录清单

### 可安全删除
| 路径 | 大小 | 原因 | 风险 |
|------|------|------|------|
| `.yoyo/` | 342MB | 历史快照，Git已有历史 | 低 |
| `.trae/` | 16KB | 用途不明 | 低 |
| `utils/sign.py` | 382行 | 完全未使用 | 零 |
| `utils/._envs_tools.py` | - | Mac临时文件 | 零 |
| `envs/powerzoo/powerzoo/._env.py` | - | Mac临时文件 | 零 |
| `algorithms/actors/maddpg.py` | - | 空实现类 | 低 |
| `algorithms/actors/matd3.py` | - | 空实现类 | 低 |

### 待废弃（需迁移后删除）
| 模块 | 文件数 | 替代方案 | 迁移时间 |
|------|--------|----------|----------|
| `envs/powerzoo/` | 8 | `envs/powerzoo_llm/` | 3个月 |
| `common/` | 3 | 合并到`utils/` | 2周 |

### 待合并
| 文件 | 合并目标 |
|------|----------|
| `envs/stackelberg/README_STACKELBERG.md` | `envs/stackelberg/README.md` |
| 4个日志记录器 | `utils/logging/base_env_logger.py` |
| 3个电路适配器 | `utils/circuit/base_circuit_adapter.py` |
| 3个负载配置加载器 | `utils/data_processing/base_load_profile.py` |

---

## 📈 成功指标

重构完成后应达到以下指标：

### 代码质量
- [ ] 相对导入数量: **0** (当前62)
- [ ] 死代码文件: **0** (当前2+)
- [ ] 重复环境实现: **1** (当前2套)
- [ ] 测试覆盖率: **≥60%** (当前<5%)

### HAPPO兼容性
- [ ] powerzoo: **≥80%** (当前23%)
- [ ] stackelberg: **100%** (当前85%)
- [ ] dsr: **≥90%** (当前72%)
- [ ] powerzoo_llm: **100%** (当前95%)

### 性能指标
- [ ] Git仓库大小: **<100MB** (当前336MB)
- [ ] 项目磁盘占用: **<500MB** (当前~1GB含.yoyo)

---

## 🗓️ 实施路线图

### Phase 1: 快速清理 (Week 1-2)
- [ ] 删除历史遗留代码（释放360MB）
- [ ] 修复P0问题（8项）
- [ ] 添加pre-commit hooks

### Phase 2: 结构优化 (Month 1)
- [ ] 修复P1问题（12项）
- [ ] 修复导入规范
- [ ] 建立核心测试（30%覆盖率）

### Phase 3: 系统重构 (Month 2-3)
- [ ] 完成P2问题（15项）
- [ ] 统一日志/配置系统
- [ ] 测试覆盖率提升到60%

### Phase 4: 持续改进
- [ ] 逐步解决P3问题
- [ ] 建立CI/CD流程
- [ ] 发布v2.0（完全移除旧版代码）

---

## 📚 参考资料

- 环境兼容性审查报告: `docs/environment_compatibility_audit_report.md`
- 项目架构审查报告: `docs/architecture_review_report.md`

---

**审查团队**:
- RL算法专家 (rl-algorithm-specialist)
- 代码质量专家 (code-reviewer)
- 环境兼容性专家 (env-compatibility-reviewer)
- 架构专家 (python-architect-expert)

**报告版本**: v1.0
**下次审查建议**: 2025-02-01
