# SHOM 解耦架构说明文档

## 概述

本文档介绍 SHOM (Sequential Heterogeneous-agent Optimization Module) 算法的解耦架构实现。该架构将灵敏度计算和智能体排序逻辑从环境中分离出来，提供了更好的可扩展性和可维护性。

## 目录

1. [架构设计](#架构设计)
2. [核心组件](#核心组件)
3. [使用指南](#使用指南)
4. [扩展指南](#扩展指南)
5. [API 参考](#api-参考)
6. [配置说明](#配置说明)

---

## 架构设计

### 原有架构 vs 解耦架构

**原有架构** (`on_policy_ha_runner.py`)：
- 灵敏度计算逻辑直接嵌入在 `train()` 方法中
- 排序逻辑与 PowerZoo 环境强耦合
- 难以扩展到其他环境

**解耦架构** (`on_policy_ha_runner_decoupled.py`)：
- 灵敏度计算通过抽象接口 `SensitivityCalculator` 实现
- 排序策略通过策略模式 `AgentOrderStrategy` 实现
- 易于扩展到任何多智能体环境

```
┌─────────────────────────────────────────────────────────────┐
│                      解耦架构示意图                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐     ┌─────────────────────────────┐    │
│  │   环境层        │     │   utils/sensitivity.py      │    │
│  │   (env.py)      │────▶│   SensitivityCalculator     │    │
│  │   提供原始数据   │     │   - PowerGridSensitivity    │    │
│  └─────────────────┘     │   - UniformSensitivity      │    │
│                          └──────────────┬──────────────┘    │
│                                         │                    │
│                                         ▼                    │
│                          ┌─────────────────────────────┐    │
│                          │   utils/agent_ordering.py   │    │
│                          │   AgentOrderStrategy        │    │
│                          │   - SensitivityOrder        │    │
│                          │   - FixedOrder              │    │
│                          │   - RandomOrder             │    │
│                          └──────────────┬──────────────┘    │
│                                         │                    │
│                                         ▼                    │
│                          ┌─────────────────────────────┐    │
│                          │   AgentOrderManager         │    │
│                          │   组合灵敏度计算和排序策略    │    │
│                          └──────────────┬──────────────┘    │
│                                         │                    │
│                                         ▼                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │   OnPolicyHARunnerDecoupled                         │    │
│  │   使用 AgentOrderManager 确定更新顺序                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. SensitivityCalculator (`utils/sensitivity.py`)

灵敏度计算的抽象基类，定义了计算智能体灵敏度的标准接口。

```python
class SensitivityCalculator(ABC):
    @abstractmethod
    def compute(self, env_info: Dict) -> Dict[str, float]:
        """计算单步灵敏度"""
        pass

    @abstractmethod
    def aggregate(self, sensitivity_history: List[Dict]) -> Dict[str, float]:
        """聚合多步灵敏度"""
        pass
```

**内置实现：**

| 类名 | 用途 | 适用环境 |
|------|------|----------|
| `PowerGridSensitivity` | 基于无功电压灵敏度矩阵 | PowerZoo 电网环境 |
| `UniformSensitivity` | 均匀灵敏度（默认） | 任何环境 |

### 2. AgentOrderStrategy (`utils/agent_ordering.py`)

智能体排序的策略模式基类。

```python
class AgentOrderStrategy(ABC):
    @abstractmethod
    def get_order(self, num_agents: int, **kwargs) -> List[int]:
        """返回智能体更新顺序"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass
```

**内置策略：**

| 策略 | 说明 | 配置 |
|------|------|------|
| `SensitivityOrder` | 按灵敏度排序 | `useS=True` |
| `FixedOrder` | 固定顺序 [0,1,2,...] | `ordered=True, useS=False` |
| `RandomOrder` | 随机排列 | `ordered=False` |

### 3. AgentOrderManager

组合灵敏度计算器和排序策略，提供统一接口。

```python
manager = AgentOrderManager(
    strategy=SensitivityOrder(...),
    sensitivity_calculator=PowerGridSensitivity(...)
)

order = manager.compute_order(
    num_agents=5,
    buffer_infos=critic_buffer.infos,
    verbose=True
)
```

---

## 使用指南

### 方式一：使用解耦版 Runner

```python
# 在你的训练脚本中
from runners.on_policy_ha_runner_decoupled import OnPolicyHARunnerDecoupled

runner = OnPolicyHARunnerDecoupled(args, algo_args, env_args)
runner.run()
```

### 方式二：在原有 Runner 中集成

如果你想在原有代码基础上使用解耦组件：

```python
from utils.agent_ordering import create_order_manager

# 在 __init__ 中初始化
self.order_manager = create_order_manager(
    env_name=args["env"],
    env_args=env_args,
    algo_args=algo_args,
    num_agents=self.num_agents,
    agents_bus_mapping=self.get_agents_bus,
    agent_id_mapping=self.get_ordered_agents_pairs
)

# 在 train() 中使用
agent_order = self.order_manager.compute_order(
    num_agents=self.num_agents,
    buffer_infos=self.critic_buffer.infos,
    verbose=True
)
```

---

## 扩展指南

### 添加新的灵敏度计算器

假设你有一个新环境 "MyEnv"，需要自定义灵敏度计算：

**步骤 1：创建新的灵敏度计算器类**

```python
# utils/sensitivity.py

class MyEnvSensitivity(SensitivityCalculator):
    """自定义环境的灵敏度计算器"""

    def __init__(self, custom_param):
        self.custom_param = custom_param

    def compute(self, env_info: Dict) -> Dict[str, float]:
        """从环境信息中计算灵敏度"""
        # 实现你的计算逻辑
        sensitivity = {}
        for agent_name, data in env_info.get('my_data', {}).items():
            sensitivity[agent_name] = self._calculate(data)
        return sensitivity

    def aggregate(self, sensitivity_history: List[Dict]) -> Dict[str, float]:
        """聚合多步数据"""
        aggregated = {}
        for step_info in sensitivity_history:
            for key, value in step_info.items():
                aggregated[key] = aggregated.get(key, 0) + value
        return aggregated

    def _calculate(self, data):
        # 你的计算逻辑
        return sum(data) * self.custom_param
```

**步骤 2：更新工厂函数**

```python
# utils/sensitivity.py

def create_sensitivity_calculator(env_name, env_args, ...):
    # ... 现有代码 ...

    # 添加新环境支持
    if env_name == "myenv":
        return MyEnvSensitivity(env_args.get("custom_param", 1.0))

    return UniformSensitivity(num_agents)
```

### 添加新的排序策略

假设你想实现一个基于奖励的排序策略：

**步骤 1：创建新策略类**

```python
# utils/agent_ordering.py

class RewardBasedOrder(AgentOrderStrategy):
    """基于累积奖励的排序策略"""

    def __init__(self, descending: bool = True):
        self.descending = descending

    def get_order(self, num_agents: int, rewards: Dict[int, float] = None, **kwargs) -> List[int]:
        if rewards is None:
            return list(range(num_agents))

        # 按奖励排序
        sorted_agents = sorted(
            range(num_agents),
            key=lambda x: rewards.get(x, 0),
            reverse=self.descending
        )
        return sorted_agents

    @property
    def name(self) -> str:
        return "reward_based"
```

**步骤 2：在代码中使用**

```python
from utils.agent_ordering import AgentOrderManager, RewardBasedOrder

manager = AgentOrderManager(
    strategy=RewardBasedOrder(descending=True),
    sensitivity_calculator=None  # 不需要灵敏度计算
)

# 传入奖励数据
order = manager.compute_order(
    num_agents=5,
    rewards={0: 10.5, 1: 8.2, 2: 15.0, 3: 5.1, 4: 12.3}
)
# 输出: [2, 4, 0, 1, 3] (按奖励从高到低)
```

### 组合多种策略

你可以创建一个组合策略：

```python
class HybridOrder(AgentOrderStrategy):
    """混合排序策略：先按类型分组，再按灵敏度排序"""

    def __init__(self, agent_types: Dict[int, str], sensitivity_calc):
        self.agent_types = agent_types
        self.sensitivity_calc = sensitivity_calc

    def get_order(self, num_agents: int, sensitivity_values=None, **kwargs):
        # 按类型分组
        type_groups = {}
        for agent_id, agent_type in self.agent_types.items():
            type_groups.setdefault(agent_type, []).append(agent_id)

        # 每组内按灵敏度排序
        ordered = []
        for type_name in ['regulator', 'capacitor', 'battery']:  # 固定类型顺序
            if type_name in type_groups:
                group = type_groups[type_name]
                if sensitivity_values:
                    group.sort(key=lambda x: sensitivity_values.get(x, 0), reverse=True)
                ordered.extend(group)

        return ordered

    @property
    def name(self) -> str:
        return "hybrid"
```

---

## API 参考

### SensitivityCalculator

```python
class SensitivityCalculator(ABC):
    def compute(self, env_info: Dict[str, Any]) -> Dict[str, float]:
        """
        计算单步灵敏度值。

        参数:
            env_info: 环境返回的信息字典

        返回:
            Dict[str, float]: 智能体名称到灵敏度值的映射
        """
        pass

    def aggregate(self, sensitivity_history: List[Dict]) -> Dict[str, float]:
        """
        聚合多个时间步的灵敏度数据。

        参数:
            sensitivity_history: 多步灵敏度信息列表

        返回:
            Dict[str, float]: 聚合后的灵敏度值
        """
        pass
```

### AgentOrderStrategy

```python
class AgentOrderStrategy(ABC):
    def get_order(self, num_agents: int, **kwargs) -> List[int]:
        """
        确定智能体更新顺序。

        参数:
            num_agents: 智能体总数
            **kwargs: 策略特定参数（如 sensitivity_values）

        返回:
            List[int]: 智能体索引的有序列表
        """
        pass

    @property
    def name(self) -> str:
        """返回策略名称（用于日志输出）"""
        pass
```

### AgentOrderManager

```python
class AgentOrderManager:
    def __init__(
        self,
        strategy: AgentOrderStrategy,
        sensitivity_calculator: Optional[SensitivityCalculator] = None
    ):
        """
        初始化排序管理器。

        参数:
            strategy: 排序策略实例
            sensitivity_calculator: 灵敏度计算器（可选）
        """
        pass

    def compute_order(
        self,
        num_agents: int,
        buffer_infos: Optional[Dict] = None,
        verbose: bool = False
    ) -> List[int]:
        """
        计算当前训练迭代的智能体顺序。

        参数:
            num_agents: 智能体总数
            buffer_infos: 包含灵敏度数据的缓冲区信息
            verbose: 是否打印排序结果

        返回:
            List[int]: 更新顺序
        """
        pass
```

### 工厂函数

```python
def create_sensitivity_calculator(
    env_name: str,
    env_args: Dict,
    agents_bus_mapping: Optional[Dict] = None,
    num_agents: int = 0
) -> SensitivityCalculator:
    """根据环境配置创建合适的灵敏度计算器"""
    pass

def create_order_manager(
    env_name: str,
    env_args: Dict,
    algo_args: Dict,
    num_agents: int,
    agents_bus_mapping: Optional[Dict] = None,
    agent_id_mapping: Optional[Dict] = None
) -> AgentOrderManager:
    """根据配置创建完整的排序管理器"""
    pass
```

---

## 配置说明

### 环境配置 (`configs/envs_cfgs/powerzoo.yaml`)

```yaml
# 是否使用灵敏度矩阵进行排序
useS: True

# 灵敏度排序方向
# True: 从大到小（高灵敏度优先更新）
# False: 从小到大（低灵敏度优先更新）
big2small: True
```

### 算法配置 (`configs/algos_cfgs/shom.yaml`)

```yaml
algo:
  # 是否使用确定性顺序
  # True: 使用灵敏度排序或固定顺序
  # False: 每轮随机顺序
  ordered: True
```

### 配置组合效果

| `ordered` | `useS` | `big2small` | 排序行为 |
|-----------|--------|-------------|----------|
| `True` | `True` | `True` | 灵敏度从大到小 |
| `True` | `True` | `False` | 灵敏度从小到大 |
| `True` | `False` | - | 固定顺序 [0,1,2,...] |
| `False` | - | - | 每轮随机顺序 |

---

## 测试

运行单元测试验证功能：

```bash
# 运行所有优化测试
python tests/test_optimization.py

# 使用 pytest（如果可用）
python -m pytest tests/test_optimization.py -v
```

---

## 常见问题

### Q: 解耦版和原版 Runner 有什么区别？

A: 功能完全相同，区别在于代码组织：
- 原版：排序逻辑直接写在 `train()` 方法中
- 解耦版：排序逻辑通过 `AgentOrderManager` 管理，易于扩展

### Q: 如何在不修改原有代码的情况下使用新功能？

A: 直接使用 `OnPolicyHARunnerDecoupled` 替换 `OnPolicyHARunner`：
```python
from runners.on_policy_ha_runner_decoupled import OnPolicyHARunnerDecoupled
```

### Q: 新环境不需要灵敏度排序怎么办？

A: 设置 `useS: False` 和 `ordered: False`，系统会使用随机顺序。或者使用 `FixedOrder` 策略。

---

## 文件列表

```
PowerZoo/
├── utils/
│   ├── sensitivity.py          # 灵敏度计算模块
│   └── agent_ordering.py       # 排序策略模块
├── runners/
│   ├── on_policy_ha_runner.py           # 原版 Runner（保持兼容）
│   └── on_policy_ha_runner_decoupled.py # 解耦版 Runner
├── tests/
│   └── test_optimization.py    # 单元测试
└── docs/
    └── SHOM_DECOUPLED_ARCHITECTURE.md   # 本文档
```

---

## 版本历史

- **v1.0** (2026-01-25): 初始版本
  - 添加 `SensitivityCalculator` 抽象接口
  - 添加 `AgentOrderStrategy` 策略模式
  - 添加 `OnPolicyHARunnerDecoupled` 解耦版 Runner
  - 添加单元测试
