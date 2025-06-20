# Claude Code PowerZoo新环境建设指示

## 项目概述
你需要为PowerZoo强化学习环境实现一个基于异步多智能体强化学习的双级非合作博弈论需求响应框架。这个框架将整合Stackelberg-Nash博弈机制，支持公用事业公司（UC）和消费者之间的分层决策。

## 核心要求

### 1. 环境架构
PowerZoo环境已经存在，你需要在其基础上扩展，而不是重新创建。保持与现有接口的兼容性：

```python
# 继承现有的Env类
from envs.powerzoo.powerzoo.env import Env as BaseEnv

class StackelbergEnv(BaseEnv):
    """扩展的Stackelberg博弈环境"""
    def __init__(self, folder_path, info, dss_act=False):
        super().__init__(folder_path, info, dss_act)
        # 添加新功能
```

### 2. 异步更新机制实现

创建文件：`envs/powerzoo/powerzoo/async_wrapper.py`

```python
class AsyncMultiAgentWrapper:
    """
    实现异步多智能体更新机制
    - UC首先行动
    - 消费者观察UC动作后响应
    - 维护动作历史缓冲区
    """
    def __init__(self, env):
        self.env = env
        self.uc_action_buffer = None
        self.consumer_response_delay = 1
        
    def step_uc(self, uc_action):
        """UC先行动"""
        # 存储UC动作
        # 不立即执行环境step
        pass
        
    def step_consumers(self, consumer_actions):
        """消费者后响应"""
        # 基于UC动作和消费者动作执行环境step
        pass
```

### 3. SN-MAPPO算法实现

创建文件：`algorithms/actors/sn_mappo.py`

```python
from algorithms.actors.mappo import MAPPO

class SN_MAPPO(MAPPO):
    """Stackelberg-Nash MAPPO算法"""
    def __init__(self, args, obs_space, act_space, device):
        super().__init__(args, obs_space, act_space, device)
        self.is_leader = args.get('is_leader', False)
        self.hierarchy_level = args.get('hierarchy_level', 0)
        
    def update(self, sample):
        """异步更新逻辑"""
        if self.is_leader:
            # UC更新逻辑
            return self.update_leader(sample)
        else:
            # 消费者更新逻辑
            return self.update_follower(sample)
```

### 4. 智能负荷聚合器

创建文件：`envs/powerzoo/powerzoo/load_aggregator.py`

```python
class IntelligentLoadAggregator:
    """
    实现智能负荷聚合
    支持zone、priority、random三种方法
    新增基于图结构的智能聚合
    """
    def __init__(self, circuit, method='zone'):
        self.circuit = circuit
        self.method = method
        self.load_groups = {}
        
    def aggregate_loads(self, n_agents):
        """
        将实际负荷聚合为n_agents个智能体
        返回聚合映射关系
        """
        if self.method == 'zone':
            return self.zone_based_aggregation(n_agents)
        elif self.method == 'priority':
            return self.priority_based_aggregation(n_agents)
        elif self.method == 'graph':
            return self.graph_based_aggregation(n_agents)
```

### 5. 多时间尺度协调器

创建文件：`envs/powerzoo/powerzoo/multi_timescale.py`

```python
class MultiTimescaleCoordinator:
    """
    实现日前-日内-实时三级协调
    """
    def __init__(self, horizon_hours=24):
        self.day_ahead_horizon = horizon_hours
        self.intraday_horizon = 4
        self.real_time_horizon = 0.083  # 5分钟
        
    def day_ahead_schedule(self, forecast_data):
        """日前调度计划"""
        pass
        
    def intraday_update(self, current_state, day_ahead_plan):
        """日内滚动修正"""
        pass
        
    def real_time_dispatch(self, immediate_state):
        """实时调度"""
        pass
```

### 6. 集成真实电力约束

扩展文件：`envs/powerzoo/powerzoo/circuit.py`

```python
# 在现有Circuit类中添加方法
def check_n_minus_1_security(self, action):
    """N-1安全校验"""
    # 遍历所有线路
    # 模拟单一故障
    # 检查系统是否仍然安全
    pass
    
def calculate_carbon_emissions(self):
    """计算碳排放"""
    emissions = 0
    for gen in self.generators:
        emissions += gen.power * gen.emission_factor
    return emissions
```

### 7. 测试基准实现

创建文件：`tests/test_stackelberg_env.py`

```python
import pytest
from envs.powerzoo.powerzoo.stackelberg_env import StackelbergEnv

class TestStackelbergEnv:
    def test_async_update(self):
        """测试异步更新机制"""
        env = StackelbergEnv(...)
        # 测试UC先动
        # 测试消费者响应
        # 验证时序关系
        
    def test_load_aggregation(self):
        """测试负荷聚合"""
        # 测试不同聚合方法
        # 验证聚合后的动作空间
        # 检查聚合的可逆性
```

## 实现步骤

### 第1步：环境扩展（优先级：高）
1. 创建`stackelberg_env.py`，继承现有Env类
2. 添加UC和消费者的分层结构
3. 实现异步step方法

### 第2步：算法实现（优先级：高）
1. 创建`sn_mappo.py`，基于现有MAPPO
2. 实现异步更新逻辑
3. 添加Stackelberg均衡计算

### 第3步：负荷聚合（优先级：中）
1. 扩展现有的负荷聚合功能
2. 添加基于图的智能聚合
3. 确保与DSR环境兼容

### 第4步：约束集成（优先级：中）
1. 在Circuit类中添加N-1校验
2. 集成碳排放计算
3. 添加更严格的电压和潮流约束

### 第5步：测试验证（优先级：高）
1. 为每个新功能编写单元测试
2. 创建集成测试用例
3. 基准性能测试

## 注意事项

1. **保持兼容性**：所有新功能都应该是可选的，不破坏现有功能
2. **模块化设计**：每个新功能都应该是独立模块，便于维护
3. **文档完善**：为每个新类和方法添加详细的docstring
4. **性能优化**：注意大规模系统（8500-Node）的计算效率
5. **错误处理**：添加适当的异常处理和日志记录

## 代码规范

1. 遵循PEP 8规范
2. 使用类型提示（typing）
3. 添加适当的注释，特别是复杂逻辑
4. 变量命名清晰，避免缩写

## 测试要求

1. 单元测试覆盖率>80%
2. 所有公共API都需要测试
3. 边界条件和异常情况测试
4. 性能基准测试

开始实现时，请先创建基础框架，然后逐步添加功能。每完成一个模块后进行测试，确保稳定性。