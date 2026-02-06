# PowerZoo

PowerZoo是一个基于Python和OpenDSS的电力系统强化学习仿真环境。它旨在提供一个灵活且易于使用的平台，用于训练和评估强化学习算法在电力系统中的应用。

## 特性

- 🔌 **电力系统仿真**: 基于OpenDSS的电网仿真
- 🤖 **多智能体强化学习**: 支持14种MARL算法（HAPPO, MAPPO, MADDPG等）
- 🔋 **多种环境**: PowerZoo（基础）和PowerZoo_LLM（完整）
- 📊 **丰富的设备模型**: 电容器、调压器、电池、光伏系统
- 🎯 **单/多智能体支持**: 灵活的智能体配置
- 📈 **完整监控**: TensorBoard集成

## 安装

### 方式1：使用Conda（推荐）

```bash
git clone https://github.com/XJTU-RL/PowerZoo.git
cd PowerZoo
conda env create -f environment.yml
conda activate PowerZoo
```

### 方式2：使用pip

```bash
git clone https://github.com/XJTU-RL/PowerZoo.git
cd PowerZoo

# 安装核心依赖
pip install -r requirements.txt

# 或者使用开发模式安装（包含测试工具）
pip install -e ".[dev]"
```

## 快速开始

### 多智能体训练

```bash
# HAPPO算法训练PV控制
cd examples/multi_agent/launchers
./quick_train_powerzoo_pv.sh aggressive
```

### 单智能体训练

```bash
# PPO算法训练
cd examples/single_agent/launchers
./train_single_agent_powerzoo_pv.sh
```

### Python脚本训练

```python
from envs.smartgrid.base_env import VVCEnv
from stable_baselines3 import PPO

# 创建环境
config = {
	"dss_folder_path": "node_systems/34Bus_PV_Aggressive",
	"dss_file": "ieee34Mod1_duty.dss",
	"num_agents": 10,
	"episode_length": 96
}
env = VVCEnv(**config)

# 训练模型
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## 测试

运行测试套件：

```bash
# 运行所有测试
pytest

# 运行特定环境测试
pytest -m smartgrid

# 生成覆盖率报告
pytest --cov=envs --cov-report=html
```

详见 [tests/README.md](tests/README.md)

## 项目结构

```
PowerZoo/
├── envs/                      # 环境实现
│   ├── powerzoo/             # 基础PowerZoo环境
│   ├── smartgrid/         # 完整PowerZoo_LLM环境
│   ├── stackelberg/          # Stackelberg博弈环境
│   └── dsr/                  # 需求侧响应环境
├── algorithms/               # RL算法实现
├── runners/                  # 训练运行器
├── models/                   # 神经网络模型
├── configs/                  # 配置文件
│   ├── envs_cfgs/           # 环境配置
│   └── algos_cfgs/          # 算法配置
├── examples/                 # 训练示例
│   ├── multi_agent/         # 多智能体示例
│   └── single_agent/        # 单智能体示例
├── tests/                    # 测试套件
├── node_systems/             # IEEE标准测试系统
└── data/                     # PV和负荷数据
```

## 文档

- [PowerZoo环境说明](PowerZoo环境说明.md)
- [示例脚本说明](examples/README.md)
- [单智能体训练](examples/single_agent/README.md)
- [多智能体训练](examples/multi_agent/README.md)
- [测试文档](tests/README.md)

## 贡献指南

### 分支管理

本项目使用规范的分支管理策略。在开发新功能或修复bug前，请先阅读：

- **[分支管理文档](BRANCH_MANAGEMENT.md)** - 详细的分支管理指南和最佳实践
- **[分支清理清单](BRANCH_CLEANUP_CHECKLIST.md)** - 当前分支状态和待处理事项

**重要**: `fea/vvc` 分支是独立研发分支，不应合并到main。

### 开发流程

1. 从main分支创建功能分支：`git checkout -b feat/your-feature`
2. 开发并提交代码
3. 运行测试确保代码质量：`pytest`
4. 提交Pull Request
5. 代码审查通过后合并到main
6. 及时删除已合并的分支

## Roadmap

- [x] Volt-Var Regulation (SHOM)
- [x] PowerZoo & PowerZoo_LLM环境
- [x] 14种MARL算法
- [x] 单智能体支持
- [ ] Frequency Regulation (OHASAC)
- [ ] RL based EVCS Control
- [ ] RL based Hybrid Energy System

---
Please cite the following paper if you use this repo for scientific research:

[1] X. Zheng, S. Yu, H. Cao, T. Shi, S. Xue, and T. Ding, “Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network,” IEEE Transactions on Smart Grid, pp. 1–1, Feb. 2025, doi: 10.1109/TSG.2025.3540416.
```
@article{zhengSensitivityBasedHeterogeneousOrdered2025,
  author  = {Xiaodong Zheng and Shixuan Yu and Hui Cao and Tianzhuo Shi and Shuangsi Xue and Tao Ding},
  title   = {Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2025},
  month   = {Feb},
  doi     = {10.1109/TSG.2025.3540416},
  issn    = {1949-3061},
  url     = {https://ieeexplore.ieee.org/document/10879343},
  urldate = {2025-02-12}
}
```

