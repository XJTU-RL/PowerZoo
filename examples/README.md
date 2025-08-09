# PowerZoo训练示例

本目录包含PowerZoo环境的所有训练示例和脚本，按照智能体类型进行组织。

## 📁 目录结构

```
examples/
├── multi_agent/           # 多智能体强化学习
│   ├── scripts/          # Python训练脚本
│   └── launchers/        # Shell启动脚本
├── single_agent/         # 单智能体强化学习
│   ├── scripts/          # Python训练脚本
│   └── launchers/        # Shell启动脚本
└── scripts/              # 共用工具脚本
```

## 🚀 快速开始

### 多智能体训练

```bash
# 使用HAPPO算法训练PV控制（激进方案）
cd multi_agent/launchers
./quick_train_happo_pv.sh aggressive

# 使用完整参数配置
./train_happo_pv_full.sh --plan optimized --gpu 0 --threads 32
```

### 单智能体训练

```bash
# 使用PPO算法训练
cd single_agent/launchers
./train_single_agent_powerzoo_pv.sh

# 使用DDPG算法测试
./test_ddpg.sh
```

## 📊 算法支持

### 多智能体算法
- **HAPPO**: 异构智能体近端策略优化
- **MAPPO**: 多智能体近端策略优化
- **HATRPO**: 异构智能体信任域策略优化
- **MADDPG**: 多智能体深度确定性策略梯度
- **MATD3**: 多智能体双延迟深度确定性策略梯度

### 单智能体算法
- **PPO**: 近端策略优化
- **SAC**: 软演员评论家
- **TD3**: 双延迟深度确定性策略梯度
- **DDPG**: 深度确定性策略梯度
- **A2C**: 优势演员评论家

## 🔧 PV方案配置

系统支持三种PV渗透率方案：

1. **保守方案** (Conservative)
   - 容量: 720kW
   - 渗透率: 40.7%
   - 风险: 低

2. **优化方案** (Optimized)
   - 容量: 900kW
   - 渗透率: 50.8%
   - 风险: 中等

3. **激进方案** (Aggressive)
   - 容量: 1080kW
   - 渗透率: 61%
   - 风险: 高

## 📝 详细文档

- 多智能体训练详情: [multi_agent/README.md](multi_agent/README.md)
- 单智能体训练详情: [single_agent/README.md](single_agent/README.md)

## 🎯 使用建议

1. **初学者**: 从单智能体训练开始，使用PPO算法
2. **进阶用户**: 尝试多智能体HAPPO算法，优化方案
3. **研究人员**: 使用完整参数配置脚本，自定义奖励权重

## ⚠️ 注意事项

- 确保已安装所有依赖包
- GPU训练需要CUDA支持
- 首次运行建议使用`--dry_run`参数检查配置