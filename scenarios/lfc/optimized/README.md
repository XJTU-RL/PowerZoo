# LFC 优化版

优化版是独立的工程与科研场景线。不替换论文原版，也不自动开启实验功能。每个变体记录实际馈线、接口和版本。

| 变体 | 改进方向 | 已通过／下一关 |
| --- | --- | --- |
| 13 时钟与储能 | 可选的秒级对齐、失败后必须重置、用实测端口功率核算能量 | [时钟研究分支](https://github.com/XJTU-RL/PowerZoo/blob/research/lfc-13-coupling-validation/docs/lfc-13-research-validation.md)：真实短测通过；储能账本当前只有离线测试，尚未验证真实闭环。 |
| 34 物理校准 | 核对电压、相别、SOC、变压器负荷，建立可行的输配接口初态 | [运行点诊断](https://github.com/XJTU-RL/PowerZoo/blob/sheldon123z/lfc-34bus-calibrated/docs/lfc-34bus-operating-point.md)：功率平衡可重现，电压筛查尚未通过；不进入联合仿真。 |
| [123 原生储能](../ieee123-native/README.md) | 复用原馈线和四台原生储能，以实测功率接入 Bus 8 | [配电侧检查](https://github.com/XJTU-RL/PowerZoo/blob/benchmark/lfc-123-native/docs/lfc-123-native-probe.md)与无扰动传感器暖启动已通过；脉冲传递尚未验收，未验证双向耦合。 |
| OEDI 123 扩展 | 用公开负荷、光伏与测点做部分观测对照 | [时序潮流验收](https://github.com/XJTU-RL/PowerZoo/blob/benchmark/external-temporal-parity/docs/external-temporal-parity.md)已完成；15 分钟曲线用于构造工况，秒级调频扰动单独定义。尚未接入 LFC。 |

## 控制与优化的验证顺序

1. **先验物理与接口**：无扰动运行点、实测 P/Q、时钟和采样、能量守恒，以及电压、设备容量、SOC、爬坡和可用备用。单向 P/Q 传递不等于双向闭环。
2. **建立基线**：在同一初态和扰动下比较无控制、固定下垂与带约束的优化基线。同步记录频率偏差、最低／最高点、恢复时间、配网越限及储能用量。
3. **再比较学习算法**：把预测控制和 OHASAC 等作为独立实验配置，使用相同物理模型、分离验证／测试工况，报告多随机种子及不可行、未恢复的情况。更强是待验证的目标，不是当前结论。
