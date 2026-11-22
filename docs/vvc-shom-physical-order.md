# IEEE34 SHOM 物理排序复现与扩展

本路线沿用 IEEE34 经典 VVC 的异构设备和动作定义，不要求开启储能专用模式，
也不依赖离散基线 PR。仅更改 SHOM 的 actor 优化顺序；不会重新排列仿真器的
设备、动作下发、观测或状态轨迹。论文式打分及雅可比复用已有研究实现
`49522e54`，本分支不把工程回归当作已复现论文性能。

| 模式 | 计算口径 |
| --- | --- |
| `legacy` | 默认，保留原有入口行为，不要求物理测量。 |
| `paper` | 从同一已收敛运行点计算 V–Q 雅可比。按各设备映射节点的绝对行和与带符号的回合奖励权重评分。 |
| `native` | 在同源隔离的 OpenDSS 副本中复核运行点，分别翻转电容器、移动调压器一档、扰动储能相邻可用指令。以实际下发的归一化差求电压响应，每次核对副本恢复到基准；按完整回合和 worker 平均，平分按稳定设备 ID 排序。此模式是工程扩展，不是论文公式。 |

物理模式要求 VVC、EP、独立 actor、联合概率比和自然终止的完整回合。
`algo.sensitivity_mode` 必须匹配 `env.sensitivity_measurement`；物理模型、
动作映射、场景和时刻必须一致。缺失、未收敛、失配或无有效扰动时拒绝物理
排序，不静默退回 `legacy`。`native` 当前不能用逐设备 SMART-DS 时序重建；
其它馈线须单独核验探测可用性。

## 复现范围

```sh
python scripts/train.py --config configs/experiments/vvc_34bus_shom_native_quickstart.yaml
python scripts/train.py --config configs/experiments/vvc_34bus_shom_native_quickstart.yaml \
    --set algo.sensitivity_mode=paper --set env.sensitivity_measurement=paper
python -m pytest -q -o addopts= tests/test_vvc_shom_physical_order.py
```

短验证使用合成工况、真实 OpenDSS，以及 IEEE34 的 10 台受控设备。逐列与独立
设备控制比较，核对主仿真轨迹不受隔离探测影响，并挂钩 actor 优化器核验实际
更新次序与模型重新加载后的轨迹。未验证论文收益、排名、其它数据集或长训练。
