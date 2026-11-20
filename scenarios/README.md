# 场景与复现索引

| 层级 | 入口 | 记录内容 |
| --- | --- | --- |
| 基础电路 | [标准算例登记](../node_systems/case_registry.json) | 电路版本、来源、可用性及验收范围；登记不等于所有算例都已内置或已通过时序仿真。 |
| 任务场景 | [VVC](../docs/vvc-physics.md) · [SmartGrid](../docs/smartgrid-data-runtime.md) · [恢复](../docs/dsr-physics.md) · [分区调度](../docs/district-physics.md) · [顺序定价](../docs/pricing-physics.md) | 使用的电路、设备、初态、数据和验证口径。 |
| 实验配置 | [实验配置目录](../configs/experiments/) | 算法、种子及运行预算。更换算法不复制基础电路；原训练入口保持不变。 |

## LFC：论文原版与优化研究分开

- **论文原版：** [OHASAC 复现入口](../reproductions/lfc/README.md)。从原论文三机九节点输电模型和配套馈线出发，按已验收的联合仿真步骤运行。
- **优化研究：** [34 节点静态运行点](https://github.com/XJTU-RL/PowerZoo/blob/fdfb8680a15e76fd87b4801c1cb5f0c85172a3ac/docs/lfc-34bus-operating-point.md) · [123 节点原生储能接口](https://github.com/XJTU-RL/PowerZoo/blob/da33c66c4dec8a1580cb3fd77076591d5c097756/docs/lfc-123-damped-droop.md)。两项各自有验收边界，不替代论文复现，不代表控制收益。
- **外部时序算例：** [OEDI 123 时序潮流验证](https://github.com/XJTU-RL/PowerZoo/blob/benchmark/external-temporal-parity/docs/external-temporal-parity.md)。15 分钟时序是独立工况来源，不是论文原始 LFC 变体，尚未接入 LFC。
