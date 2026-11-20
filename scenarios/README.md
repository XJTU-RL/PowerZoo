# 电路、场景与实验

| 层 | 负责什么 | 放在哪里 |
| --- | --- | --- |
| 基础电路 | 接线、相别、线路阻抗、变压器和设备额定值 | 通用 OpenDSS 模型在 [node_systems/](../node_systems/)；论文专用电路保留在各自环境内。 |
| 场景 | 引用精确电路及其版本，记录设备配置、初态、数据、时钟、扰动和验收范围 | 本目录按任务与场景归档。已有运行配置保留原路径。 |
| 实验 | 选择算法、控制基线、种子、训练预算和评价口径 | [configs/experiments/](../configs/experiments/)；结果写入独立运行目录，不反写场景。 |

**只换算法或种子不复制电路。** 同底图的布点或负荷变化优先使用小配置；变压器、线路或电压等级等基础参数发生改变时，保留单独的电路版本和来源。跨方法比较应冻结同一场景、初态与扰动。每个场景至少记录入口、外部数据来源、适用任务、已通过的验证和未验证内容。没有实际入口的候选只写说明，不放空壳训练配置。

本目录当前是可追溯索引，不是运行时注册表。已有 DSS、YAML 和训练入口保持原位。

## 算例从哪里开始

| 网络 | 已登记入口和变体 | 验收边界 |
| --- | --- | --- |
| IEEE 13 | [原生 VVC](../docs/vvc-physics.md)、[恢复版](../docs/dsr-physics.md)、[论文 LFC](lfc/ieee13-paper/README.md) | 恢复和 VVC 各自验收；LFC 是独立的缩放电路和联仿。 |
| IEEE 34 | [原生 VVC](../docs/vvc-physics.md)、[光伏 SmartGrid](../docs/smartgrid-data-runtime.md)、[三区域调度](../docs/district-physics.md)、[论文 LFC](lfc/ieee34-paper/README.md) | 不混用不同电压、设备或容量版本；LFC 校准版应单列。 |
| IEEE 123 | [原生 VVC](../docs/vvc-physics.md)、[原生储能 LFC 候选](lfc/ieee123-native/README.md) | VVC 短训练不等于 LFC 通过；OEDI 时序是独立变体。 |
| IEEE 4/37、Baran–Wu 69 | [静态引入与回归](https://github.com/XJTU-RL/PowerZoo/blob/benchmark/opendss-static-regression/docs/opendss-case-intake.md) | 独立分支静态验证。69 的同容量电容选址是对照场景，不替换底图。 |
