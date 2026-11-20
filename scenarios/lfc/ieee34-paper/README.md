# IEEE 34：论文 LFC 场景

- **基础电路**：[论文配电模型](../../../envs/lfc/IEEE9BusSystem/systems/34Bus/ieee34Mod1_duty.dss)属于三机九节点联仿配套，不等同于 VVC、光伏版或三区域调度电路。
- **场景输入**：以论文配套电路及其设备、负荷数据为基准。原始模型和后续校准模型应独立登记；不得把校准结果标为原版结果。
- **实验入口与验收**：[联仿内核](../../../envs/lfc/IEEE9BusSystem/LFC_gym.py)已在仓库中。本索引不提供已验收的 34 联仿入口；[13Bus 的命令](../../../docs/lfc-ieee9-reproduction.md)不能直接作为 34 的验收结果。
