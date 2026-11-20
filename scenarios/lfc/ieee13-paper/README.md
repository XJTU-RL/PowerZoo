# IEEE 13：论文 LFC 场景

- **基础电路**：[论文配电模型](../../../envs/lfc/IEEE9BusSystem/systems/13Bus/IEEE13Nodeckt_duty.dss)接 Simulink 三机九节点系统，不是标准 VVC 13Bus。
- **场景输入**：使用论文包内[负荷倍率示例](../../../envs/lfc/IEEE9BusSystem/systems/13Bus/loadshape/LoadShape1.CSV)、7 台光伏、3 台由发电机对象代理的储能。保留论文原始时钟；实验性的对齐时钟不能回填为论文原始结果。
- **实验入口**：[快速训练](../../../configs/experiments/lfc_13bus_ohasac_quickstart.yaml)及[完整训练](../../../configs/experiments/lfc_13bus_ohasac_full.yaml)；不训练的检查请见[复现说明](../../../docs/lfc-ieee9-reproduction.md)。
- **验收**：真实 MATLAB/OpenDSS 单步及配对三步通过。训练完成、论文曲线及首步满一秒响应不在验收范围内。
