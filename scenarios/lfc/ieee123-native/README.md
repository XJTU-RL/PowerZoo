# IEEE 123：原生储能 LFC 候选

- **基础电路**：引用现有[原生 OpenDSS 馈线](../../../node_systems/123Bus/network.dss)，不为 LFC 复制电路。[VVC 快速配置](../../../configs/experiments/vvc_123bus_smoke.yaml)共享电路，但不共享动作、数据或验收。
- **场景设计**：先用底图已有四台原生储能，独立记录设备映射、并网点功率基准、初态和秒级扰动。验证配网接口后，再验证 MATLAB/OpenDSS 联仿。
- **验收**：当前是候选，不提供已验收的 LFC 入口。[OEDI 123](https://github.com/XJTU-RL/PowerZoo/blob/benchmark/external-temporal-parity/docs/external-temporal-parity.md)是独立的时序变体；15 分钟曲线只用于选定工况，不能当作秒级 LFC 扰动。
