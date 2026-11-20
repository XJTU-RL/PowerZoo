# LFC 论文复现入口：OHASAC

论文：[Order-based Heterogeneous Agents Reinforcement Learning Method with the Coordination of Distribution Network and Transmission Network for Load Frequency Control](https://doi.org/10.1109/TPWRS.2025.3598951)。

这是原论文实现的入口，不包含优化版 LFC 或 SHOM 电压无功控制实验。完整命令、数据来源和三步对照见 [已验收的联合仿真指南（#64）](../../docs/lfc-ieee9-reproduction.md)。

| 部分 | 模型与验证状态 |
| --- | --- |
| 输电主系统 | [Simulink IEEE 三机九节点](../../envs/lfc/IEEE9BusSystem/IEEE9BusSystem.slx)。三台同步机接在 Bus 1、2、3。这是输电侧，不是第二条配电馈线。5 个训练和评估模型文件与[原论文实现固定提交](https://github.com/rokey0001/MARL_LFC/tree/63f466186b0649896bda1013d296df2dcf682017/harl/envs/LFC/IEEE9BusSystem)逐字节一致。 |
| 原论文 13 节点配电侧 | [OpenDSS duty 电路](../../envs/lfc/IEEE9BusSystem/systems/13Bus/IEEE13Nodeckt_duty.dss)，7 台光伏、3 台按论文内核更新能量的储能，与 VVC 的 IEEE13 不同。已验证联合仿真一步和三步动作对照。负荷是随包归一化倍率，未核实为实测。 |
| 随包 34 节点配电侧 | [原论文目录](../../envs/lfc/IEEE9BusSystem/systems/34Bus)有模型和负荷数据，但本次未验收其 MATLAB 联合仿真。不要用 13 节点结果或优化版 34 节点诊断替代。 |

## 先跑联合仿真检查

已验证快照为 PowerZoo [`de8ce3c7`](https://github.com/XJTU-RL/PowerZoo/tree/de8ce3c7e08306f526ee4fb6f0fc8499fcbd33af)，使用 Linux、MATLAB R2025b、配套 MATLAB Python Engine、Simulink/Simscape Electrical 和 DSS-Python 0.15.7。在有许可证和项目 Python 依赖的环境里，从仓库根目录运行：

```bash
xvfb-run -a python scripts/check_lfc.py --matlab
```

预期看到 `[PASS] matlab` 和 `LFC_CHECK_RESULT=OK`。已验收的一步返回有限团队奖励（约 −1.0687）。三步对照在配电接口核对了 P/Q，并在第二步观察到频率及同步机状态差异。原实现的 reset 和第一次 step 共用 1 s 暂停目标，**第一次动作不能当作完整一秒的响应**。不加 `--matlab` 的检查不能替代联合仿真；这也不证明输配双向闭环或控制收益。

要启动论文训练，见 [快速档](../../configs/experiments/lfc_13bus_ohasac_quickstart.yaml) 和 [完整档](../../configs/experiments/lfc_13bus_ohasac_full.yaml)。15,000 步训练产物、论文曲线和算法收敛不在已验收范围内。
