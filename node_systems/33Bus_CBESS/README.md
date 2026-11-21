# 33Bus_CBESS —— IEEE 33-bus + CBESS/PV 覆盖层（PISTG-LDCPPO 复现算例）

来源：Huang, Yang & Wang, "Physics-Informed Graph-Based Safe RL with
Constraint-Specific Critics for Community Battery Coordination in
Distribution Network"（IEEE TSG 2026, arXiv:2610.33654）Sec. IV-A。
复现代码 [`research/pistg_ldcpo/`](../../research/pistg_ldcpo/)，
论文原文 [`papers-repro/pistg-ldcpo/`](../../papers-repro/)。

| 文件 | 内容 |
|---|---|
| `network.dss` | 33 母线辐射网（Baran-Wu 1989 参数，经 pandapower `case33bw` 生成；基准潮流对拍文献值 Vmin=0.9131@bus18、损耗 202.7 kW，偏差 <1e-3） |
| `der_overlay.dss` | 26 台 PV（DER 母线 4-18/19-22/27-33）+ 4 台 CBESS（母线 6/11/16/31，±250 kW、1000 kWh）恒 PQ Generator 覆盖层 |
| `case_meta.json` | 线路 r/x/额定、DER/CBESS 母线表——PIAM 邻接（式 10）、PIAE 物理损失（式 19/20）、支路容量三处的单一事实源 |
| `build_case.py` | 生成期脚本（需 pandapower，仅重建算例时运行，运行时无依赖） |

接入框架：`configs/systems/33Bus_CBESS.yaml`（`system.dss_folder` 指向本目录）。
运行时消费方为 `research/pistg_ldcpo`（`config.CASE_FOLDER`）与
`tests/test_pistg_repro.py`。
