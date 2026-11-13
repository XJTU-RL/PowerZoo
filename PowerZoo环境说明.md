# PowerZoo 环境说明

本文是当前六类电力任务的导航。安装步骤见 [README](README.md)，
配置字段与覆盖顺序见 [配置说明](docs/configuration.md)。旧版直接向构造器传入
`dss_folder_path`、`reward_type: "powerzoo"` 的示例已不适用。

## 任务与入口

| 环境参数 | 任务 | 说明 |
| --- | --- | --- |
| `vvc` | 经典电压无功控制 | [VVC 物理与训练记录](docs/vvc-physics.md) |
| `smartgrid` | 模块化电网控制，含 PV、Storage、SVG 与 CMDP 奖励 | [数据和运行设计](docs/smartgrid-data-runtime.md) |
| `district_dispatch` | 区域 DER 协同调度、实际功率计量与成本 | [区域调度说明](docs/district-physics.md) |
| `dsr` / `dsr_13bus` | 有源静态配电网恢复 | [恢复任务说明](docs/dsr-physics.md) |
| `stackelberg` / `stackelberg_13bus` | 顺序定价、真实负荷响应与逐主体结算 | [顺序定价说明](docs/pricing-physics.md) |
| `lfc` | OHASAC 论文算例：OpenDSS 配电网 + Simulink 输电网联合仿真的负荷频率控制（13/34Bus） | [LFC 开发约定](docs/development.md) |

`powerzoo` 是 `vvc` 的历史别名。DSR 当前表示恢复任务，不沿用旧版需求侧响应语义。
环境注册集中在 [`utils/environment_registry.py`](utils/environment_registry.py)。

## 配置和训练

从项目或任务 worktree 根目录运行公共入口：

```bash
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml --print-config
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml --validate-config
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml
```

第一条展示最终配置，第二条校验参数，第三条实际启动训练。
通过 `--set env.voltage_min=0.94` 等完整字段路径覆盖参数；实验文件只写差异。
环境专用参数由对应 dataclass 校验，DSS 模型和运行数据由公共运行准备流程处理。
完整优先级、场景数据约定、结果重放和新增算例方法见 [配置说明](docs/configuration.md)。

同步电力任务使用独立 spawn worker；顺序定价由同一入口分派到 `sequential_ppo`。
算法登记不代表每种环境组合都经过训练验收，不能仅更换算法名称推定兼容。
配置校验也不代替实际潮流、训练和物理行为验证。

## 共享物理层与集中控制

vvc/smartgrid/dsr/stackelberg/district 五类任务使用 [`envs/powergrid/`](envs/powergrid/) 的原生 DSS context、设备与场景组件，
各自保留任务目标、奖励和交互顺序。模型容量、接线与设备额定值取自 DSS。
运行时使用模型副本，场景数据按 manifest、单位、时间和摘要校验。
LFC 为忠实论文内核，使用全局 DSS 引擎并按 worker 进程隔离（见 [LFC 开发约定](docs/development.md)）。

集中式基线用于与多智能体控制比较，不另写一套物理仿真：

- SmartGrid：[`SingleAgentVVCEnv`](envs/smartgrid/single_agent/adapter.py)，使用 `SmartGridConfig`；
  运行方法见 [集中控制说明](envs/smartgrid/single_agent/README.md)。
- 经典 VVC：[`SingleAgentVVCEnv`](envs/vvc/single_agent.py)，使用 VVC 任务配置。
- 公共集中控制训练：[`scripts/train_centralized.py`](scripts/train_centralized.py)。

两类同名适配器对应不同任务，不能相互替代。旧权重标定工具因存在占位实现已退役，
当前不提供自动 P95 奖励标定能力。

## 运行检查与历史材料

安装基线验证使用 `python scripts/check_runtime.py --train`；各任务还有
`check_vvc.py`、`check_smartgrid.py`、`check_district.py`、`check_dsr.py` 和 `check_pricing.py`，
LFC 为 `check_lfc.py`（MATLAB 联合仿真层经 `--matlab` 分层执行）。
命令参数、验收范围和已知限制以各任务说明和
[物理实施记录](docs/physics-refactor-progress.md) 为准，短训练不证明策略已收敛。

旧实现保存在 [`archive/`](archive/)，带 manifest 和内容校验。
[物理一致性方案](docs/physical-consistency-plan.md) 的原始问题章节、
[配置审计](docs/configuration-audit.md) 与 `docs/plans/` 中的历史设计保留用于追溯，
不能作为现行 API 的用法指南。开发分支、测试与提交约定见 [开发流程](docs/development.md)。
