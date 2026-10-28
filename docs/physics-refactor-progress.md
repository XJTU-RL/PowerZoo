# 物理重构实施记录

目标：完成[物理一致性与环境重建方案](physical-consistency-plan.md)的全部阶段，优先完成并校验SmartGrid。
基线 `fab44772`；分支 `codex/physics-refactor`；工作目录 `D:\Exps\PowerZoo-worktrees\physics-refactor`。
本记录按实际证据更新；整个重构目标尚未完成，不将阶段验收称为五环境全部可用。

## 阶段

- [ ] P0：固定反例、支持边界、训练奖励/交互契约（已完成共享奖励保护和旧实现训练阻断，后续随各任务重建补全反例）。
- [x] P1：公共物理核心，PV/Storage/SVG原生校准，SmartGrid迁移。
- [x] P2：VVC迁移与两种任务基准，见[VVC迁移记录](vvc-physics.md)。
- [ ] P3：区域协同调度重建。
- [ ] P4：静态故障恢复重建。
- [ ] P5：顺序定价博弈重建。

## SmartGrid验收

- [x] 静态模型入口，不依赖运行时正则重写DSS程序。
- [x] 独立context、模型只读、输入路径与设备绑定校验。
- [x] 原生设备清单、稳定动作顺序、模型/动作清单记录。
- [x] 原生PV的可用功率削减、Q双向控制、原生能力和夜间策略。
- [x] Storage原生能量推进一次，效率/待机/边界/dt/reset验证。
- [x] SVG静态等效、Q方向、容量与低电压电流边界。
- [x] 请求/执行/实际量测可追踪，功率守恒与相别正确。
- [x] 时钟、数据、观测、奖励和有限时域终止语义。
- [x] 读取/求解错误不伪装为正常量测，不推进失败区间的能量。
- [x] 单/多智能体相同物理动作结果一致。
- [x] 六类场景完整运行、全日场景与独立验证集。
- [x] Windows双worker、CPU/CUDA HAPPO与SB3 PPO实际训练。
- [x] 原生调压/固定控制对照及原始物理指标。
- [x] 相关回归、pip check、check_runtime和文档更新。

## 已实现的边界

P1首先建立三个主要物理模块：`envs/powergrid/session.py` 管理context、复制模型、时钟、求解与量测；
`devices.py`负责实际原生设备发现、物理指令及能力限幅；`state.py`声明有单位的状态。
P2在第二个实际任务接入后提取共享场景、设备运行流程、动作适配、观测清单和回放模块；
不引入通用插件框架。
SmartGrid只选择模型/数据、转换动作、组织观测与奖励；其他任务不会继承SmartGrid任务类。

- 原生模型以 `node_systems/34Bus_PV/network.dss` 为唯一静态定义，旧duty入口重定向到它。
  PV方案单独编译；可选SVG算例使用 `network_with_svg.dss`。
- 容量、相别、接线、电压、效率、曲线、SOC以DSS为准；YAML仅指定实际 `Generator.svg834` 等控制绑定。
- 负荷为原生Load的kW/kvar基准乘时序倍率；原生端口功率保持kW/kvar，原生Losses只转换一次。
- 每个区间先整帧校验/执行，再求潮流，成功后调用一次原生 `FinishTimeStep()`。
  PV可用功率探测和额外求解均不推进能量。Python不再重复积分Storage。
- PV采用原生Pmpp/效率/温度求可用AC功率，再用 `%Pmpp` 削减上限；不改铭牌Pmpp。
  当前三个PV方案的原生 `VarFollowInverter=No` 允许夜间供无功；这是本算例的静态能力假设，
  已在零辐照、正负Q动作下验证，不能外推为所有真实PV逆变器均具备该能力。
- SVG采用零有功Model=1、明确wye接线的理想静态无功等效，并根据实测最大相电流迭代限幅。
  本机CAPI 0.14.5的零P Model=7探针存在Q方向反转，故不采用；不宣称实现PLL或变流器动态。
- 观测记录实际相节点和实际端口P/Q；零电压保留，失败读数或求解直接报错。
  线路按原生NormalAmps计算负载率，变压器按每绕组kVA/kV换算相电流额定值。
- 电压/功率代表本次准静态恒定区间，时钟和SOC为收尾后的区间末状态。reset只作零时间初始潮流。
- 网损奖励使用固定Load铭牌之和作基准；动作平滑比较实际执行设定，包含Storage和SVG。
- 训练写出物理模型文件摘要、master/overlays、引擎版本、dt、原生设备信息和稳定动作清单。
  恢复/回放先核对清单，模型选择、原生版本或动作绑定不符即拒绝。
- 回放从 `config.resolved.yaml` 和真实的Discrete/Box空间恢复每个actor，不再猜64维观测/10维动作。
  手动覆盖同样按正式设备ID和空间校验；统计kWh乘以真实dt，排除step=0的初始快照。
- 修复Windows日志中strftime不支持 `%f` 导致文件日志为空的问题，改用logging毫秒字段。

## 可复现命令与结果

在本worktree，使用 `D:\Development\CondaEnvs\powerzoo-win\python.exe`，设置 `PYTHONUTF8=1`。

```powershell
python -m pip check
python scripts/check_runtime.py --train --require-cuda
python scripts/check_smartgrid.py --train --require-cuda
python scripts/check_smartgrid.py --config configs/experiments/smartgrid_svg_smoke.yaml --output results/validation/smartgrid_svg --train --require-cuda
```

| 检查 | 实测证据 |
| --- | --- |
| pip check | No broken requirements found |
| Windows运行 | POWERZOO_RUNTIME_OK；双DSS接口、VVC单/双worker、CPU/CUDA HAPPO更新 |
| SmartGrid普通/SVG | 各有固定控制、原生调压两种控制器×6场景×360步，全部求解成功 |
| 守恒 | 对照最大有功残差约 `7.2e-6 kW`；采用原生 `tolerance=1e-8`，未放宽50W验收限值 |
| 独立小电路 | Storage充/放电dt=60/300/900秒、额定/能量边界、0/1%待机、额外求解、reset与失败 |
| SVG | 正负Q、源电压1/0.7/0.4pu，实测相电流不越额定 |
| 全日 | 1440步，对应原生86400秒；空闲SOC与reset正确 |
| 双worker | 各26步跨自动reset，普通观测[2,19,153]，SVG观测[2,20,155] |
| CPU HAPPO | 两算例各240步/10更新，独立validation各12回合 |
| CUDA HAPPO | 两算例各240步/10更新 |
| SB3 PPO | 两算例各96步/4更新，集中适配器共享同一物理内核 |
| checkpoint | 两算例分别检查88/92个有限参数文件；普通/SVG新训练19/20个actor分别实际恢复、回放24步 |
| 回归 | 配置/入口/数据/原生物理/Windows/SmartGrid活动测试整组262项通过、无跳过（包括原生收尾失败后的状态失效） |

完整结果在 `results/validation/{smartgrid,smartgrid_svg}/report.json`，日志在
`results/validation/physics_refactor/`，这些运行输出不提交Git。固定/原生控制输出网损kWh、
主网交换kWh、PV kWh、累计相电压越限分钟、最大支路负载率、守恒残差和最终SOC。
这些是可运行性和物理一致性检查，不是算法收敛或全部场景可行性的证明。

原生调压对照仍有晚高峰最低约0.918pu、夜间约0.936pu；固定控制更差。
不调整负荷或过滤母线来掩盖此结果。需要后续通过设备能力/容量设计和策略评价判断可行边界。
原生恒阻抗/ZIP等负荷模型会使调压改变实际负荷，因此同时比较购电量和电压，不能只看网损。

## 旧路径与后续

17个不再使用的SmartGrid物理/数据源文件及旧测试原样归档在
`archive/physics_refactor_smartgrid/`，清单记录原路径和SHA256；新的包级Env导出指向正式运行内核。
旧测试依赖退役接口或存在空断言/大量skip；替代测试直接验证原生功率、SOC、时钟和设备状态。
活动奖励/日志测试重新按实际API和独立公式断言，不把构造不存在的类当作覆盖。

P0保护提交：`af98d86c`拒绝EP非团队奖励；`48237b08`在正式准备阶段阻断DSR、District、Stackelberg旧物理训练。
配置预览和只读审查仍可使用；每个任务完成重建与实际验收后，才移除相应阻断。
P2已将正式VVC迁移到同一公共物理层；旧Circuits暂仅供已阻断的DSR/Stackelberg研究路径。
项目其他旧测试中仍存在猜测类名/过时接口；本记录只声称列明的相关回归通过，不声称全仓测试全绿。
Gradio页面未做浏览器交互验收；本轮已验证数据快照、动作覆盖、扫描配置和真实checkpoint回放引擎。


## P2实际结果

VVC13/34/123均完成固定/原生调压两种控制器×6个24小时场景、双worker、CPU HAPPO及PPO训练，
每个HAPPO CPU训练有12个独立validation回合，HAPPO与PPO各实际恢复并回放24步。
IEEE13另有CUDA HAPPO240步/10更新及12个验证回合。普通/SVG SmartGrid在公共执行抽取后重新完成
CPU/CUDA HAPPO、PPO和19/20个actor的实际恢复回放。相关整组272项回归通过、无跳过。
依赖与原生/Windows/CUDA检查通过；Plotly可选图表依赖修正单独提交为54173a39。
完整边界、物理修正和复现命令见[VVC迁移记录](vvc-physics.md)。
下一阶段为P3区域协作：替换虚拟交换、实现真实区域计量、统一能量时钟和实际货币目标。
