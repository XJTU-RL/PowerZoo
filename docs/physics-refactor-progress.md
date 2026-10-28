# 物理重构实施记录

目标：完成[物理一致性与环境重建方案](physical-consistency-plan.md)的全部阶段，优先完成并校验SmartGrid。
基线 `fab44772`；分支 `codex/physics-refactor`；工作目录 `D:\Exps\PowerZoo-worktrees\physics-refactor`。
本记录按实际证据更新；整个重构目标尚未完成，不将阶段验收称为五环境全部可用。

## 阶段

- [ ] P0：固定反例、支持边界、训练奖励/交互契约（已完成共享奖励保护和旧实现训练阻断，后续随各任务重建补全反例）。
- [x] P1：公共物理核心，PV/Storage/SVG原生校准，SmartGrid迁移。
- [x] P2：VVC迁移与两种任务基准，见[VVC迁移记录](vvc-physics.md)。
- [x] P3：区域协同调度重建，见[区域调度记录](district-physics.md)。
- [x] P4：静态故障恢复重建，见[恢复记录](dsr-physics.md)。
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
P2已将正式VVC迁移到同一公共物理层；P4之后旧Circuits仅供已阻断的Stackelberg研究路径。
项目其他旧测试中仍存在猜测类名/过时接口；本记录只声称列明的相关回归通过，不声称全仓测试全绿。
Gradio页面未做浏览器交互验收；本轮已验证数据快照、动作覆盖、扫描配置和真实checkpoint回放引擎。


## P2实际结果

VVC13/34/123均完成固定/原生调压两种控制器×6个24小时场景、双worker、CPU HAPPO及PPO训练，
每个HAPPO CPU训练有12个独立validation回合，HAPPO与PPO各实际恢复并回放24步。
IEEE13另有CUDA HAPPO240步/10更新及12个验证回合。普通/SVG SmartGrid在公共执行抽取后重新完成
CPU/CUDA HAPPO、PPO和19/20个actor的实际恢复回放。相关整组272项回归通过、无跳过。
依赖与原生/Windows/CUDA检查通过；Plotly可选图表依赖修正单独提交为54173a39。
完整边界、物理修正和复现命令见[VVC迁移记录](vvc-physics.md)。
后续P3区域协作已完成下述阶段验收；接下来为P4静态恢复重建。

## P3实际结果

- District直接组合DSSSession，不继承SmartGrid任务；真实区域拥有各自PV/Storage/SVG。
  42母线/70 Load/5 PV/3 Storage/1 SVG，动作5/6/3维，actor各25维、全局critic187维。
- 删除虚拟Load/Generator交换与重复容量表，按实际端口核对区域P/Q、跨区线路/变压器和有符号损耗。
  固定/TOU规则×6类全天场景×96步，全部推进至86400秒；区域P/Q最大残差约3.89e-6 kW/kvar。
- 原生Storage推进一次；费用按真实kWh和价格积分，价格不被归一化抵消，末步计能量偏差。
  固定物理动作下电价翻倍使账单翻倍，碳价十倍使碳金额十倍；独立小例核对900/3600秒账单一致。
- 公共数据换算、异构动作校验/投影、回放和有单位日志共享；局部actor与全局critic观测明确区分。
  区域母线归属也绑定观测摘要。买电、卖电和净进口分别计量，团队reward/金额只累计一次。
- 新增Vsource索引0有效性与原生控制未完成的回归；修正公共会话把首个电源索引0误判不存在的问题。
  未验收设备/自动控制器、多级电容、无效/重复设备归属均拒绝，失败不伪装正常量测。
- 双worker各26步跨reset通过；CPU/CUDA HAPPO各240步/10更新、各12个独立validation回合。
  24个新保存参数文件均有限；3个局部actor实际恢复并回放24步。PPO96步/4次更新并实际恢复24步。
  replay不再依赖启动目录或猜测观测/动作尺寸，训练与回放采用相同动作投影。
- 普通/SVG SmartGrid及VVC13/34/123在公共流程改动后重跑原生/worker/短训练与实际恢复回放。
  pip check及Windows原生/CPU/CUDA运行检查通过；相关整组293项回归通过、无跳过。
- 64个旧源/配置/页面文件保留原提交字节与SHA256，目录archive/physics_refactor_district。
  旧Gradio页面和量测兜底未迁移，已归档；当前正式界面为公共训练CLI、原生快照及无头回放。

TOU规则在六个全天场景的总目标成本低于固定DER控制，但部分最低电压变差；
晚高峰最低约0.90456pu，固定控制约0.90831pu。没有因成本改善而宣称电压安全或策略收敛。
数据、动作和约束范围见专项说明；原生模型摘要75155b0a3766274642cc36df351ba1dc823e212fc5da6babbee769d8d3830ddd。
完整报告在results/validation/district/report.json，当前整组回归和各专项日志在results/validation/physics_refactor/p3_*.log。

该阶段之后的P4完成情况见下；P5与全局收尾仍待完成，整个物理/环境重构目标保持进行中。

## P4实际结果

- DSR改为真实有源静态恢复，控制组绑定原生开关与Load，不再存在随机DG位置、假PV动作和Enabled恢复率。
  两套算例为小网络与IEEE13派生备用路径，3个actor动作数分别7/3/3与7/3/5，全局观测63/295维。
- 公共物理层增加逐端子/导体开断、动态导通图、真实Vsource参考与变压器供电路径；永久故障与操作位置分开。
  单独掩码和联合径向性均校验，非法组合整帧拒绝；写入前验证且先开后合。
- 以原生交付P、相源连接、电压和供电路径容量分别计实际供电与合格恢复服务，原生秒数积分kWh。
  失电与求解失败、负荷停用与故障、带电越限与未供电分别记录。PV/Storage失源时停用注入，不作为虚构黑启动源。
- 编译连接几何固定，避免Enabled变化后停用元件NodeOrder相别重排；节点退出求解有明确in_solution状态。
  固定观测不随停用负荷改变尺寸，原生快照不补电压；整网无量测统计为null，不伪造1pu/0功率/50%SOC。
- 每套3个split×4场景×2规则×12步完整运行，并分别穷举32/64种控制状态核对静态AC末状态恢复上限。
  IEEE13测试集主干故障约100%，变压器故障88.46%，双区段75.68%，源停运0；不把不可恢复负荷计为成功。
  最大有功残差约8.72e-7/6.10e-5kW，仍采用50W门槛。
- 每套双worker跨reset、CPU/CUDA HAPPO各240步/10更新与各8个validation回合；24个保存参数文件有限。
  三个actor实际恢复回放12步，修复共享推理器离散掩码缺少batch维度的问题。
  集中PPO使用真实MultiDiscrete操作空间，各96步/4更新并恢复回放12步。
- 55个旧DSR源/配置/页面按原提交字节和摘要归档；旧8500恢复预设退出正式入口，GUI未迁移。
  数据仍由公共prepare_run准备，故障JSON使用恢复专属字段，无强制天气或重复额定值。

当前专项报告为results/validation/dsr/report.json；P4验收日志为results/validation/physics_refactor/p4_*.log。
SmartGrid普通/SVG、VVC三馈线、District及Windows原生/CPU/CUDA检查在公共层改动后再次验收。
相关整组313项回归通过、无跳过；pip check通过。包含开断几何、全网无量测、真实掩码回放和先开后合转换断言。
P5顺序定价与全局P0收尾尚未完成；Stackelberg旧实现继续阻断正式训练。
