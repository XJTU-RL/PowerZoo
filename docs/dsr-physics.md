# 有源静态故障恢复

DSR已重建为公共DSSSession上的真实开关/负荷恢复任务。旧随机DG母线、只更新Python的PV动作、
按Enabled计算恢复率、重复电路/负荷生成器与旧GUI已归档。正式任务不宣称支持黑启动或保护动态。

## 任务和算例

首版研究已知故障隔离后的源连接静态恢复：上级电压源供电，通过原生开关和负荷投入恢复服务。
故障资产在整个episode不可用；开关位置、资产可用、源连接、实际供电与合格服务分别记录。

| 算例 | 物理定义 | 控制组动作数 | 固定观测维数 |
| --- | --- | --- | --- |
| RestorationSmall | 含上级变压器、两条馈线、一个常开备用路径和100/50kW负荷 | 7 / 3 / 3 | 63 |
| Restoration13Bus | 基于已校准IEEE13，增加独立分段开关、备用电缆与常开开关 | 7 / 3 / 5 | 295 |

三个协作actor均看到同一全局状态并共享团队reward，不将它称为局部分布式部署。
`control_groups`绑定真实Line开关/Load ID，决定角色和动作清单；列表整体替换，组名和设备归属必须唯一。
未指定组时由原生开关/负荷发现生成一个集中调度组；控制资产数量不再手写。
每组动作0为等待，随后每个设备依次为断开/停用与闭合/投入。真实顺序写入physics.manifest.json。

IEEE13保留原线路阻抗和长度，在632—670段前加入独立串联开关，并通过633—671提供备用路径。
不能对普通线路直接设置switch=yes来冒充开关：该属性有阻抗副作用，见
[EPRI线路说明](https://opendss.epri.com/Properties14.html)。新增开关在声明switch后显式设置等效参数，
原电缆保持原LineCode。实际端子开断采用Open/Close，参见[EPRI端子Open语义](https://opendss.epri.com/Open1.html)。

13Bus保留原模型Storage，但本恢复版本没有储能调度actor；新增PV可作为源连接时的原生固定出力设备。
PV/Storage不会被当作电压参考；失去合格源连接时停用其注入，不能借grid-following PV形成虚构黑启动。
电压参考来自真实启用的Vsource，源路径纳入线路和变压器。变压器供电资格按完整绕组传播；
缺相变压器、孤岛成网、频率/暂态、保护与抢修过程不属于本版本的验证范围。

## 数据与配置入口

继续使用ConfigLoader与prepare_run；恢复数据使用任务所需的故障JSON，避免伪造天气CSV。
`data.path`指向data/restoration/small或13bus下的manifest，单位、step_seconds、来源与唯一episode ID均校验。
每个场景只声明split、永久不可用的原生设备ID和负荷倍率；容量、相别、接线与基准kW/kvar始终以DSS为准。

两套数据各有train/validation/test三组、每组四个场景。小例为单馈线A故障、单馈线B故障、
源停运与双馈线故障；IEEE例为主干区段、负荷变压器、源停运与双区段故障。
各split采用不同声明负荷倍率。这些是可追溯的合成基准，不是现场故障统计，也不据此宣称泛化能力。
训练快照保存数据路径和SHA256，后续文件改变会拒绝沿用该快照。

任务参数集中在env：服务电压边界、实际控制归属、逐Load优先级及四项实际消费的代价。
不再接受n_dg/n_pv/n_switch、obs_reserved_dim/default_voltage、假PV容量、随机断线概率等旧字段。
新增馈线时编写独立`configs/envs_cfgs/dsr_<case>.yaml`，选择模型、故障清单和控制ID即可通过
`--env dsr_<case>`加载，无需新增工厂或训练器。字典按键合并、列表整体替换；独立算例预设可避免继承旧设备优先级映射。
未验收的8500恢复预设已归档并从UI注册表移除，不能凭节点规模或旧配置声称可用。

## 物理执行、时间与可行性

reset先进行零时间的预故障原生调压整定，再施加隔离后的永久故障；恢复期间固定机械控制。
每个控制组每轮最多一个操作。掩码检查当前状态、永久不可用和单独动作的径向性；
联合动作还要整帧检查，若组合成环或含无效请求则整帧拒绝，不能由agent次序决定部分执行。
写入前完成验证，操作按先开后合下发，再求原生潮流，成功后仅一次FinishTimeStep。
数值失败直接抛错，不将其作为正常停电继续计算reward。

默认12轮、每轮300秒。动作在区间开始下发，随后新稳态保持300秒并进行供电积分，
区间末读取时钟/状态；该值是控制间隔，不代表实测开关延时或现场抢修时间。
参考规划器的多次潮流试算不推进时钟或Storage能量，也不计为实际操作；试算后恢复原状态。
固定有限时域结束，不用虚构成功阈值提前结束回合。

径向约束为母线级森林；分相调压器组视为同一连接，重叠相并联支路仍检测为环。
线路相供电按实际导体/端子开断传播；单相调压与三相变压器路径均纳入。此定义对复杂分相重构较保守，
不是任意多导体网络的保护或动态稳定判据。

每个Load同时记录：请求kW、原生实际交付kW、Enabled、所需相的源连接、电压条件、供电路径容量条件与优先级。
实际交付量最多按该场景需求计入服务；只有负荷投入、所需相有源、电压合规、源路径不过载时才计合格恢复量。
因此存在“实际交付为正但服务不合格”的状态，不能将这些功率伪造为零或直接声称恢复成功。

`unserved_kwh`为需求与实际交付之差；`weighted_service_unserved_kwh`另计优先级加权的合格服务缺口。
reward为后者除以固定原生一小时加权负荷基准的负值，再减实际操作、无效请求、带电相电压越限及支路过载代价。
需求分母不因停用负荷而减少；合法失电相不与普通带电电压越限混为同一个指标。

## 节点有效性与稳定观测

本机CAPI中，停用元件会重建NodeRef；停用元件的当前NodeOrder可能不再保留原来的相别映射。
公共会话现在固定编译时连接几何，动态读取Enabled与逐端子开断；改变接线需重新编译，不能沿用旧清单。

当某相最后一个有效元件停用时，该相可能退出求解节点表。这是未参与求解的节点，不是正常0pu量测。
DSR观测按固定编译相顺序输出，并为每相提供in_solution状态位：缺席节点仅在数组编码中使用0占位，状态位为0；
实际0pu量测的状态位仍为1。原生GridState/快照保留缺席，不填充电压，整网无量测时统计为null。
读取失败或非有限读数仍报错，不能借“inactive”掩盖求解/接口错误。
原生几何、设备/动作顺序和固定观测schema绑定checkpoint，恢复时不猜测尺寸。

## 验收

```powershell
python scripts/check_dsr.py --train --require-cuda
python -m pytest -o addopts='' -p no:faulthandler -q tests/test_dsr_physics.py tests/test_powergrid_session.py
python scripts/train.py --config configs/experiments/dsr_small_smoke.yaml
python scripts/train.py --config configs/experiments/dsr_13bus_smoke.yaml
python scripts/train_centralized.py --config configs/experiments/dsr_13bus_smoke.yaml --steps 96
```

每个算例都有三个split×四场景×等待/AC贪心两策略×12轮的完整轨迹；
小例穷举32种、IEEE例穷举64种控制状态，筛除故障重合与成环后求静态AC恢复上限。
它是末状态恢复量上限，不是完整多步奖励的动态最优策略证明。
测试同时覆盖无源、缺相、容量不足、永久故障、联合成环、原生相节点消失、整网无电压量测与集中/协作物理一致性。

两算例分别完成Windows双worker跨reset、CPU/CUDA HAPPO各240步/10更新与各8个validation回合，
3个actor实际恢复回放12步；PPO使用真实MultiDiscrete操作空间，96步/4更新后实际恢复回放12步。
共享推理器修复了离散掩码缺少batch维度的问题，训练与回放均使用真实可行掩码。
这些是运行与物理契约验收，不证明策略收敛。

报告在results/validation/dsr/report.json。测试集IEEE13中，主干故障可恢复至约100%；
负荷变压器故障的上限约88.46%，双区段故障约75.68%，源停运为0。
AC贪心达到这些末状态上限，但没有把物理不可恢复部分计为算法成功。
本轮全轨迹最大有功残差，小例约8.72e-7kW、IEEE例约6.10e-5kW，未放宽50W验收阈值。

55个旧源/配置/页面文件按原提交字节保存在archive/physics_refactor_dsr，清单记录SHA256与退役/替换状态。
正式支持公共训练CLI、原生快照、动作清单和无头回放；旧恢复GUI尚未迁移，未重新启用旧量测兜底。
