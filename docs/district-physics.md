# 区域协同调度：公共物理层上的正式任务

区域任务使用同一个真实交流网络，各区域只控制自己的DER；跨区线路/变压器潮流由OpenDSS求解。
正式入口为`scripts/train.py`，配置为`configs/envs_cfgs/district_dispatch.yaml`。
原生网络为`node_systems/District_34Bus_3Zone/network.dss`；不再运行时改写旧DSS程序。

## 设备、配置和数据

当前算例为42个原生母线、70个Load（含两个EV标注的外生负荷）、5个PVSystem、3个Storage、
1个显式SVG Generator，以及原生电容/调压器。上级电源保持连接。

| 区域 | 原生控制设备 | 连续动作维数 |
| --- | --- | --- |
| upstream | pv_d0_1、pv_d0_2、ess_d0 | 5 |
| midstream | pv_d1_1、pv_d1_2、ess_d1、svg_mid | 6 |
| downstream | pv_d2_1、ess_d2 | 3 |

PV每台两个动作：`available_command`通过`(a+1)/2`换算原生可用AC有功削减比例，另一维控制双向Q。
Storage正动作放电、负动作充电；SVG正动作发无功、负动作吸无功。实际容量/能量/电流限额由公共设备层处理。
原生RegControl/CapControl作为内层自动控制；没有第二个actor覆盖它们。未验收的设备或其他自动控制器明确拒绝。

PV与Storage保留所提供DSS模型的额定值；两台Storage各100kW/500kWh，第三台60kW/300kWh，
初始50%、保留10%、充/放电效率各95%、原生待机损耗1%。这是算例参数，不是特定真实硬件的标定。
新增SVG是852母线100kVA的零有功静态双向无功等效；不宣称具有动态控制、保护或实测损耗模型。
铭牌、接线、效率、初始SOC只在DSS维护，YAML不再复制数量和容量。

配置路径为：ConfigLoader → worker一次构造DistrictDispatchConfig → 同一任务对象 → DSSSession。
市场、成本、区域归属放在任务配置的聚焦分节；实验只写差异。最终快照包含所有默认值与覆盖来源。
`regions`明确绑定实际母线和`类别.名称`控制ID；所有母线恰好归属一个区域，所有PV/Storage/SVG恰好有一个控制者。
不存在、重复或跨错区域的绑定会报错。邻区从真实支路连接推导，不再手写联络关系或交换容量。

负荷/天气沿用公共ScenarioDataset，数据种子与训练种子分离，训练/validation/test使用独立文件。
三个运行任务共用`apply_operating_row()`：原生Load的kW/kvar基准乘场景倍率；PV接收kW/m²辐照和组件温度°C。
未知负荷名称的稳定分组是合成假设，可用`data.load_groups`显式指定。
EV没有到离站/服务电量数据，按外生Load处理；不存在可控EV服务动作。
市场首版采用配置中的公开24小时TOU阶梯序列；不是另一套隐藏的负荷生成器，也不伪称真实市场历史数据。

## 物理时间与区域账本

默认96步×900秒为完整日。smoke实验为24步×3600秒，既有训练采样长度24保持独立。
reset只产生初始潮流；其他设备先下发，再作原生PV可用功率探测，最后执行PV设定并进行带原生控制的潮流。
所有额外求解均为零时间，成功后仅一次FinishTimeStep；Storage状态直接读原生kWhStored/%stored。
无效输入整帧拒绝；未完成的原生控制、未收敛或失效会话不能继续发布正常状态。

电压/PQ代表刚完成的恒定区间，原生时钟和Storage能量为区间末。info记录本区间开始时使用的价格；
下一观测中的公开价格对应下一个决策时刻，不会将刚完成区间按新价格重新计费。
天气、负荷和电压观测保留已实现的当前/最近区间量测，没有未声明的未来天气预测。

逐区读取原生Load、DER、电源、电容和每个支路端口；保留kW/kvar符号。
跨区线路L15及跨区变压器XFM1都计入边界功率。每个区域满足：

```text
边界进口 + 本地电源注入 + DER注入 + 并联电容注入
    - 实际负荷 - 区内支路损耗 ≈ 0
```

所有区域边界进口之和为跨区支路损耗的负值；区内损耗之和加跨区损耗等于原生全网损耗。
无功支路“损耗”可为负，反映充电效应，不能取绝对值。量测失败不回填1pu/0kW/50%SOC。
info保存归一化动作、请求、限幅后设定、原生实际读数和区域账本。

## 目标与信息结构

`reward = -total_cost_yuan / costs.scale_yuan`，固定尺度默认100元，不随本轮电价或负荷变化。
账单与人为代理代价分别记录：

| 分量 | 实际计算 |
| --- | --- |
| grid_bill_yuan | 进口kWh×买价 − 出口kWh×卖价 |
| carbon_cost_yuan | 进口kWh×碳强度kg/kWh×碳价元/kg |
| battery_wear_cost_yuan | 原生储能能量变化绝对值kWh×代理单价 |
| curtailment_cost_yuan | 原生可用AC功率与实际PV输出之差积分×代理单价 |
| voltage_cost_yuan | 实际相电压越限平方均值×小时×系数 |
| thermal_cost_yuan | 有额定值支路的过载比例平方均值×小时×系数 |
| terminal_energy_cost_yuan | 仅末步：各Storage相对初始能量的绝对偏差×单价 |

购电量已经包含网损，不另加一笔重复的网损电费。弃光参考为当前区间其他设备设定下的零Q原生可用AC探测。
成本系数、TOU、售电比例及碳强度均为显式实验假设；磨损系数不是电池寿命模型，越限代价也不是安全保证。
终端能量代价避免免费耗尽初始电量，但不是期末SOC的硬等式约束。

三名区域actor共享相同团队reward，critic使用全局状态；各actor只看本区电压统计、负荷、设备实际P/Q/可用量/SOC、
本区净进口、公开价格/时间/进度，以及启用时的一跳邻区电压和净进口摘要。
当前actor统一padding到25维，全局critic为187维，包含原生电容位置和调压Tap状态。
padding只用于空间对齐，没有虚构传感器量测。
`use_neighbor_obs=false`确实删除邻区信息并改变观测schema，不改变控制设备和动作维数。
动作顺序、局部/全局观测语义、区域母线归属、原生模型和引擎摘要绑定checkpoint。

HAPPO与集中PPO共用同一核心。集中PPO只改变控制组织，动作相同时物理状态和成本相同。
正式回放按真实局部观测与异构动作空间恢复actor；高斯策略动作在训练和回放都按同一空间投影。

## 验收与实际范围

```powershell
python -m pip check
python scripts/check_runtime.py --train --require-cuda
python scripts/check_district.py --train --require-cuda
python -m pytest -o addopts='' -p no:faulthandler -q tests/test_district_physics.py tests/test_powergrid_session.py
python scripts/train.py --config configs/experiments/district_dispatch_smoke.yaml
python scripts/train.py --env district_dispatch --algo happo --set env.market.base_price_yuan_kwh=0.7 --explain env.market.base_price_yuan_kwh
```

结果写入`results/validation/district/report.json`，原始训练日志也在同目录。
全天检查包含固定DER控制（原生调压保持启用）与TOU/电压规则两种控制器×六场景×96步，
逐步检查全网有功、区域有功/无功账本及86400秒原生时钟。
另验收双worker跨reset、CPU/CUDA各240步10次更新与各12个validation回合、
三个actor的实际24步恢复回放，以及PPO96步/4次更新与24步恢复。
短训练证明端到端链路和数值有效性，不证明收敛、最优或全部电压合规。

首轮全天对照中TOU规则降低了六类场景总目标成本，但部分最低电压下降；
晚高峰最低约0.905pu，固定控制约0.908pu。不能只凭总成本改善认定规则安全。
报告保留原始账单、各代理代价、购售电量、峰值、电压越限、热负载率及最终能量，不调整数据掩盖越限。

64个旧源/配置/页面文件按原始提交字节归档于`archive/physics_refactor_district/`，manifest记录原路径和SHA256。
退役虚拟交换、旧量测兜底、重复配置投影及固定动作标签。旧Gradio区域页面依赖这些假设，已随源文件归档；
当前正式支持训练、公共原生快照、绑定清单的手动动作和无头回放，不宣称原八页GUI已迁移。
可视化可从公共快照重建，不能重新启用旧数据提取器来填充正常量测。
