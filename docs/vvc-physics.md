# VVC物理迁移与经典基准

本阶段在 `codex/physics-refactor` 实施，基线为SmartGrid阶段 `2bc55ab7`。
VVC与SmartGrid是独立任务，共用 `envs/powergrid/` 的场景、设备执行、动作适配、状态和回放；
VVC不继承SmartGrid，不构造SmartGrid配置，也不经过旧 `to_env_info/to_sys_info` 字典视图。

## 物理模型与时间

- IEEE13、34、123使用各自目录中的静态 `network.dss`。不在worker中生成DSS或LoadShape文件。
- `network.dss`保留原馈线接线、Load模型/额定值、变压器、RegControl、电容及电压基准；
  不执行旧daily模式或从历史CSV重复施加负荷。123的负荷定义单独保存在 `IEEE123Loads_static.dss`。
- 旧Generator电池迁移为原生Storage。因为旧模型没有可核验能量额定值，明确采用每台100kW/200kWh、
  50%初始SOC、10%保留、95%充放电效率、0空载有功损耗的研究假设；DSS是唯一容量来源。
- 123节点原来两台三相Delta电池标为2.4kV，现按接入母线的4.16kV线电压定义；单相Wye仍为2.4kV。
- 34/123原脚本引用的坐标文件不在仓库中。静态模型移除无效BusCoords指令，不伪造坐标；
  原生 `coord_defined` 保持false，展示布局不被解释为真实地理位置。
- 储能、功率限幅、绕组选择和时间推进完全复用DSSSession；NativeStorage每成功区间更新一次。
- VVC默认24个3600秒区间，即一天；可通过data.step_seconds与max_episode_steps明确改变。
  合成场景最多一天；更长外部文件必须提供连续且有单位的实际时轴。

RegControl按原生清单逐个绑定指定Transformer/winding；只设置Tap，不修改NumTaps。
IEEE13现在是2电容、3个单相调压器、1储能，共6个actor。旧运行记录的5个actor及旧观测/动作顺序
不能静默用于新环境。34与123分别有10与15个actor，设备数量不在配置中手写。

本次将错误物理表示替换为原生模型，因此它是修正后的经典VVC基准，不宣称无损复现旧论文实验。
历史DSS输入保留以便比较；旧正式环境、LoadProfile、原始读取器与旧接口测试已归档，清单包含原始SHA256。
旧Circuits及隔离在envs/legacy_vvc中的LoadProfile暂供已阻断的DSR/Stackelberg代码审查，正式VVC不导入。

## 保留的任务目标

```text
reward = voltage_reward - power_w * clip(loss_kW / fixed_base_load_kW, 0, 1)
         - cap_w * actual_cap_switches - reg_w * actual_tap_steps
         - dis_w * sum(actual_discharge_kW / rated_kW)
         - terminal_only(soc_w * sum(abs(final_SOC - initial_SOC)))
```

voltage_reward保持经典VVC的逐母线最高/最低相电压越限幅度之和。
损耗归一化基准固定为原生Load铭牌kW之和，不随当步动作、电压、价格改变。
机械成本按实际开关与原生档位数计量，终端SOC成本只在自然回合结束时计入。
SmartGrid继续使用其独立的DER/CMDP目标，不把两类任务合并为一套奖励开关。

VVC控制Capacitor、RegControl和Storage；overlays可加入固定出力PV等设备，但不因此新增PV控制actor。
需要主动PV/SVG协调时使用SmartGrid任务。所有设备读取和固定PV天气输入仍由同一公共物理层执行。
当前协作基线共享全状态观测和团队奖励，不宣称是局部观测分布式部署。

## 配置与数据

```powershell
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml
python scripts/train.py --config configs/experiments/vvc_34bus_smoke.yaml
python scripts/train.py --config configs/experiments/vvc_123bus_smoke.yaml
python scripts/train_centralized.py --config configs/experiments/vvc_13bus_smoke.yaml --steps 96
```

VVCConfig仅保留真实模型选择、时间/数据、动作档位、电压范围和经典奖励权重。
不再暴露旧env_name、scale、source_bus、load_noise、useS、use_render、record_node、dss_act及显示参数。
负荷倍率与噪声分别通过data.load_scale、data.noise设置；原生控制只作为单独的物理对照。
主训练入口的render.use_render仍属于算法运行设置，与已退役的env.use_render不是一回事。

`envs/powergrid/config.py`集中声明DataConfig与公共运行校验；scenario.py是唯一物理时序生成/读取入口。
旧SmartGrid数据模块仅作同一模块的历史导入映射，没有另一份加载实现。
缓存位于results/powergrid_data，按生成参数寻址并以进程锁保护；训练父进程冻结数据路径及摘要。
评估worker明确使用validation集合。六种合成场景标签沿用公共数据集；无PV的VVC案例不产生PV能量，
不能把其high_pv标签解释为实际光伏干预，实际PV数量与kWh在报告中明确为0。

恢复模型同时检查task、observation_schema、action_schema、master/overlays、引擎及物理摘要。
HAPPO与SB3 PPO都已实际恢复并回放。旧检查点缺少清单或包含退役参数时会拒绝；不自动猜测映射。

## 验收

```powershell
python -m pip check
python scripts/check_runtime.py --train --require-cuda
python scripts/check_vvc.py --train --require-cuda
python scripts/check_smartgrid.py --train --require-cuda
python scripts/check_smartgrid.py --config configs/experiments/smartgrid_svg_smoke.yaml --output results/validation/smartgrid_svg --train --require-cuda
```

| 案例 | 原生设备：电容/调压器/Storage | 双worker观测形状 | 实际训练及回放 |
| --- | --- | --- | --- |
| 13Bus | 2/3/1 | [2,6,54] | CPU/CUDA HAPPO各240步/10更新、各12个validation回合；PPO96步/4更新；两种模型各回放24步 |
| 34Bus | 2/6/2 | [2,10,113] | CPU HAPPO240步/10更新、12个validation回合；PPO96步/4更新；两种模型各回放24步 |
| 123Bus | 4/7/4 | [2,15,303] | 同34Bus |

每个案例还有固定控制/原生调压两种控制器各6种场景×24小时，双worker各26步跨自动reset。
报告保存网损kWh、主网交换kWh、PV kWh、相电压越限分钟、支路负载率、SOC及有功守恒残差。
三馈线最大残差小于4e-5kW，不放宽原50W验收限值。

相关回归包括原生档位不变、指定绕组、终端能量、集中/多智能体一致性、与SmartGrid相同物理动作
得到相同状态转移、恢复任务/观测绑定、图表P/Q单位和回合统计。最新整组为272项通过、无跳过。
运行输出位于results/validation/vvc及results/validation/physics_refactor，未提交Git。

对照中IEEE34最低电压约0.767pu，13/123部分工况支路负载率超过1pu；这些结果如实保留。
本任务仍是准静态VVC基准，不能据短训练声称消除了越限、满足全部热约束或获得安全保证。
机械动作视为区间内完成，不模拟设备延迟、寿命动态或暂态。
8500节点自2026-11起改用network.dss静态快照主文件（移除Generator.batt占位电池与
缺失的loadshape/BusCoords依赖），环境可构造并通过短回合冒烟（观测8562维）；
但其完整物理验收（对照组、24小时全天）尚未执行，仍不列入已验证支持矩阵。
9500节点（bal/unbal双变体）同期接入：铭牌负荷下Newton停滞（2%电压平台），
compile期初始潮流用前推回代（direct）收敛并整定分接——direct把恒功率负荷按
导纳处理，该解仅用于初始化，不进入运行期物理；运行期求解固定Newton快照，
保证Model=1恒功率负荷语义正确（曾试以direct作运行期求解，经审查指出并实验
证实其系统性改变负荷响应——0.7倍负荷下Newton与direct解平均差0.0047pu，
已回退）。随机分接/电容组合下Newton不收敛属恒功率负荷的真实电压崩溃工况
（鞍结），实测随机组合收敛率8/10且direct初值/hybrid/loadmult渐进均无法救活；
9500训练可用性需要环境级失败语义（求解失败→回合终止+大惩罚）支持，列为
后续工作。178个PVSystem建议保持不可控训练（vvc家族默认不控PV；smartgrid
需显式关闭pv_control）。
8500×{HAPPO,SHOM}已完成240步CPU冒烟训练并保存checkpoint（smoke只验证
链路，不声称策略收敛）；9500环境层验证通过（构造/reset/step/投影），训练
受上述失败语义限制。
大规模系统观测建议开启关键母线投影（observation_neighborhood，k=1..4）：可控
设备母线∪电压源母线经拓扑k跳扩展，投影时追加全网电压统计特征；9500实例k=2把
观测从9943维降至约1500维且奖励逐位不变。默认关闭投影，全母线展开行为不变。
Gradio页面未做浏览器交互验收；原生快照、图表数据、动作覆盖及训练模型回放有独立验证。

Plotly图表测试使用5.24.1；现有图表的titleside等属性在6被移除，故先固定已测试版本。
参考[官方迁移说明](https://plotly.com/python/v6-migration/)。图表测试依赖列在requirements-dev，
完整浏览器界面的可选依赖仍在requirements-viz，不作为命令行训练前提。
