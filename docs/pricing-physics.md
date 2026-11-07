# 顺序需求响应与逐主体训练

正式算例为`stackelberg`（可手算小网络）和`stackelberg_13bus`（IEEE13真实负荷分组）。
使用公共`ConfigLoader → prepare_run → DSSSession`，任务代码不继承SmartGrid。
旧34/123预设、占位DER、重复ESS状态、异步包装器和旧GUI已归档，不能据此声称支持这些能力。

## 一个工程区间的顺序

1. 应用本轮场景行，零时间求解完整基准需求；这是声明的已知需求基线，不计成交电量。
2. 运营者策略根据当前需求、批发电价、原生电气摘要和上一轮动作公布一个零售价格。
3. `publish_price()`构造含**本轮新价格**的消费者观测，随后才调用各消费者策略。
4. 消费者分别选负荷比例，写入其真实Load的kW/kvar；求解实际潮流，计量交付功率。
5. 分别结算运营者和消费者效用；`FinishTimeStep()`仅执行一次。价格广播和反事实求解不推进时间或SOC。

```mermaid
flowchart LR
    O[当前需求与原生基线] --> L[领导者策略：公布价格]
    L --> F[新价格进入观测后采样跟随者]
    F --> P[真实Load P/Q → 潮流 → 一个dt]
    P --> R[实际kWh与逐主体元结算]
    R --> B[各主体独立buffer / 价值 / PPO更新]
```

同步联合`step()`和同步MARL worker明确拒绝此任务。`happo`、`mappo`及旧`sn_mappo`不能通过正式入口
替代顺序采样器。公共EP团队奖励路径仍拒绝不相同的主体收益，避免再次丢失用户目标。

## 当前模型与单位

| 项目 | 明确约定 |
| --- | --- |
| 物理量 | DSS原生kW/kvar/kV/kWh；线路与变压器实际相电流及热限额 |
| 角色 | 1个运营者、2个消费者；每个真实Load恰好归属一个消费者 |
| 价格动作 | 默认0.2/0.4/0.6/0.8/1.0元/kWh，直接配置有单位列表 |
| 响应动作 | 默认完整需求的0.6/0.8/1.0倍；按原DSS功率因数同比缩放kW/kvar |
| 时间 | 默认24×3600秒；批发电价是24个小时价格，当前场景行的时刻决定计费小时 |
| 设备 | 首版只研究主网连接的负荷削减；PV/Generator售电明确拒绝，不能伪装成用户DER动作 |
| 固定设备 | 回合开始由原生控制确定机械状态，之后冻结；已有原生Storage固定空闲，不作为博弈动作 |
| 数据 | 共用运行场景manifest+CSV、三组独立split；无PV时天气列不参与效用，非默认irradiance_scale拒绝 |
| 观测 | leader13维、每个follower16维；是公开的压缩信息，未声称局部隐私或完整Markov状态 |

小例为两个三相恒PQ负荷，100/50kW，经过真实变压器与线路供电。
IEEE13算例15个真实Load按显式清单分给两个用户，保留原生接线、负荷模型与空闲Storage。
当前IEEE13合成负荷倍率0.5是公开算例假设，并非实测标定；额定值不在任务YAML复制。
缺相/无源、负负荷、遗漏/重复归属均拒绝，故障恢复应选择DSR任务。

设实际交付功率`P_i`、同区间完整需求的原生交付基线`B_i`，区间小时数`dt`，零售价`p`：

- 用户电费：`bill_i = p * P_i * dt`。
- 削减不适：`discomfort_i = beta_i/2 * max(B_i-P_i,0)^2 / max(B_i,1e-9) * dt`。
- 用户效用：`U_i = -bill_i - discomfort_i`。
- 批发成本：`C = max(grid_import_kw,0) * wholesale_price * dt`，包含线路/变压器实际损耗和固定设备吸收。
- 运营者效用：`U_0 = sum(bill_i) - C - network_cost`。
- 网络代理成本：相电压超限幅度之和加支路热过载幅度之和，乘`network_cost_yuan_per_hour * dt`。

因此`sum(U) = -C - network_cost - sum(discomfort)`；消费者支付和运营者收入是同一笔转移。
无本轮价格归一化，不以主网进口反算用户负荷。实际潮流下恒阻抗/ZIP负荷的交付可能偏离请求值，结算始终用实际端口P。
`beta`是合成的元/kWh不适参数，网络成本也是公开代理项；它们不是实测偏好或法定罚价。
当前没有负荷移峰回补、参与退出、需求弹性标定、双向售电、用户储能或跨期合同，因此不能用本例给出现实最优零售价。

## 精确响应与偏离评价

`exact_response_table(price)`对本轮离散联合响应逐一进行零时间AC求解，再检查每个用户的全部单边动作。
无改善超过1e-7元的联合动作才列为纯策略响应均衡；若不存在，明确失败。
多个响应均衡时按动作字典序选择；`exact_reference()`为**每个价格重求响应**，再选择运营者效用最大者，收益相同时取较低价格。
上限4096种联合响应仅约束精确枚举器，不伪造大网络近似解。

恒PQ小例的用户响应可独立用`argmin_f(price*f + beta/2*(1-f)^2)`核对；价格从低到高，响应确实改变。
所有反事实都恢复负荷状态，不产生虚构电量。该参考只解决一个工程区间，固定机械状态、有限价格/负荷网格及上述均衡选择下的阶段博弈；不是连续动作或全日动态最优解。

checkpoint回放另外报告用户本轮单边偏离增益，以及**重新调用已训练用户策略**后的领导者变价增益。
这些是阶段诊断，不能把累计reward差、两个主体效用差或一期偏离指标称为动态Nash gap。
短训练仍测到正偏离收益，未宣称均衡收敛。

## 训练、配置与恢复

```powershell
python scripts/train.py --config configs/experiments/stackelberg_smoke.yaml --validate-config
python scripts/train.py --config configs/experiments/stackelberg_smoke.yaml
python scripts/train.py --config configs/experiments/stackelberg_13bus_smoke.yaml --set device.cuda=true
python scripts/check_pricing.py --train --require-cuda
```

`sequential_ppo`复用已安装SB3 PPO；每个主体拥有各自策略、价值网络、优化器、GAE和rollout buffer。
领导者存储公布价格前的观测，用户存储看到本轮价格后的观测；各自收益写入各自buffer。
首版按完整有限回合采样，单训练/验证采样进程；不支持RNN或共享参数等未实现开关。
CPU/CUDA都已实际训练，Windows双独立进程各跨reset运行48步，不将此称为训练器支持并行采样。

配置继续使用八节结构，算法YAML只声明本训练器的实际消费者。
新算例写独立`configs/envs_cfgs/stackelberg_<case>.yaml`，选择模型、消费者Load清单、数据、价格和偏好。
无需新建配置加载器或按设备计数生成配置。数组整体替换，映射逐键合并；空映射不会清除父值。

每次运行保存`intervals.jsonl`原始结算、`summary.json`逐主体收益，以及checkpoint下的
`config.resolved.yaml`与来源摘要、`physics_manifest.json`、三个`agent_i.zip`。
manifest绑定模型/引擎/设备、阶段契约、动作/观测尺寸和经济假设；不匹配时拒绝恢复。
无头恢复使用同一阶段调度器，例如：

```python
from envs.stackelberg.training import load_checkpoint, collect_episode
env, models = load_checkpoint(r"<运行目录>\checkpoint", split="test")
try:
    utilities_yuan, intervals = collect_episode(env, models, training=False)
finally:
    env.close()
    for model in models:
        model.get_env().close()
```

## 实际验收

两算例各3个split×6场景×2规则×24步，36条完整轨迹；每个split/场景首区间另做5价格×9响应穷举。
全天对照为固定0.6元价格下的完整服务和精确用户响应，不称为全日最优。
最大功率残差约4.18e-8/6.02e-5kW；仍使用固定50W验收门槛。
每套CPU/CUDA各240工程步/10更新、4个独立validation回合，三个实际策略分别恢复并回放24步。
实测报告`results/validation/pricing/report.json`，日志`results/validation/physics_refactor/p5_pricing.log`。
旧源文件、配置和最后的全局VVC物理/负荷生成器按原提交字节与SHA256归档在`archive/physics_refactor_stackelberg/`。
