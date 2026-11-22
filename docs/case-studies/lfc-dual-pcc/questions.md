# 双 PCC 接力与电压受限补位

## 已验证

起点是公开 main `1a4a4e05357781d1d2de0e2e1ed1917e1495071e`。沿用 #67 的 IEEE34 私有工作点、公共馈线和 Simscape 模型。静态接线指向 Bus 8 / Sensor7 和 Bus 6 / Sensor2。Bus 6 的 90 MW / 30 Mvar 原固定负荷块已停用，动态块承担该基准。Bus 8 保留 100 MW / 35 Mvar 的独立固定负荷。原模型功率基准为 100 MVA。

## 在做

阶段 0 在两个独立原生 DSS 上下文中使用相同 IEEE34 拓扑与初始化。每个输电端口只写 `原支路基准 + 当前可用的实测馈线进口 − 该馈线初始进口`。支路实行 W/var 与 kW/kvar 的 1000 倍换算，P、Q 正号均为进口。能量按 kW × 秒 / 3600 计 kWh。配网各自保留固定电压源；主网提供统一 Bus 8 实测频率。没有两个独立频率测点或输电电压回写配网的假设。

预先固定两个中性批次、A 单动、B 单动和同时动作，均独立重置。每次先读严格早于暂停的原生频点。双端写入来自已完成且额外传输 0.1 s 后到达的报告，两处均确认读回才推进同一个主网求解器。设备动作作用在接下来的一秒，报告在区间结束后才产生。评分按实际写入时刻和原生 0.2 s 频点配对，两个电气步是写入护栏。见同研究目录的 `stage0.json`。

## 待验证问题与停止条件

1. 两只独立物理传感器是否按端口与符号响应？是否有重复负荷、异端串写、失配、潮流失败或使用未来报告？未过门槛即停。
2. 两条同拓扑、同总资源馈线的快慢与能量差异，能否由同信息规则或学习策略减轻接力缺口和二次频跌？换地点或负荷后，是否能单独核实逐相电压造成的短交付？需要冻结开发/评估事件、基线、保护与预算，才进入对比。
3. 拟实现慢层主网与两配网代表分配 PCC 增量、快层异质资源跟踪及实测报告回传；与固定份额、下垂和同信息规则进行公平比较，分别消融同频、实测反馈及功率–时长信息。训练、冷迁移、重训独立记账。阶段 1 可靠后才能启用 IEEE34+IEEE123 留出组。

## 可复跑入口与预算

南洋 HARL Python 3.10.14、DSS-Python 0.15.7、MATLAB Engine 25.2.2，模型运行于 R2025b。每臂使用一个新的 MATLAB 进程与两个独立 DSS 上下文。指定未存在的输出，保留失败记录。

```sh
cd "$(git rev-parse --show-toplevel)" || exit
export PATH="$HOME/matlab_2025b/bin:$PATH" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py="$HOME/miniconda3/envs/HARL/bin/python"
ref="${LFC34_REFERENCE:?需要已验收的 IEEE34 私有工作点}"
out=$(mktemp -d) || exit
for arm in zero_a zero_b a_only b_only joint; do
  timeout --kill-after=15s 420s xvfb-run -a "$py" -m research.lfc_dual_pcc.stage0 "$arm" "$ref" "$out/$arm.jsonl" || exit
  sleep 10
done
"$py" -m research.lfc_dual_pcc.score_stage0 "$out" "$out/summary.json"
```

`f25b7f6e` 首个中性臂通过 18 帧原生双端口核对。额外通信延时 0.1 s 在 1 s 边界上的后果是使用两帧前开始、上一帧之前结束的数据。第二个中性臂第一次启动在 MATLAB 授权服务关闭时失败，失败发生在求解前；原件保留，恢复时仅链接已验收的首臂，另写新文件。设备存在性核查显示每条 IEEE34 有两台原生储能，额定各 100 kW / 200 kWh，原工作点将它们禁用。阶段 0 使用 890 柔性负荷作为端口标尺。后续储能与 PV 要独立实测并建立新工作点。

阶段 0 五臂串行，单臂超时 420 s。阶段 1 拟先做两类工况的 DSS 资源门，再用少量完整引擎运行冻结同信息规则对照；每个策略/工况至少要计一次独立初始化。按阶段 0 实测时间更新训练预算，原生训练仅在物理机制和规则门通过后开启。区分实际授权/启动失败、保护限发和物理不收敛。


### 阶段 0 动态验收

冻结的五臂在南洋分别独立完成 18 帧，两个中性重复的 P/Q 与频率逐点差均为零。A 与 B 单独有功动作时，首次稳态窗实测两口 P 增量分别为 [+9.020, +0.089] 和 [−0.040, −9.039] kW。两个端口共同动作也通过了逐帧增量判据。两只物理传感器的首个严格后写样本在 1/60 s 后，尾窗首个参与均值的样本在写后约 0.7833 s。五臂的原始时间戳均单独核验。停机时最后一帧末端少一个 1/60 s 电气步：全长 DSS/施加量能量账与实际传感器可观测时段积分分开记录；未外推最后一个样点。新实验将在计分时段之后多留护栏帧。

## Case A 最小归因试验

每个慢窗为两个 1 s 快步。主网策略依据原生 Bus 8 频差、严格已到达的两端报告与上次报价做分配。两个配网代表本次报价在下一慢窗才能消费。两套本地策略从六个物理动作中选择原生储能放电和 PV 增发，在线只用本地状态、实际收到的目标与安全屏蔽。两条馈线各激活两台既有 100 kW / 200 kWh 储能，增加一台 8 kW / 10 kVA 研究 PV，静置调度 4 kW。A 的初始 SOC 为 10.0018%，B 为 50%，两者储能保留线均为 10%。这是刻意设置的低可用能量机制。A/B 上升限速为 10/4 kW 每秒，下降限速均为 10 kW 每秒，各方法一致。

反事实只在采集结束后重建同一初态、点位、外生输入及已执行前缀，穷举两个快步所有允许的离散动作路径。各路径取两步中较小的实际 PCC 交付，再取最大值 C；获胜路径另做两个完整原生步并核对。只认证此有限动作表和固定馈线源电压。y 为相同两个实际 PCC 快步交付的较小值，按报告编号对齐到后续两次实际输电写入。当 R≥0 时，R_c=min(R,max(0,C))，上层原超额 R−R_c，下层欠交 max(0,R_c−y)，过交 max(0,y−R_c) 先抵扣上层超额；超过 R 的部分单列。反向交付单记。不能重建或证实的窗不使用归因标签。下层动作与主网分配的学习信号对齐它们后续实际回传窗，配网代表学习信号再后移到报价被使用的慢窗。两层均保留相同频率项；关闭归因时等分相同的缺口罚量。底层为 PyTorch 类别策略梯度和价值基线。归因有无共享初始化、种子、架构、动作、安全约束和一次更新预算。

以下命令从研究 checkout 执行，dry run 使用合成观测，只验证分发、梯度、存取模型与独立评估。原生训练显式要求五臂门和新资源门。每个 Case A 回合 启动一个 MATLAB 引擎与两个实际 DSS 馈线，认证仅用独立 DSS 上下文。候选预算为每臂一次原生训练回合及一次独立评估，不做超参数搜索。验证事件改变幅值与起止时刻。

```sh
py="$HOME/miniconda3/envs/HARL/bin/python"
o=$(mktemp -d) || exit
"$py" -m unittest research.lfc_dual_pcc.test_credit
"$py" -m research.lfc_dual_pcc.train --dry-run --out "$o/dry"
"$py" -m research.lfc_dual_pcc.evaluate --dry-run --models "$o/dry" --arm attribution_on --out "$o/dry-eval"
"$py" -m research.lfc_dual_pcc.commission --stage0 "$LFC_STAGE0" --reference "$LFC34_REFERENCE" --out "$o/mechanism.json"
set -- --stage0 "$LFC_STAGE0" --reference "$LFC34_REFERENCE" --mechanism "$o/mechanism.json"
timeout 480s xvfb-run -a "$py" -m research.lfc_dual_pcc.evaluate "$@" --development --arm none --out "$o/dev-none"
timeout 480s xvfb-run -a "$py" -m research.lfc_dual_pcc.evaluate "$@" --development --arm rule --out "$o/dev-rule"
"$py" -m research.lfc_dual_pcc.compare --reference "$o/dev-none/none.jsonl" --out "$o/dev-rule-check.json" "$o/dev-rule/rule.jsonl"
timeout 2400s xvfb-run -a "$py" -m research.lfc_dual_pcc.train "$@" --out "$o/train"
"$py" -m research.lfc_dual_pcc.compare --training-pair --reference "$o/dev-none/none.jsonl" --out "$o/train-check.json" "$o/train/attribution_off.jsonl" "$o/train/attribution_on.jsonl"
for a in none fixed droop rule attribution_off attribution_on; do
  timeout 480s xvfb-run -a "$py" -m research.lfc_dual_pcc.evaluate "$@" --models "$o/train" --arm "$a" --out "$o/eval-$a" || exit
done
"$py" -m research.lfc_dual_pcc.compare --reference "$o/eval-none/none.jsonl" --out "$o/eval-check.json" "$o"/eval-*/*.jsonl
```

## 双时标与逐相图策略

研究入口为 `research.lfc_dual_pcc.run_clock_graph`。它复用同一组三机九节点、两个实际 PCC 和两套 IEEE34 的原生设备，馈线源端电压固定。模型由七对独立的 actor/critic 构成。

| 策略 | 决策与信息 | 动作与学习 |
| --- | --- | --- |
| 主网 | 每 2 s 读取严格先于决策到达的两份 PCC P/Q、已交付量、单步能力及报价、SOC 条件时长、限速与约束标志、消息年龄和真正对应的生效请求 | 九个联合动作选择支援份额和请求修正。两次实际送入主网的响应按各自时长积分和折扣，慢窗的折扣为 `0.95 ** 实际窗长秒数`。 |
| 馈线 A、B | 每 1 s 读取当前有效指令、剩余有效期、最近本地已解算测量和已到达的主网频率 | 各自选目标系数和报送的功率档。报价来自同一 DSS 状态允许的候选；计量、时标和可行性检验由接口产生。存在下一慢窗消费者时，报价动作取得该窗的延迟反馈。 |
| 储能 A、B 与光伏 A、B | 每个设备类型在本地读取逐相图、电压、有功／无功、SOC、光伏余量、馈线目标与指令年龄 | 储能头选两台储能的合计放电档并等分，光伏头在与已选储能档联合可行的动作中选增发档。各有独立图 actor 和图 critic，按原生频率及该馈线实际跟踪误差逐快步更新。 |

原生 DSS 的线路和变压器复导纳矩阵保留不同相之间的耦合、接线与分接头，按相节点电压基准缩放后归一化。额定值特征仅取已核实的设备及绕组。每次实际动作另行验证逐相电压、已建模绕组量、逆变器、爬坡及能量平衡。能力报告中的持续时间是固定工况下单个可行快步的 1 s；由剩余能量推算的时间另存，在更长时段没有电压可行性认证。没有可信线路热额定。物理加权图的参考见 Huang、Yang 与 Wang 的 [Physics-Informed Graph-Based Safe RL](https://arxiv.org/html/2610.33654v1)。

训练用一次在策略采样和一轮按反馈时序排列的裁剪策略梯度更新，概率比裁剪区间为 [0.8, 1.2]。每快步更新六个本地头，每个完整慢窗结束时同时更新主网头，价值基线来自各自 critic。训练用后续实际消费者的频率回报；部署只接收已到达的频率、报告和有效指令。旧的反事实归因实验保留独立入口。

在南洋从运行提交的干净研究 worktree 执行下列命令。输出目录应为新目录。阶段 0 记录、机制文件和私有工作点须彼此匹配。此处的 60 Hz 命令只用于工程预试。

```sh
py="$HOME/miniconda3/envs/HARL/bin/python"
export PATH="$HOME/matlab_2025b/bin:$PATH" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
set -- --stage0 "${LFC_STAGE0:?}" --reference "${LFC34_REFERENCE:?}" --mechanism "${LFC_MECHANISM:?}" --frequency 60
out=$(mktemp -d) || exit
"$py" -m unittest -v research.lfc_dual_pcc.test_clock_graph research.lfc_dual_pcc.test_credit || exit
timeout 480s xvfb-run -a "$py" -m research.lfc_dual_pcc.run_clock_graph train "$@" --out "$out/train" || exit
timeout 480s xvfb-run -a "$py" -m research.lfc_dual_pcc.run_clock_graph eval "$@" --checkpoint "$out/train/model.pt" --out "$out/eval" || exit
"$py" -m research.lfc_dual_pcc.audit_clock_graph "$out/train" "$out/eval" "$out/audit.json"
"$py" -m research.lfc_dual_pcc.audit_policy_inputs "$out/train" "$out/eval" "$LFC34_REFERENCE" "$out/inputs.json"
```

在同一工作树可用 `research.lfc_dual_pcc.frequency_inventory` 记录原模型参数，再用 `research.lfc_dual_pcc.probe_50hz` 在内存副本上检查 50 Hz。零事件尾段门为与 50 Hz 的绝对差不超过 0.01 Hz，且相邻 0.2 s 样点的变化率不超过 0.002 Hz/s。该探针单独记录端口功率与两条馈线电压。

### 南洋原生预试与 50 Hz 门

| 项目 | 实测证据 |
| --- | --- |
| 原生训练 | 运行头 `a1eb08ab310428f04bd056579cda985ba2ce0635`。开发事件 +32 kW，22–34 s，种子 1701。两馈线各 95 个相节点、32 条线路、8 台变压器、478 条有向跨端相耦合。24 次联合交换，8 个慢窗、16 个快步，104 个决策。慢头参加 8 次优化，六个快头各参加 16 次。 |
| 参数与留出 | 训练后状态摘要 `0ef0336058a28b494629452cdf6e2353cea6ca8d2d01419d8d9979df2c86dd46`；权重文件 SHA-256 `a1183e44a588ec6b07fd85f14acd7840e3a64bd11cdc1d1994dfef4d58f0ee27`。独立进程读回并冻结，在 +28 kW、23–36 s、种子 2781 上完成 24 次联合交换。训练／留出频差积分分别为 0.00193665／0.00198501 Hz·s，属不同事件。 |
| 独立审计 | 七项测试通过。两个回合的 104 个学习目标各自逐项与真实时刻和实际消费者核对。七对 actor/critic 参数改变量均非零。四个设备策略各自 actor/critic 的节点编码及消息层权重改变量 L2 最小 0.12275。对冻结模型屏蔽已到达报告后，慢头 actor／critic 梯度差 L2 为 1.04326／1.01697。保持真实观测固定，移除编码器中 832.1 与 852r.1 的耦合后，四个设备 actor 及 critic 梯度均改变。这是输入敏感性检验。 |
| 50 Hz | 独立临时副本修改并读回 58 个有名称的物理频率参数，Ts=0.02 s。两条 DSS 馈线求解并通过电压与功率平衡检查，逐相 0.956091–1.044999 pu。三机模型 19–22 s 的原生频率为 49.977762996–49.977763025 Hz，偏差约 0.022237 Hz，超出预设 0.01 Hz。须从 50 Hz 潮流对齐三台机的调速器初始机械转矩、励磁状态及端口工作点，再做双端因果验收。正式 50 Hz 训练和留出尚缺。 |

原生训练与留出原始轨迹的 SHA-256 分别为 `9007a04ed1d0a5ad3450e8b05588b590c4662dfb22745efd374051af492db8f0` 与 `c8668ca08a4edcb4e7e67800a88bd65a54cb7711b1efa4c0984aa9d83d513ba2`。运行目录中 `train/summary.json`、`eval/summary.json`、`audit.json`、`inputs.json` 对应两次独立运行和两次审计。当前证据覆盖可训练性、采样因果性和模型读回，算法性能优越性仍待比较。

历史预试 `b4615735` 的已采集轨迹因把带负荷稳态频率误用于额定初样检查而在更新前停止。后续 `2e9672fc` 逐个重建核对 104 个在策略动作再完成更新；旧原始文件保留。最终运行头另行修正了延迟报告与原生生效指令的绑定，以及逐快步、逐慢窗的实际优化时序。

## 第一层的容量约束与通信退化

主网每 2 s 签发两站有功支援请求。每站独立选择已经到达、处于有效区间且数据完整的最新报告。到达时刻必须严格早于决策时刻，过期时刻不包含在有效区间。重复内容只消费一个版本，同序号内容冲突时排除该序号。报告缺失时，该站的观测有效位、年龄与载荷置零，并把该站可分配容量置零。传输链路状态与馈线本地快步测量独立。

九个动作选择期望的站间份额和频差增益。正频率缺额下，请求由频差下垂项与各站已配对的请求减实测交付残差组成。集群上限取配置总限额，各站上限取有效报告的一秒报价和可用功率的较小值。分配先裁剪总量，再把某站超出上限的份额转给仍有余量的站。频率缺额为零或负时不开新上调请求。报告是一秒可行的功率见证，实际每个快步仍以 DSS 候选掩码为准。零请求下如果零动作被否决，选当前可行集合中设定值总量最低的降载档。

主网观测保持 31 维。频差信号后是 A、B 各自的报告有效位和采样年龄，再接两个站的原有 13 项报告数据。训练与评估入口用提交和配置指纹绑定权重，新语义要求重训。下层图、动作库和优化器复用原实现。学习回报沿用实际 PCC 消费时间，馈线报价的辅助反馈只记到对应站报告的首次消费者。

合成验收入口如下。

```sh
python -m unittest -v research.lfc_dual_pcc.test_master_dispatch \
  research.lfc_dual_pcc.test_clock_graph research.lfc_dual_pcc.test_credit
```

原生预试中，在学习版冻结评估之外，用同一权重、留出事件和种子再跑一次 `eval --master-policy fixed`。固定对照用等份额、名义下垂及中档残差增益，并复用同样的通信、容量投影、馈线设备策略和安全掩码。向 `audit_clock_graph` 传入 `--fixed-eval <固定评估目录>` 可核对同一留出事件和冻结权重，并计算实测频差积分、最低点、跟踪缺口及实测端口能量差。性能目标是在成对留出工况上降低频差积分，且每个快步和端口仍通过同一物理门。合成验收验证因果选择、约束和回退。50 Hz 验收还需完成三机工作点重初始化。

正式 50 Hz 联合验收沿用零扰动尾段的 ±0.01 Hz 和变化率限值，全程读取原生频率。在同一零扰动窗口并列记录三个机组按各自电气基准换算的转速，以及 PCC 的原生测频，用于区分测量基准偏移和真实低频。保持两 PCC 的稳态负荷，按原 Simulink 模型的潮流初始化方法求得 50 Hz 对应的机械转矩和励磁初值，再核对调速器参考、两端 P/Q 与功率平衡。

物理图消融先冻结工况、策略结构与参数量、观测、动作掩码和训练预算，将跨节点消息置零并保留其余路径。随后保持相别、边集合和训练设置一致，比较复导纳权重与无权边。每个条件使用相同的留出事件和成对种子。

### 冻结回放与候选表示

在 `83d30ce1` 的 60 Hz 配对留出中，26–36 s 的六个慢窗里，九个主网动作经过容量投影都得到同一分配。24 s 两臂的 A 站请求不同，经过馈线目标系数后为 12.5031 与 13.5880 kW，均超过该站约 10.8704 kW 的单步预览上限。在两个原生快步中，两臂最近可行联合档相同，原策略也选择了相同的设备档。冻结回放逐步核对所选候选的 PCC 与 SOC、原掩码和原策略选择。

`diagnose_dispatch` 接收两份完成的留出目录、训练检查点和已验收的 IEEE34 工作点。用冻结代码和相符的环境执行，按历史动作逐步重建两个独立的 DSS 状态。输出含每窗的投影前偏好和投影后分配、每站本地目标、可行候选 PCC 预览及冻结策略打分。候选预览不推进 DSS 时钟或 SOC，回放只执行原轨迹动作，全程无需启动 Simulink。

可选参数 `--tracking-logit-weight` 默认 0。正权重用原物理掩码内候选的实测 PCC 进口降幅计算跟踪偏差，转换为设备 actor 的 logit 偏置。储能档使用其最佳可行 PV 配对的偏差，PV 阶段以已选储能档为条件。该偏置表示可达到的最好配对；策略仍能依据学习得分选择其他合法档。扣除阶段最小偏差后，偏置随目标连续变化。若两个目标均超过最大可行支援，两个偏置相同。采样、更新和轨迹重放共用记录的偏置，权重绑定到检查点。

合成测试覆盖跨物理候选分界、同侧饱和、物理掩码、学习得分覆盖偏置和采样概率一致性。候选表示尚未参与新的原生训练或闭环仿真。
