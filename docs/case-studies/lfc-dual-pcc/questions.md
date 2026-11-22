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
