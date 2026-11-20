# VVC 回合指标汇总（scripts/vvc_benchmark_metrics.py）

针对 XJTU-RL/PowerZoo（基线 commit de8ce3c7）native 评测输出的**窄口径离线统计工具**：读一个回合的 JSONL 记录，向 stdout 输出保守、确定性的汇总 JSON。仅本地文件读写，无网络；仅依赖 Python 3.10 标准库；不改动框架、环境（`envs/powergrid/`）或任何既有文件。

## 适用范围（先读这个）

- **指标只刻画一个回合**（one episode）。所有输出均为单回合统计，不构成跨回合、跨切分或跨数据集的结论。
- **仅允许在 validation 切分上选择 checkpoint**。本工具也可汇总该 checkpoint 在 test 上的单回合记录，但不做跨回合比较或泛化性声明。
- 数据只来自 `envs/powergrid/scenario.py` **已冻结且互不相交**的既有 train/validation/test 清单；真实 trace 的哈希跨切分去重。注意：noise-zero 的合成样例在切分之间可能完全相同，**它们不是真正的 holdout**，不能据此声称泛化。
- **MAPPO 与 SHOM 均未验证**。本工具不评估、不实现任何策略，也不添加训练、网络调用或 native 控制改动。

## 输入记录（JSONL，每行一个 JSON 对象）

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `dss_converged` | bool | 除原生失败记录外必填，严格布尔 | 本步求解器是否收敛 |
| `physical_failure` | bool | 可选，严格布尔 | 原生失败步标为 true；不得同时标为已收敛 |
| `step_seconds` | number | 收敛步必填，正有限 | 本步实际步长区间（秒），允许逐步不同 |
| `power_loss_kw` | number | 收敛步必填，非负有限 | 本步网损功率 |
| `voltage_violation_count` | int | 收敛步必填，非负整数 | 本步电压越限计数 |
| `voltage_min_pu` | number | 收敛步必填，有限 | 本步最低电压（p.u.） |
| `voltage_max_pu` | number | 收敛步必填，有限 | 本步最高电压（p.u.），不得低于 min |

首个 `dss_converged` 为 `false` 或原生 `physical_failure` 为 `true` 的步即**求解失败步**：该步其余字段允许缺省（不会被读取），统计在该步停下并打标记；其后的记录**一律不再消费**，只在 `failure.trailing_records_ignored` 中报告被忽略的条数——失败之后不存在被静默采纳的数据。失败步自身不计入能量、电压与越限计数。其余未列出的字段一律忽略。

## 输出字段

| 字段 | 说明 |
| --- | --- |
| `valid` | 仅当回合完整（记录数等于 expected_steps）且全部收敛时为 true |
| `status` | valid / incomplete / solver_failure |
| `expected_steps` / `steps_completed` / `steps_processed` | 期望步数 / 收敛步数 / 实际消费记录数（含失败步） |
| `energy_kwh` | 完整回合能量；不完整或失败时为 null |
| `diagnostic_energy_kwh` | 收敛前缀的部分能量（诊断用，始终给出） |
| `voltage_min_pu` / `voltage_max_pu` | 收敛步上的最低 / 最高电压 |
| `voltage_violation_count_total` | 收敛步越限计数的直接求和（**计数，不是速率**） |
| `step_seconds_min` / `step_seconds_mean` / `step_seconds_max` | 每步实际步长区间的观测统计；**不是回合时长**（汇总不推断回合结束时间） |
| `failure` | null，或 {step_index, reason, trailing_records_ignored} |
| `voltage_violation_interval_hours` | 有至少一相越限的区间小时；每个区间只计一次，不是相小时 |
| `voltage_max_exceedance_pu` | 全回合最大越限幅度（p.u.）：max(0, 下限−Vmin, Vmax−上限) |
| `eligible_for_day_ranking` / `day_ranking_refusal` | 只在完整且收敛、区间合计为 24 小时时通过覆盖门槛；否则明确拒绝原因 |

上述两个全回合电压时长/幅度字段在失败或截断时为 null；`diagnostic_`
字段保留收敛前缀用于诊断，不能当作全天结果。幅度必须有显式电压上下限，
离线 CLI 用 `--voltage-lower-limit-pu` 和 `--voltage-upper-limit-pu` 同时设置，
原生导出直接使用 checkpoint 对应的 VVC 配置限值；不代填。
越限计数与同一限值下的极值矛盾时拒绝输入。覆盖门槛只验证 24h 和完成状态，
跨方案比较还须使用相同的冻结场景、时段与限值。

能量按每步**实际区间**积分：`energy_kwh = Σ power_loss_kw × step_seconds / 3600`。

## 用法

```bash
python scripts/vvc_benchmark_metrics.py episode.jsonl --expected-steps 24
```

- 输入：本地 JSONL 文件（每行一个 JSON 对象，空行跳过）。
- 输出：stdout 单行 JSON（键排序）。
- 退出码：0 成功；2 文件读写失败或校验失败（stderr 给出原因）。

校验失败（ValueError）包括：记录条数超过 `--expected-steps`、`dss_converged` 非严格布尔、收敛步缺字段或数值非法（非正 step_seconds、负网损、非有限值、非整数计数、min 大于 max 等）。

## 不做的事

- 不输出越限速率或相小时，也不从记录推断绝对结束日期。
- 不新增策略 / 训练 / 网络调用 / native 控制改动；不改清单与环境。
- 不做任何跨回合聚合或泛化性声明。

独立测试（无需加载仓库测试夹具或 OpenDSS）：`python -m pytest --noconftest -o addopts="" tests/test_vvc_benchmark_metrics.py`。

## 原生回放导出（只开放 validation）

在仓库根目录执行：

```bash
python -m scripts.export_vvc_validation \
  --checkpoint artifacts/happo/models \
  --manifest artifacts/frozen/manifest.json \
  --scenario-index 0 --scenario-index 3 --seed 17 > validation.json
```

回放仍使用正式 HAPPO checkpoint 加载、设备动作和物理绑定校验。
训练快照必须已绑定 train 清单路径和 SHA-256，显式传入同一清单；
导出只切换到 validation，不读取 test 曲线。场景索引按经过训练配置
场景过滤后的 validation 列表排列，从 0 开始，必须显式且不重复。
每个回合报告实际场景 ID、清单 SHA-256、种子、分集、供统计的逐步记录
及单回合指标。初始快照不计入，物理失败单独标记；不导出设备动作计数。
这不是完整的基准评测或模型自动选择器。实际 HAPPO/数据组合需另行验收，
不得用 test 挑 checkpoint，也不能把文件不同但曲线重复的样本视为有效留出。

`--allow-mappo` 显式放行 VVC 的非共享 MAPPO checkpoint，默认仍只接受
HAPPO。要求 checkpoint 的正式 resolved 配置显式记录 `share_param: false`
及原始模型参数，不补猜 RNN 设置、不共享或复制 actor 权重；物理清单校验
和严格加载保持不变。一般渲染和其他任务没有启用该选项。实际 MAPPO
checkpoint 逐步验证通过前，此范围仅为待验收的兼容入口。

### IEEE 34Bus fixed-device and local-regulator validation

Use the training run's own resolved configuration and physics sidecar; this
command does not open any model weights:

```sh
python -m scripts.export_vvc_baselines \
  --resolved-config artifacts/config.resolved.yaml \
  --physics-manifest artifacts/physics.manifest.json \
  --manifest artifacts/scenario_manifest.json \
  --scenario-index 0 --scenario-index 3 --seed 17 > baselines.json
```

Use the same explicit validation indices and seed in the policy exporter.
Each pair starts from a separate, normally compiled and reset VVC core.
Intervals use the same profile/time and applied scenario data as policy
replay. `hold_initial` never writes capacitor states or transformer taps
and disables DSS automatic controls. `native_regulator` instead lets the
enabled RegControls own their respective transformer windings in DSS STATIC
mode. Both explicitly command storage idle (0 kW, 0 kvar), leave capacitors
untouched, and check measured idle power and unowned states after solving.
The exporter refuses unfamiliar DSS element classes or ambiguous regulator
ownership, reports the controller inventory and the initial/last converged
device state, and uses the same physical metric aggregation as policy replay.

This is a local **regulator-only** comparator, not an all-device heuristic or
a real-time delay model. Endpoint tap differences are not event counts.
Neither baseline emits a policy reward; failed intervals use the same
incomplete-day refusal rules. Other feeders (including 123Bus), SMART-DS,
test splits and cases with additional control classes are out of scope.

For classic 34Bus validation, both exporters also save `device_endpoints`
(the measured reset state at step 0 and one state per converged interval).
`device_metrics` counts individual capacitor stage bit changes and regulator
devices whose tap ratio changed by more than 0.000001 p.u. between adjacent
endpoints. A regulator contributes at most one change per interval, even if
it crosses several taps. Moves and reversals within the same interval are
not visible in this statistic. These are observed endpoint changes, not
requested commands or native controller operation counts.

Storage has no discrete action count here. Charge and discharge are reported
separately as estimates: multiply each device's signed endpoint terminal
power by the actual interval duration and sum each sign separately (positive
injection is discharge); do not net devices. The reset state has no duration.
The physical metric duration and solver status determine eligibility: failed,
incomplete or non-24-hour replays retain measured endpoints but have null
device totals and an explicit status.

### Paired classic IEEE34 acceptance (no training or leaderboard)

The same existing **adapted** `34Bus` / shared `ieee34` deck is used as the
original classic benchmark. Its exact active DSS Redirect closure is checked.
The two research storage devices, capacitors, and regulator action/observation
layout are unchanged. No `34Bus_PV` deck, SHOM paper weights, or new physical
overlay are inferred. Neither the existing smoke config nor default validation
exporters change.

An operator freezes an external JSON plan **before** running the pair, after
choosing checkpoints without observing any test outcomes. Paths are relative
to that JSON or absolute local paths, not repository files. From the repository
root, run `python -m scripts.export_vvc_baselines --paired-plan artifacts/plan.json`.
Required fields:

```json
{
  "split": "validation",
  "manifest": "dataset/manifest.json",
  "manifest_sha256": "<sha256>",
  "scenario_indices": [0, 1],
  "environment_seeds": [17],
  "checkpoints": {
    "happo": {"path": "happo/models", "sha256": {"config.resolved.yaml": "<sha256>", "physics.manifest.json": "<sha256>", "actor_agent0.pt": "<sha256>"}},
    "mappo": {"path": "mappo/models", "sha256": {"config.resolved.yaml": "<sha256>", "physics.manifest.json": "<sha256>", "actor_agent0.pt": "<sha256>"}}
  }
}
```

Include every actor file from each selected run in its map; the one-actor
example is abbreviated. Use either exact earlier local checkpoints or
new independently initialized smoke-run checkpoints with the same resolved
environment, same manifest and valid sidecars. No training is launched by
this command. It records each checkpoint's training seed separately from the
common environment replay seeds and includes the verified plan, manifest and
checkpoint-file digests and configured training budgets (not an inferred
number of completed updates). It checks actual reset observations and
action masks, device endpoints, selected scenario and physics identities,
and actual step-duration agreement for hold-initial, local-native regulator,
HAPPO and explicitly nonshared MAPPO. Policies respond independently and
need not take identical actions. Results are per-arm, per-scenario and per-seed;
any incomplete arm refuses full-day comparison.

Explicit `"split":"test"` is a final paired run only, with no checkpoint search,
selection or training. The same manifest must contain train/validation/test;
numerically identical exogenous trajectories across splits are refused even
if file formatting or absolute timestamps differ. Distinct synthetic traces
do not establish field representativeness, and numerical disjointness cannot
prove independence of the source data. Freeze and retain the plan digest and
split provenance externally. Ordinary validation is still available if a
genuinely independent holdout is unavailable. No test result or leaderboard
is implied by the example.

**Verified native acceptance:** runner
`a6c78e8e3a59a3b123ade46f4c27f74c74222661` completed one four-arm
validation on `validation_000` with the common environment seed 17. Each
arm converged for 24 one-hour intervals (86,400 s), from the same measured
reset observation/action masks and device endpoints. The HAPPO and nonshared
MAPPO actors were existing checkpoints saved after update 200, not newly
trained for this check. The private report's SHA-256 was
`e0fab35b178b3c4239b80d25aa406e541aa9e05b355e06bea00f6ef1a113c181`;
the report and checkpoints are not checked in. This verifies compatible
replay and full-day coverage for one scenario, not formal holdout results,
comparative performance, or a leaderboard.

## 显式 MultiDiscrete 集中式 PPO 小范围验收（2026-11-20）

功能唯一分支：`sheldon123z/vvc-34bus-ppo`，从本地 `e6990bab` 建立，
代码验收 SHA 为 `2bc3b7806e5914b5bde31d76c9286ca7505c061b`。
`699cb36a` 是基线的祖先。南洋核实 `main=7a2c5f44` 与验收分支共同祖先为
`de8ce3c7`，其差异仅 README 和 LFC 文档；未合并 main 或算例导航分支。
本功能不包含 IPPO、QMIX/VDN、COMA 或 SHOM 排序，后续功能各自独立 worktree。

### 接入与调用

复用 `SingleAgentVVCEnv`、`CentralizedAdapter` 和 SB3 PPO，不重写训练器或电路。
`scripts/train_centralized.py --action-space discrete --torch-threads 1` 显式使用
`MultiDiscrete([2,2,33,33,33,33,33,33,33,33])`，按已有设备顺序控制
两电容、六个单相调压器、两储能。未指定参数时沿用原任务动作模式。
离散动作必须是有限、完整、范围内整数，非法动作在进入物理内核前拒绝。

```sh
python scripts/train_centralized.py --config artifacts/frozen-ppo-input.yaml \
  --steps 96 --action-space discrete --torch-threads 1 --output artifacts/ppo
python -m scripts.export_vvc_validation --sb3 --checkpoint artifacts/ppo \
  --manifest artifacts/frozen-manifest.json --scenario-index 0 --seed 17
python scripts/check_vvc.py --equivalence-config artifacts/config.resolved.yaml \
  --output artifacts/equivalence.json
```

PPO exporter 只允许 validation；paired-plan 可选 `checkpoints.ppo`，其 path 是
PPO 文件前缀，sha256 必须包含 `.zip`、`.environment.yaml`、`.physics.json`、
`.sb3.json` 四个文件。未提供第五臂时保留原四臂行为。
显式 VVC 验证/配对入口通过 `require_vvc_binding=True` 校验完整三文件摘要、
resolved 快照、模式及动作 nvec；配对计划另外绑定 `.sb3.json` 自身。
公共 `load_model` 默认仅沿用原有 `.physics.json` 绑定，不要求 legacy DSR 或其他
MultiDiscrete 模型新增 SB3 manifest；SB3 仍验证模型空间。
环境快照继承原配置格式（run.algo 可仍是环境模板的 HAPPO），实际 PPO 参数以
`.sb3.json` 和 SB3 archive 为准，不能把模板里未被 PPO 消费的通用模型标志当成实际设置。
首轮为 global actor 观测，和既有 global MARL 观测信息一致；不声称局部 CTDE 对照。

### 实际验证

- Mac 纯测试最终 **114 passed**：centralized contract、validation export、baseline export、
  benchmark metrics、device metrics、MAPPO checkpoint gate；无求解器和训练。
- 扩展本地训练入口/配置测试曾得到 **142 passed, 5 failed**，失败为缺少 `dss`、
  `tensorboardX` 及 sandbox 禁止写 results；未当作通过，未为此修改断言。
- 南洋 HARL 原先没有 SB3/pytest。只将仓库已锁定的官方 PyPI SB3 **2.6.0** wheel
  放入本任务私有依赖目录，`--no-deps --no-index --target` 安装，不改共享 HARL。
  wheel SHA256：`d4a4c95b5593c6135a5764767d829498677559069110aa3768fe1254f1c457e3`。
  南洋在线 PyPI 因 TLS 失败，改由 Mac 缓存取得同一包并核对两端摘要，没有关闭 TLS 校验。
  实际 HARL 为 Python 3.10.14、torch 2.3.0+cu121、NumPy 1.26.4、DSS-Python 0.15.7、
  DSS backend 0.14.5；不是声称整个 HARL 都符合仓库完整 requirements 锁定版本。
  `pip check` 返回 No broken requirements found；`check_runtime.py --train` 通过，
  包含 CPU HAPPO 24 步及 1/2 worker 检查。未做 GPU 训练。
- 原生等价检查仅在南洋、单线程、120 秒超时下运行：`validation_000`、环境种子 17，
  各 24×3600 秒。从量测 reset 档位构造动作，电容翻转、两相调压器相反方向一档、
  两储能相反方向一档；按设备名/绕组核对请求、下发、端口 P/Q 和原生 SOC。
  全网状态、观测、团队标量奖励、终止与时钟逐步一致（数值容差 1e-7；float32 团队奖励 1e-6）。
  不把重复给每个 agent 的团队奖励求和。最终检查耗时 **4.856 s**。
- 顶层 solve 各 **24** 次，内核 `_solve` 各 **72** 次。现有 legacy 空 PV 探测仍有内部求解；
  本次 adapter 不增加求解或时间推进，未重构该公共物理路径，不能声称全天只做 24 次原生潮流。
- 曾有补充断言误把储能请求 6.25 kW 当成应量测功率，实测约 6.02345 kW 导致失败。
  修正为分别核对请求/下发及独立原生端口量测，保留 1e-7 容差，不改变模型或放宽阈值。
  终点惩罚敏感性测试仅在独立配置设 soc_w=5；SOC 总偏差 **0.06138986473018898**，
  终点控制奖励 **-0.3069493236509449**。正式五臂环境保留原 soc_w=0。
- PPO 训练种子 **23**、环境验证种子 **17**；训练源仍为既有冻结 train manifest。
  最终模型真实交互 **96** 步，rollout/batch=24，n_epochs=1，CPU，真实 optimizer.step **4** 次，
  参数变化且保存恢复逐项相等；learn 耗时 **1.0129 s**，进程约 **5.731 s**。
  为增加 optimizer 审计前还跑过一次同配置 96 步，合计 PPO 交互 **192** 步；
  二次运行是审计补充，未根据 validation 结果调参或选模，首次工件保留。
- 固定设备、原生调压、既有 HAPPO、非共享 MAPPO、PPO 的最终五臂回放均收敛并完整覆盖
  同一 validation 日，每臂 **24** 步，耗时 **5.422 s**。HAPPO/MAPPO 使用原地已有
  update-200 checkpoint（训练种子均 1），校验冻结文件摘要后恢复，没有复制旧权重。
  五臂初始化、观测/动作绑定、清单和时钟通过一致性门槛。

### 私有工件与边界

工件保存在任务私有目录，以本节的相对文件名与 SHA256 定位；公开文档不记录机器绝对路径。
依赖在 `deps/`；最终模型前缀 `ppo-verified`；训练日志 `train-verified.log`。

| 工件 | SHA256 |
|---|---|
| `equivalence-final-v2.json` | `2442861708cdc8c1073a6b4666a490c6457583681e01a4890ac775cd92b72036` |
| `five-validation-plan-final.json` | `98c6c94073f999e95e29f7d1627d201cd8e6275a75d3bed436a872d2d4511510` |
| `five-validation-final.json` | `d2a4f7e5ccf3c9b65fef730bbc6e00888c2302948c08da95a6482eb8064941ad` |
| `ppo-verified.zip` | `f3b97cb936fab6d39c1c5e57a26c3f612eceb3ea86c7d099727bad08a4095fb2` |
| `ppo-verified.sb3.json` | `3fa9322b0fb68c293068c3294981267399329103aeeef2d52e14f9d33f5f6f0b` |

清单 SHA256 为 `0a310450224e114ceba3d380b1721b80dc85f902d2611625140f2b3814333333`。
所有模型/数据/逐步结果留在私有目录，未入库。冻结 test 未打开或评分。
结果仅证明接入、恢复与单验证日兼容性，不证明策略安全、学习收敛、泛化、等预算公平性或排名。
正式多种子、同预算对照仍需独立协议与验收；未启动长训。
当前复用 SB3 Gymnasium 接口；未来 RLlib 需要独立注册/适配与 checkpoint 协议，
不应放进此次 PPO 依赖或改写现有训练器。


### 发布前复核修正

本次仅修改加载前的兼容/完整性门槛及文档，不改变训练算法、动作、奖励、物理或已有工件，
因此沿用上述原生证据，未重训或访问留出数据。严格 VVC 拒绝空/部分/额外摘要条目、
文件缺失或替换、非 resolved 快照及模式不匹配；旧默认恢复继续要求合法物理 sidecar。

任务累计历史 `e6990bab..ff968464` 中，`ff968464` 曾加入私有机器绝对路径。
当前净树已移除，但新增清理提交不能删除历史内容。本任务不改写历史、不推送。
发布时必须从经过审查的公开安全基线，按审查后的净树生成新提交，同时补齐必要依赖改动；
不能直接推送、合并或 cherry-pick 含该路径的旧提交，也不能仅在旧历史上追加清理后发布。


复核纯回归 **153 passed，0 failed，0 skipped**（0.44 s），涵盖默认 legacy DSR/VVC/
SmartGrid/区域调度加载、合法与非法物理 sidecar、严格 VVC 三文件逐一缺失/替换、
空/部分/额外摘要、resolved/模式/nvec/算法绑定，以及 paired-plan 对 `.sb3.json`
自身摘要缺失/替换时在读取任何场景之前拒绝。测试使用模拟 archive 与加载器，
不反序列化真实权重，不初始化原生引擎；不将其表述为新一轮原生通过。
`git diff --check` 通过。当前文档中的示例均使用相对路径，任务历史审计没有改写任何提交。

### 原生严格加载及退出码复核（2026-11-20）

此前纯测试的任务标识使用小写 mock token，漏掉了实际原生契约：
`DeviceControlEnv.get_physics_manifest()` 的 task 来自 `type(config).__name__`。
严格 VVC 恢复现按 `VVCConfig.__name__` 校验，继续拒绝 DSRConfig、其他任务及伪小写标识。
测试夹具也改由真实 VVCConfig 类型产生 token。此前 153 项纯测试不能作为原生严格加载通过的证据。

配对命令含 PPO 时改由 `all_five_full_day` 决定退出码；缺失第五臂覆盖标记也失败关闭。
不含 PPO 时保留四臂门槛。新增回归确认 `all_four=True/all_five=False` 返回 1，
旧四臂成功仍返回 0。最终纯测试 **161 passed，0 failed，0 skipped**（0.34 s）。

代码 SHA `d5144ac81774576aa5f0d99b5c77338124fe5b5b` 在南洋进行了**一次**限时原生严格验证：
先校验既有 PPO 归档及三个 sidecar 的原记录摘要，再使用真实 `VVCConfig`/DSS 环境加载。
经 `require_vvc_binding=True` 加载原模型，调用 SB3 save 到新的私有临时目录，
重新绑定新文件名的完整摘要并严格 reload，模型参数逐项相等；随后通过真实 SB3 validation
exporter 的严格路径回放 `validation_000`、种子 17，完成 **24×3600 秒**、全部收敛且全天覆盖。
用时 **1.789131925 s**，SB3 2.6.0；本次训练 **0 步**、安装 **0 项**，未读取 test。
原物理指纹匹配，既有模型和实验文件没有修改。此记录取代“严格加载仅经 mock 验证”的空缺，
仍不构成公平训练预算比较、策略安全或正式排名。

| 新私有证据（仅文件名） | SHA256 |
|---|---|
| `replay.json` | `e01ed9193c6f18cfc623fec6bbc66feff4f411347b004ad6791254878cdc546a` |
| `summary.json` | `9ca64522293985c3f60c12460d352ae413c5568b7c2c91b1037cd7a0e1023079` |

候选沿公开 `699cb36a` 的干净祖先继续追加修复，不改写历史；旧试验分支保持只读。
发布仍暂停，未推送、建 PR 或合并。公开文档不记录私有绝对路径或原始工件。
