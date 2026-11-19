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

- 不输出速率（计数不除以步数或时长），不推断、不输出回合时长。
- 不新增策略 / 训练 / 网络调用 / native 控制改动；不改清单与环境。
- 不做任何跨回合聚合或泛化性声明。

独立测试（无需加载仓库测试夹具或 OpenDSS）：`python -m pytest --noconftest -o addopts="" tests/test_vvc_benchmark_metrics.py`。

## 原生回放导出（只开放 validation）

在仓库根目录执行：

```bash
python -m scripts.export_vvc_validation \
  --checkpoint /path/to/happo/models \
  --manifest /path/to/frozen/manifest.json \
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
