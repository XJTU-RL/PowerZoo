# 审核返工报告（第三阶段复审修复）

针对审核复现的五项缺陷与两处关键缺项的返工记录。分支：
`codex/scale-local-obs`（追加提交）。**修复优先级按审核指示**：分组训练/
光伏状态恢复/过载误报 → 真实时序与留出 → 完整性与计时 → 数百台短训练。

## 1. 审核缺陷逐项修复

### B1 分组共享在 MAPPO 下只更新第一组（已修）
- 缺陷：`runners/on_policy_ma_runner.py` 在 `share_param` 真值时只调
  `actor[0].share_param_train(全部 buffer)`——group 模式下其余组的独立
  网络从不更新；`on_policy_base_runner` 的 lr 衰减同样只打第一组。
- 修复：新增 `OnPolicyBaseRunner._shared_groups()`（按网络实例分组）；
  MAPPO 训练按组 `share_param_train(组内成员 buffer 子集, len(组内成员))`；
  lr 衰减按组代表逐组施加。
- 验证：34Bus_PV（19 智能体、3 个空间签名组）MAPPO group 480 步训练完成，
  checkpoint 3 组参数互异且组内逐张量相同——**三组都训练并保存**；
  p4udt4（6 PV 单组）同样通过。

### B2 探测破坏 fixed-pf 语义（已修）
- 缺陷：全量（legacy）探测写 `%Pmpp=100 kvar=0` 且**保持**——模型按
  pf=-0.95 运行的 PV 被改成固定无功 0，辐照变化时 Q 不再跟随 P。
- 修复：探测统一快照/恢复，按**驱动变量**恢复：经 `session.apply` 显式
  写过 kvar 的 PV（`_pv_explicit_q` 登记，reset 清空）恢复显式 kvar；
  其余恢复 `pf`（模型 fixed-pf 语义保留，辐照变化时 Q 跟随 P）。全量与
  子集探测同一逻辑；探测后补解使量测栅栏透明（既有）。
- 验证：真实 p4udt4 的 pf=-0.95 PV 探测后 pf 保持、午间/低辐照 Q/P 比一致
  （测试锁定）；显式 Q 接管设备恢复显式值（测试锁定）。
- 行为变化声明：legacy 全量探测不再把未接管 PV 钉在"满发零 Q"——现在
  保持模型设定运行（更忠实；34Bus_PV 等模型的 PV 无显式 pf 时引擎默认
  pf=1，与旧行为数值等价）。

### B3 light 模式支路过载记成 0（已修）
- 缺陷：基准/基线脚本聚合 `max(... or 0)` 把"未采集"当 0 输出，易误读为
  无过载。
- 修复：`scripts/benchmark_scale.py` 与 `scripts/check_smartds.py` 的支路
  统计在未采集时输出 `null` 并标注 `branch_stats_collected=false`；
  未额定支路（单支路 loading=None）不计入过载的语义写明。

### B4 UK 完整性检查可放过损坏/缺依赖模型（已修）
- 缺陷：zip 场景 verify 只在 zip 文件存在时校验其哈希；展开文件的损坏或
  依赖缺失不被检查。
- 修复：下载时对展开文件逐个 SHA-256 写入 manifest
  （`extraction.files`）并跑**依赖闭包门禁**（redirect/include/LoadShape
  file= 引用缺失即拒绝安装）；verify 默认快检含闭包缺失检查，
  `deep=True` 全量比对逐文件哈希（`--verify --deep`）。
- 验证：13 项 external_cases 测试（含篡改检出）全过。

### B5 耗时重复计算（已修，基准已重测）
- 缺陷：`available_pv_power` 内两次 `solve()` 同时计入 `step.probe` 与
  `step.solve`（嵌套计时重复累计），分段和高于实际。
- 修复：探测内部直调 `_solve()`（无计时包装），两次求解全部归
  `step.probe`；验证 probe 期间 `step.solve` 增量为 0。
- 重测（去双计口径，本机）：
  - 9500Node：步均 293 ms（solve 1.2 / probe 22.7 / apply 20.7 / read
    195.0）
  - 9500Node+light：107 ms（read 11.5）
  - UK+light：1,383 ms（solve 11.9 / probe 122.8 / apply 463.2 / read
    110.6）
  分段之和与步均吻合（无重复）；UK 次级瓶颈为 apply 段逐负荷 Edit
  （463 ms）与观测/奖励组装（余项），为下一步优化目标。

## 2. 真实逐设备时序接入训练环境（关键缺项 1，已补）

- `SmartDSEpisodeDataset`（`envs/powergrid/smartds_timeseries.py`）：
  一个 episode=一天（96×15min），逐负荷 P/Q 与逐 PV 辐照/温度由绑定会话
  的 `SmartDSTimeseries` 直接写入；profile 列仅作观测上下文（全网平均
  辐照/温度/负荷倍率），不参与写入。
- 配置：`env.smartds_series={case, entry, train_days, validation_days,
  split}`（vvc/smartgrid 均支持）；要求 `data.step_seconds=900`。
- **按天留出**：train_days 与 validation_days 不相交；评估 worker
  （`envs_tools` 强制 validation）只见评估天——杜绝相邻时刻随机划分泄漏。
  修复过程中发现并改正 `STEPS_PER_DAY` 常量错误（曾把 96 写成 1440，
  导致 day 索引偏移到 day15——现由测试锁定逐设备数值逐位一致）。
- 验证（`tests/test_real_timeseries.py` 4 项）：train split 只见
  day000/001；validation 只见 day002；env 步进写入的逐负荷 kW 与官方
  曲线**逐位一致**（含日偏移与步进）；全网写入量与引擎总负荷同阶
  （2% 内，无重复计负荷）；两天数据确不相同（防假留出）。
- 短训练（真实数据替代合成）：p4udt4 480 步 2 线程 CPU
  （`configs/experiments/smartgrid_p4udt4_smartds_shorttrain.yaml`）：
  无失败步，电压违规率 1.42%（一月冬季低辐照，PV 出力/Pmpp 0.93%——
  与当日真实曲线一致）。

## 3. 数百台 PV 短训练（关键缺项 2 部分补齐）

- **rdt1534（12.47kV，4,173 节点，472 台受控 PV）真实时序短训练完成**：
  HAPPO + 局部观测（23 维统一，含类型 one-hot）+ `share_param=group`
  + light 读取 + 按天留出，480 步 2 线程 CPU：
  - manifest：observation_mode=local、受控 PV=472、局部 schema=472；
  - checkpoint：472 个 actor 文件加载后参数逐张量相同——**472 台共用
    一个策略网络**（而非 472 份）；
  - 末段指标：平均奖励 -34.9，电压违规率 0.29%（1,152 相节点·步），
    PV 出力/Pmpp 6.0%（冬季），网损均值 64.3 kW。
  - 480 步（2 线程）约 40 分钟完成（含每回合 ~60s 的会话 reset）。
- 诚实声明：这是**链路与规模的验证**（472 个受控智能体端到端训练
  执行），不构成控制效果、收敛性或跨场景泛化结论；英国 112,887 节点
  系统仍只有 2 个调压智能体——节点规模≠智能体训练规模（审核正确指出，
  本阶段未改变该算例的可控设备数量；其大规模训练不在本阶段范围）。

## 4. 其他修正（返工中发现）

- 局部观测维度统一为 23（类型 one-hot 5 + 母线电压 3 + 邻域摘要 5 +
  时间 2 + 进度 1 + 设备特征定长 7）：异维观测（12/13/18）在并行环境
  按 (n_envs, n_agents, dim) 堆叠时崩溃；统一维度+类型编码后混合类型
  系统可训练，分组共享仍按动作空间分。
- 实验文件命名：新实验避开 `*smoke.yaml` 契约（240 步约定），改名
  `*_shorttrain.yaml`。
- `vvc_config` 缺 `Optional` 导入（新字段引入）修复。
- 探测恢复语义更新后，任务 2 的一个旧断言（依赖直改 command 的探测
  遗留值）改为经 apply 的显式路径。

## 5. 测试与回归

- 新增/更新测试：`tests/test_real_timeseries.py`（4 项：留出/逐设备一致/
  防假留出）、`test_smartds_import.py` 增 2 项（pf 保持、显式 kvar）、
  `test_local_observation.py` 维度断言更新。
- **全套回归 417 项全过**（`python -m pytest tests/ -o addopts="" -q
  --ignore=tests/envs`，含既有 9500 慢测试、任务 1/2/3 全部测试）。
- 训练运行：p4udt4 真实时序短训练 ✓、34Bus_PV/p4udt4 MAPPO group ✓、
  rdt1534 472 PV 真实时序短训练 ✓（命令见各实验 YAML 头部注释）。

## 6. 复现命令

```bash
# 真实时序短训练（p4udt4，训练 day0-1/评估 day2）
python scripts/train.py --config configs/experiments/smartgrid_p4udt4_smartds_shorttrain.yaml \
  --set algo.share_param=group --set train.n_rollout_threads=2 \
  --set train.num_env_steps=480 --set train.episode_length=60 \
  --set model.use_recurrent_policy=false --set device.cuda=false
# 472 台 PV（rdt1534）
python scripts/train.py --config configs/experiments/smartgrid_rdt1534_smartds_shorttrain.yaml \
  --set algo.share_param=group ...   # 其余 --set 同上
# 基准（去双计口径）
python scripts/benchmark_scale.py --cases 9500Node 9500Node_light UKMVLV_HVUG_light --steps 12
# 完整性
python scripts/download_external_case.py --case uk_mvlv_hv_ug --verify --deep
python -m pytest tests/ -o addopts="" -q --ignore=tests/envs
```

## 7. 仍未完成（如实列出）

1. 变电站级 rhs0_1247 全站接入（数据清单+站内设备控制权设计）。
2. 英国算例的可控设备扩展（如给 LV 站增配调压/无功资源——属模型派生
   设计决策，需单独审查）与其 112,887 节点训练评估。
3. 局部 vs 全局观测的正式学习效果对比（含统计显著性）；长时序（全年）
   留出协议（本阶段仅 3 天示例：2 训练 + 1 评估）。
4. UK apply 段（463 ms 逐负荷 Edit）与观测/奖励组装段的批量化优化。
5. rdt1534 每回合 reset ≈60s（曲线解析）的派生二进制缓存优化。

## 8. 4d8af3b5 复审七项修复（分支 `sheldon123z/review-fixes-4d8af3b5`）

针对复审报告（`docs/review-remediation-4d8af3b5.md`，见规则 worktree）
确认的七项缺陷，逐项"失败复现 → 最小修复 → 针对性测试"返工。每项
一个提交：

| # | 缺陷 | 提交 | 状态 |
|---|---|---|---|
| 1 | P1 MAPPO 分组 FP 优势串位与日志错配 | `37b7494c` | 已验证 |
| 2 | P1 PV 探测恢复漏掉模型原生 constant-Q | `47d27704` | 已验证 |
| 3 | P2 按天留出缺少不相交门禁 | `e0251af6` | 已验证 |
| 4 | P2 未采集支路数据仍记过载数 0 | `e2f45cb5` | 已验证 |
| 5 | P2 性能读取分段重复计时 | `4b8c9fcf` | 已验证 |
| 6 | P2 UK 深检 CLI 未完成 | `2b2931a8` | 已验证 |
| 7 | P2 真实时序未进训练数据冻结记录 | `9028105f` | 已验证 |

### 8.1 MAPPO 分组 FP 优势串位与日志错配（P1，已验证）

- 根因：group 模式把全队 advantages 配组内 buffer 传
  `MAPPO.share_param_train`，后者用组内序号切 agent 维——组 [0,2]/[1]
  的成员 1（真实 agent2）收到 agent1 的优势；训练信息按组顺序 extend
  拼接成 [A,A,B]。另发现前置崩溃：FP 归一化用 stack 后 4 维
  active_masks 掩蔽 3 维 advantages 必然 IndexError，FP 路径到不了
  分组逻辑。
- 修复：FP 按成员索引切 `advantages[:, :, members]`；EP 保持全队共享
  优势；训练信息预分配按实际 agent ID 回填（[A,B,A]）；掩蔽 squeeze
  尾维。
- 验证：`tests/test_mappo_group_fp.py` 9 项（修复前 5 failed 复现三类
  缺陷）；覆盖 FP 切片、EP 共享、active mask 掩蔽归一化、全队共享、
  不共享 FP/EP 四条路径日志、真实 MAPPO 组件在非连续组数据上两个
  独立网络都更新。

### 8.2 PV 探测恢复保留模型原生 constant-Q（P1，已验证）

- 根因：恢复只认 apply 显式写过的 kvar（`_pv_explicit_q`）；模型原生
  `kvar=10` 的恒无功设备被当 fixed-pf 恢复（写 pf=1），探测后 Q 钉 0。
- 判据实验（OpenDSS 实测）：kvar 与 pf 互为派生、getter 读不出驱动
  标志；模型只写 kvar 时 pf getter 恒为 1（默认值不派生），写 pf 时为
  设定值；运行期 pf→kvar 切换后 pf getter 残留旧值——因此采用
  **reset 后注册 + apply 跟踪**而非读数推断。
- 修复：`_pv_drive_mode` 注册表；`|pf|<1` 判 pf 驱动，其余按 kvar
  驱动恢复（pf=±1 且 kvar=0 时两种写法行为等价）；apply 写 kvar 更新
  注册；定义行同时写 kvar 与 pf 且 kvar 靠后的模型不受支持（参数顺序
  语义，已注释）。
- 验证：`tests/test_powergrid_session.py` 新增 6 项（kvar=10 探测后降
  辐照 Q 保持 10——修复前 0，复审复现值锁定；pf=-0.95 Q 随辐照派生；
  apply 显式/reset 回原生/子集探测不越界）；全套 33 项通过。

### 8.3 按天留出门禁（P2，已验证）

- 根因：dataset 只复制日号列表、仅检查选中 split 非空；真实 p4udt4 曾
  接受重叠配置，两个 split 都 reset 到 day000（首负荷相同，留出失效）。
- 修复：`validate_holdout_days`——0..364 整数（bool 显式排除）、无
  重复、train/validation 不相交；split 只允许 train/validation；启用
  评估但 validation_days 为空在主进程 spawn 前报错。
- 验证：`tests/test_real_timeseries.py` 新增 11 项（重叠/重复/-1/365/
  0.5/"3"/True/None/未知 split/空评估留出，修复前 10 failed）。

### 8.4 支路统计三态契约（P2，已验证）

- 勘误：本报告 B3 曾声称"未采集输出 null 并标注 collected=false"——
  复审确认该契约当时并未交付（light info 实为 0/0 缺标记），本次补齐。
- 修复：`summarize_branch_stats` 三态（未采集=collected False 全 None；
  已采集=事实计数+unrated 单列）；check_smartds 提取 `branch_summary`
  去掉两处 `or 0`，summary 聚合跳过 None。
- 验证：纯函数四态 + 真实 light/full 步 info + 脚本聚合共 8 项新增
  （修复前 7 failed）；真实跑 check_smartds 四模式 × 96 步（最大载荷
  pu：native 3.529 / fixed 3.529 / rl_q_only 3.332 / rl_pq 2.645）。

### 8.5 基准分段互斥（P2，已验证）

- 勘误：本报告 B5"分段之和与步均吻合"在当时口径下不成立——
  read_state 内嵌 branch_state 双计（9500 full 分段和 392.8 > 墙钟
  296.0），负余量被 `max(other,0)` 钳零；B5 修复的 probe/solve 双计
  路径保持有效。
- 修复：`exclusive_step_ms` 把 branch_state 从 read_state 拆出为互斥
  五段；`step_residual` 保留原始余量、负值超墙钟 5% 时输出
  `step_timing_anomaly` 诊断。
- 验证：复审数据回归（204.990-145.897=59.093）+ 异常标记 + 真实
  34Bus_PV probe 不含 solve 三项新增；实跑基准分段和均低于墙钟：
  34Bus_PV 3.3<4.8、P4UDT4 2.9<3.9、9500 full 235.5<285.1、9500 light
  54.7<99.4（macOS arm64，3-5 步去首步预热）。

### 8.6 UK 深检 CLI（P2，已验证）

- 勘误：本报告 B4 提到的 `--verify --deep` 当时 CLI 参数不存在（退出
  2）；深检 API 本身有效。本次补 CLI。
- 修复：`--deep`（仅与 --verify 组合）传入 verify；帮助文本写明
  file_set 算例校验本就逐文件、旧 manifest 迁移=--force 重下重建清单
  后自行替换，不把现存未知文件自签为可信。
- 验证：新增 5 项（篡改展开文件快检过/深检败、旧 manifest 指引重下、
  --deep 单独用退出 2、CLI 三态退出码、本机真实 UK 旧 manifest 深检
  明确失败）；真实上游验证：临时目录重下 UK（9.3MB/3282 文件）深检
  退出 0，篡改 generators.dss 退出 1 并指认该文件。
- 未完成：本仓 `data/external` 的 UK 旧 manifest 未替换（迁移须用户
  决定），"真实完整性验收"待重下后标记。

### 8.7 真实时序数据冻结（P2，已验证）

- 根因：prepare_run 对 smartds 场景仍冻结合成 ScenarioDataset（worker
  不用）；真实 dataset 哈希只含路径/master/日期/步数——曲线内容改动
  不影响摘要，搬目录反而变摘要。
- 修复：`SmartDSTimeseries.content_digest`（脚本+全部曲线字节，无绝对
  路径）；dataset.sha256 惰性属性（内容指纹+日号/split+单位/时间基准）
  支持 `content_sha256` 恢复校验（control_env 构造期触发）；prepare_run
  smartds 分支冻结 dataset_sha256/manifest_sha256/days/split/master，
  不再写入合成 path/manifest_sha256；合成路径行为不变。
- 验证：新增 4 项（负荷曲线改一字节摘要变+旧摘要恢复被拒、仅搬目录
  摘要稳定、PV/温度曲线与映射脚本参与摘要、prepare_run 冻结与 worker
  构造一致且无合成写入）。

### 8.7b 真实训练补充验证与 split 维度修复（9b5ceedc）

七项修复提交后，对两条复审边界补跑真实训练，暴露并修复一个新缺陷：

- **EP 34Bus_PV MAPPO 三组短训练（48 步 ×2 线程）**：P1-1 runner 重构
  后已确认边界保持——checkpoint 实测 3 个独立网络（成员 [0,1]/
  [2–9]/[10–18]，与复审基线"三组 2/8/9"一致），组内逐张量相同、
  组间互异、三组均非初始化值；训练与评估（含 log_train 消费重构后
  的 actor_train_infos）完整跑通。
- **p4udt4 真实时序端到端（96 步 + day2 评估 worker）**：首跑暴露
  9028105f 的缺陷——dataset 摘要组成含 split（不同 split 是不同数据
  切片，设计内），但 prepare_run 只冻结 train 口径一个摘要，评估
  worker（envs_tools 强制 split=validation）构造期校验必然失败
  （实测期望 aedd170b/实际 bc1fa98e）。修复：validation_days 非空
  时额外冻结 `dataset_sha256_validation`，control_env 按 worker 的
  split 取对应期望值。修复后端到端训练+评估通过（电压违规率 5.80%、
  PV 出力/Pmpp 7.31%），train/validation 两口径校验实测生效。

### 8.8 回归验收（2026-11-15，本机 macOS arm64 / Python 3.12.8）

- 七个指定测试文件 112 项全过、0 skip（外部数据经 `data/external`
  符号链接只读引用既有算例，真实算例测试全部执行）。
- `tests/test_runners/test_algorithms/test_common/test_mappo_group_fp`
  37 项通过；`test_config_pipeline/test_training_entrypoints/
  test_unified_config` 71 项通过。
- `python scripts/check_runtime.py --train`：POWERZOO_RUNTIME_OK（含
  24 步 HAPPO smoke 与 spawn 双 worker）。
- `pip check` 冲突如实报告：共享 miniforge base 环境既有冲突
  （daytona/gradio/pyopenssl/crawl4ai/patchright/mem0ai 等），另
  powerzoo 声明的 h5py/imageio 因 `--no-deps` 安装未装（本次测试
  不依赖）；不把测试通过表述为整个依赖环境通过。
- 未验证/剩余限制汇总：真实 FP 分组短训练未跑（现有算例以 EP 为主，
  组件级已覆盖）；checkpoint 恢复端到端重放未单独实跑（info 绑定与
  构造期校验分别覆盖）；rdt1534 级数据集每次构造读全部曲线字节未加
  缓存（约秒级）；本仓 UK 旧 manifest 待用户决定是否重下迁移。


## 9. PR60 本地整合补充（基于60ac9ea6，未提交）

GLM按用户此前任务推进至60ac9ea6的内容保留；本地补充修复已在独立的
`sheldon123z/pr60-integrated` 工作区整合，未修改指定GLM工作区，也未执行Git合并。
逐split摘要保留GLM格式，checkpoint绑定完整划分表；另纳入LFC None mask、
PV实际驱动恢复、日期懒加载及可信ZIP清单迁移。旧章节中的驱动模式猜测等表述
是此前阶段记录，当前实现和371项相关回归证据以
[整合修复与验证](pr60-followup-validation.md)为准。
