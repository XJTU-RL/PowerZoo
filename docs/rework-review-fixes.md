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
