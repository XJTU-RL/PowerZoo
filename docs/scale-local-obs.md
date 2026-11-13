# 局部观测与大规模性能验证（第三阶段）

状态：**小馈线短训练 + 中/大规模固定策略基准完成，等待审查**。任务分支
`codex/scale-local-obs`（基于任务 2 的 `codex/smart-ds-timeseries`）。本阶段
不改电气求解语义：所有优化以"同完整物理基准逐位一致"为验收（测试锁定）。

## 1. 范围与硬件

机器：Apple Silicon arm64（macOS 26.4），10 核，Python 3.12.8，
dss-python 0.15.7 / C-API 0.14.5，torch 2.9.1 CPU。单进程顺序量测；
计时=会话内埋点（`DSSSession.timings`：init.copy/sha256/context、
reset.compile/connections/discover/topology/cache_hit、step.apply/probe/
solve/branch_state/read_state）+ 步外层计时，首步一次性开销剔除；策略推理
用固定 2×64 MLP 前向参考（非实际网络）。内存=进程单调峰值 ru_maxrss。
复现：`python scripts/benchmark_scale.py --cases <...> --steps 12`，
结果 JSON：`results/validation/external/scale_benchmark.json`。

## 2. 优化前基线（先测量）

| 算例 | 节点/智能体 | 单步均耗 | 其中 read_state | 求解 | 构造（含首次 reset） |
| --- | --- | --- | --- | --- | --- |
| 34Bus_PV | 116 / 19 | 4.4 ms | 1.7 ms (39%) | 0.5 ms | 0.6 s |
| 9500Node（k=2 投影） | 9,549 / 30 | 288 ms | **200 ms (69%)** | 12 ms | 1.0 s |
| UKMVLV_HVUG（k=2） | 112,887 / 2 | **4,095 ms** | **2,865 ms (70%)** | 118 ms | 15.4 s |
| SMART-DS p4udt4 | 156 / 6 | 3.8 ms | 2.0 ms | 0.4 ms | 10.0 s |
| SMART-DS rdt1534 | 4,173 / 472 | 163 ms | 84 ms (51%) | 14 ms | **278 s** |

瓶颈结论：**状态读取（全母线逐相电压循环 + 全支路电流/损耗扫描 + 全负荷
逐元件功率）占大系统单步 51–70%**；rdt1534 构造 278 s 主因 master 自带的
compile 期 35040 步全年求解（每次 reset 重放）；UK 每步 apply 段（19,072
条逐负荷 Edit）482 ms 为次级瓶颈。

## 3. 优化与等价性验收（不改变控制器与求解行为）

三项优化，全部有等价性测试锁定（`tests/test_scale_optimizations.py`）：

1. **结构缓存**（`session.reset`）：同一会话的模型临时副本生命周期内不变，
   连接清单/相别映射/拓扑/设备铭牌跨 reset 复用（按 `model_sha256` 键控）；
   缓存命中时重放 discover 的编译期副作用（Storage `DispMode=external`、
   SVG `status=fixed`）并从引擎重读 SOC/PV 动作记忆等运行态。
   *等价性：同场景同动作下缓存命中前后的逐步观测逐位一致（测试）；*
   UK 首次 reset 的 connections+discover+topology ≈5.0s 在后续 reset 归零。
2. **精简状态读取**（`read_state(light=True)`，配置 `light_state_read` 默认
   **关**）：训练步用一次 `AllBusVmagPu` 批量按节点名分组电压（与逐母线
   puVmagAngle 同源同值、按 bus_names 排序对齐键序），注入只取设备类，
   省略全网支路明细与负荷类注入（info 的支路统计/全网平衡校验明确置 None，
   不假装有）；完整读取用于快照/评估/回放。*等价性：light 与完整模式的
   逐步观测向量与奖励逐位一致（34Bus_PV 与真实 UK 双重锁定）。*
3. **SMART-DS compile 期全年求解派生注释**（第三条 FilePatch，manifest 登记
   原始/补丁哈希）：master 的 `Solve mode=yearly …35040` 在每次会话 reset
   重放（rdt1534 实测 278s），且遗留 InvControl 中间状态偏离模型文件初值；
   注释后初始状态回到模型文件定义。rdt1534 剩余 reset≈20s 为 180 条
   LoadShape 曲线（各 35,040 点 CSV）的引擎解析，属数据加载成本（记录，
   未优化）。

**优化后对照**（同口径重测）：

| 算例 | 单步：前 → 后 | read_state：前 → 后 | 构造：前 → 后 |
| --- | --- | --- | --- |
| 9500Node + light | 288 → **103 ms（-64%）** | 200 → 11.9 ms | 1.0 s → 1.0 s |
| UK + light | 4,095 → **1,435 ms（-65%）** | 2,865 → 116 ms | 15.4 → 11.4 s |
| rdt1534（补丁后） | 163 → 168 ms | 84 → 80 ms | **278 → 42 s（-85%）** |
| p4udt4（补丁后） | 3.8 → 3.9 ms | 2.0 → 2.1 ms | 10.0 → **4.4 s** |

UK 单步剩余构成：apply 482 ms（逐负荷 Edit，任务 2 记录的批量化方向）+
solve 126 ms + probe 113 ms + read 116 ms + 观测/奖励/日志 597 ms
（VVC 奖励逐母线越限循环 37k 次——下一优化点，已记录未做）。
参考策略前向（6–472 智能体单 batch）：0.02–0.8 ms/步，非瓶颈。

## 4. 局部 actor 观测与集中 critic（CTDE）

- `observation_mode: global|local`（vvc/smartgrid 配置，默认 global=旧行为
  逐字节不变）：local 模式下 **actor 观测=每智能体局部向量**，集中 critic
  的共享状态保持全网 `wrap_obs`（`share_observation_space` 与
  `observation_space` 解耦）；观测投影（`observation_neighborhood`）仍作用于
  critic 状态。manifest 记录 `observation_mode` 与逐智能体局部 schema
  （`local_observation_schema`，checkpoint/回放按 sha256 绑定拒绝旧模型，
  既有机制）。
- 局部向量组成（`control_env.agent_local_observation`）：自身母线相电压
  （≤3，pad）+ k 跳邻域电压 5 元组（min/max/mean/低/高越限比，
  `local_observation_neighborhood` 默认 1 跳）+ 日内时间 sin/cos + 回合进度
  + 设备自身特征（PV：P/Pmpp、Q/kVA、可用功率/Pmpp、正/负无功余量、前步
  有功分数与无功命令；电容/调压/储能/SVG 对应自身状态）。维度按类型固定：
  电容/调压 12、储能/SVG 13、PV 18。
- **信息边界可测试**（`tests/test_local_observation.py`）：篡改全部邻域外
  母线电压 → 所有局部观测逐位不变（阴性）；篡改邻域内 → 必须改变（阳性）。
  局部执行不读取全网状态（构造源仅限自身连接、邻域集合与设备状态）。
- **分组共享策略**：`algo.share_param: group`（新增枚举，仅 on-policy
  runner 实现）：按 (观测空间, 动作空间) 签名分组，组内共用一个网络实例
  （如 472 台 PV 一个网络而非 472 份），组间独立；混合维度走既有异构
  buffer。旧枚举 true/false 行为不变。
- 修复既有死角：混合/连续动作空间下 runner 的可用动作提取只看第一个智能体
  导致 Box 智能体把 None 包装成数组写入未分配 buffer（全 Box 系统——如
  p4udt4 全 PV——首次触发）；改为逐智能体判定。

## 5. 逐级运行（固定策略与短训练；节点规模 × 受控设备数）

| 级别 | 电网节点 | 受控设备 | 运行内容 | 结果 |
| --- | --- | --- | --- | --- |
| 小馈线 p4udt4 | 156 | 6 PV | 24h 四模式固定策略（任务 2）+ **480 步 2 线程 HAPPO 短训练（local+group，CPU）** | 见 §6 |
| 中型 rdt1534 | 4,173 | 472 PV | 固定动作 12 步 ×3（基准） | 0 失败，168 ms/步 |
| 9500Node（旧模式） | 9,549 | 30 | 既有 smoke 短训练（全局观测，240 步 10 updates） | 通过，FPS 2，检查点保存 |
| UK 大系统 | 112,887 | 2 调压 | 固定动作 12 步 ×2（基线/light） | 0 失败 |

短训练仅为链路验收（**不以短训练、节点数或累计奖励单独宣称安全或收敛**）。
变电站级（rhs0_1247 全站 3 馈线+输电层）与 UK 级短训练未执行（见 §8）。

## 6. p4udt4 局部观测短训练（结果）

命令：
```bash
python scripts/train.py --config configs/experiments/smartgrid_p4udt4_local_smoke.yaml \
  --set algo.share_param=group --set device.cuda=false
```

结果（CPU、2 线程、480 步=10 updates、6 PV actor 局部 18 维 + 集中 critic
全局观测、`share_param=group`）：训练完成无失败步，检查点保存至 Episode 10。
末段指标：平均步奖励 -0.74；电压违规率 5.61%（210 相节点/段；与固定策略
基线的模型固有过压一致）；PV 利用奖励 +4.38；控制成本 -3.07。**分组共享
验证**：checkpoint 中 6 个 `actor_agent*.pt` 加载后参数逐张量完全相同
（同一网络实例；文件字节差异仅为序列化时间戳）。manifest 记录
`observation_mode: local`、局部 schema 6 agents、`pv_control: rl` 全接管、
InvControl 停用 1 个。**该短训练仅为链路验收，不构成收敛或控制效果结论。**

## 7. 并行环境与 DSS 上下文独立性

既有机制核对（未改动）：每 worker 独立进程（`ShareSubprocVecEnv`，即使
单线程也子进程）× 独立 `dss.DSS.NewContext()` × 独立模型临时目录；启动时
跨 worker manifest 一致性硬校验。**区域划分仅用于控制与通信**（局部观测的
邻域即通信范围：1 跳邻域内母线电压摘要）；未做任何网络分解，耦合网络始终
整体求解——不存在"拆成独立理想电源宣称等价"的问题。

## 8. 已知限制与未完成项

1. **变电站级未接入**：rhs0_1247 全站（3 馈线 + subtransmission）需要新的
   FileSetCase 清单与 p4uhs0 级 master（含站内设备控制权设计），未做。
2. **环境级逐设备时序注入未做**（任务 2 遗留）：SMART-DS 逐设备 15 分钟
   时序目前由基线脚本驱动；训练用合成场景。**训练/测试按时间或馈线留出
   未见样本因此未实现**（现有机制仅合成数据的 train/validation split）。
3. **UK 每步 apply 482 ms（逐负荷 Edit 风暴）与奖励 597 ms（逐母线越限
   循环）未优化**——UK 单步 1,435 ms 的下一目标（批量化方向已在任务 1/2
   记录）。
4. rdt1534 每次 reset ≈20s（180 条曲线 CSV 引擎解析）：可经派生二进制
   LoadShape 缓存优化，未做；多回合训练时会放大。
5. 局部观测的 HAPPO 学习效果（相对全局观测的样本效率/最终性能）未评估
   ——本阶段只验收链路与信息边界；需要正式实验设计（含 §8.2 的留出）。
6. `available_actions` 动态掩码机制存在但恒全 1（夜间/缺测当前通过可用
   功率=0 在观测与物理层表达，不在动作空间层掩码）——设备级动作掩码留待
   需要"禁用某设备"语义时实现。

## 9. 修改文件清单

| 文件 | 内容 |
| --- | --- |
| `envs/powergrid/session.py` | 分段计时埋点；结构缓存（含编译期副作用重放）；`read_state(light)` 精简读取 |
| `envs/powergrid/control_env.py` | `light_state_read`/`observation_mode` 参数；局部观测（`agent_local_observation`/邻域/schema/dim）；info 支路统计可空 |
| `envs/powergrid/adapters.py` | CTDE 解耦：actor 局部观测/critic 全局状态；观测空间按模式 |
| `envs/powergrid/episode_runner.py` | 回放层局部观测同步 |
| `envs/powergrid/state.py` | light 模式 branches=None 的平衡校验显式 None |
| `envs/vvc/vvc/vvc_config.py`、`envs/smartgrid/base_env/env_config.py`、两 runtime_env | 新配置字段透传 |
| `runners/on_policy_base_runner.py` | `share_param=group` 分组共享；混合空间异构 buffer 判定；avail 逐 agent 提取修复 |
| `utils/config_pipeline.py` | share_param 的 group 枚举放行 |
| `utils/external_cases.py` | SMART-DS 第三条派生补丁（compile 期全年求解注释） |
| `scripts/benchmark_scale.py` | 分算例分阶段基准（新增 light 变体 preset） |
| `configs/systems/SMARTDS_RDT1534.yaml`、`_registry.yaml` | rdt1534 系统登记 |
| `configs/experiments/smartgrid_p4udt4_local_smoke.yaml` | 局部观测+分组共享冒烟实验 |
| `tests/test_scale_optimizations.py`、`tests/test_local_observation.py` | 等价性与信息边界测试 |
| `docs/scale-local-obs.md` | 本文档 |

## 10. 测试与回归

- `tests/test_scale_optimizations.py`：4 项（light=full 逐位一致 ×2、缓存
  重放一致、SOC 重置）；`tests/test_local_observation.py`：5 项（维度/
  critic 分离/边界阴阳性/跨 reset 稳定/global 不受影响/非法模式拒绝）。
- **全套测试 411 项全过**（`python -m pytest tests/ -o addopts="" -q`，
  2 分 29 秒）——含既有 9500 慢测试、任务 1/2 全部测试。
- 9500 旧模式短训练通过（`vvc_9500node_smoke.yaml`，全局观测，检查点
  保存）；p4udt4 local+group 短训练见 §6。
