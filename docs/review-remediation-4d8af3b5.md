# 4d8af3b5 复审后的修复建议与验收

> 以下保留当时的修复计划与操作限制作为历史证据。当前实现、验证和剩余边界见
> [整合修复与验证](pr60-followup-validation.md)；后续操作授权以最新AGENTS.md及用户指示为准。

本清单针对 `codex/scale-local-obs` 的历史提交
`4d8af3b5ac0af3b34494e635d292b2a37e200d2e`，不表示这些问题已修复。
本次只更新规则与修复建议，未修改业务代码、设置 Git 身份、提交或推送。
后续新分支和署名遵循 [AGENTS.md](../AGENTS.md) 与
[开发流程](development.md)：新分支 `sheldon123z/<topic>`，author/committer
为 `sheldon123z <zhengxiaodongfl@gmail.com>`；存量分支不自动重命名。

## GLM 执行顺序与交付

1. 先检查实际 worktree、分支、HEAD、未提交修改与本地指令。当前规则草案在
   `sheldon123z/review-rules`，尚未提交或合并，其他既有 worktree 不会自动同步。
   不切换其他会话工作目录、不覆盖用户修改；需要新分支时使用本人用户名规则。
2. 先锁定失败复现，再修 P1；随后处理留出、统计、计时、完整性及数据冻结。
   每项分别说明根因、最小修改、针对性测试、实际输出与剩余限制。
3. 环境依赖问题独立处理，不与 RL 语义/奖励/动作重构混在一起；本机共享环境
   `pip check` 已有冲突，不能把测试通过写成整个依赖环境通过。
4. 仿真在临时副本运行，产物放临时/结果目录，不覆盖原始外部模型或研究数据。
   缺数据、权限失败、skip 和原生异常都如实报告，不把固定策略成功当作学习效果。
5. 修复后再更新 `docs/rework-review-fixes.md`；逐项标记已验证、失败或未验证。
   不把本清单本身当作提交、推送、合并或删除历史的授权。

## 1. P1：MAPPO 分组 FP 优势串位与日志错配

位置：`runners/on_policy_ma_runner.py:56–61`，
`algorithms/actors/mappo.py:212–215`，`common/base_logger.py:264–267`
（行号以复审提交为准）。

- 根因：组内 buffer 子集配全队 advantages，训练器却使用组内下标取优势。
  组 `[0,2]` / `[1]` 时，agent2 收到 agent1 优势，agent1 收到 agent0 优势。
- 修复：FP 按成员索引切出 `advantages[:, :, members]`，EP 保持共享优势语义；
  训练信息预分配并按实际 agent ID 回填，不按组顺序直接拼接。
- 验收：用非连续组 `[0,2]` / `[1]`、各 agent 明显不同的优势和 active mask，
  检查实际 buffer 生成器收到的优势；验证日志回填 `[A,B,A]`，而非 `[A,A,B]`。
  分别覆盖 FP、EP、全队共享及不共享路径。实际短训练比较更新前后张量，证明
  每个独立组都被更新；“组间 checkpoint 不同”不能单独证明发生过更新。
- 已确认边界：本次 EP 的 34Bus_PV 三组成员数2/8/9，两次更新均实测参数变化；
  不应为修 FP 破坏这条已通过路径。

## 2. P1：PV 探测恢复漏掉模型原生 constant-Q 模式

位置：`envs/powergrid/session.py:326–372`。

- 根因：仅通过 `session.apply` 写过 Q 的设备进入 `_pv_explicit_q`；DSS 模型
  本身以 `kvar=10` 定义的设备被误判为 fixed-pf，恢复时写 pf 丢掉恒无功模式。
- 修复：在初始化/发现及运行控制权变更时保留真实驱动模式；探测恢复原驱动变量，
  不能仅由当前 P/Q 推断模式。兼顾探测失败、reset、原生控制与量测有效性。
- 最小复现：三相12.47kV电路、负荷100kW/20kvar，PV 100kVA/Pmpp100、
  irradiance=.4、`kvar=10`、cutin/cutout=0；先求解、探测、再降辐照为.2。
  复审结果探测前约40kW/10kvar、探测后40/0、低辐照20/0；不探测对照20/10。
- 验收：全量及子集探测后 constant-Q 仍为10kvar；模型 fixed-pf=-.95 的 Q/P
  随辐照保持；经 apply 显式接管 Q 仍恢复；未接管设备和 InvControl 控制权
  保持既定契约；探测不推进时间或 SOC。对故障路径确认会话失败状态与恢复策略。
- 已确认边界：官方 SMART-DS fixed-pf 和 apply 显式 Q 测试已通过，继续保留回归。

## 3. P2：按天留出缺少不相交门禁

位置：`envs/powergrid/smartds_timeseries.py:278–282`，
`envs/powergrid/control_env.py:49–54`，`utils/envs_tools.py:80–85`。

- 根因：仅复制列表、检查选中 split 非空，未验证两集合不相交。
- 修复：日号必须是0..364整数（排除bool）、各列表无重复、训练与验证不相交；
  启用评估时验证留出非空，非法 split 给出明确错误，不延迟到 worker 中崩溃。
- 验收：`train_days=[0], validation_days=[0]` 必须失败；重复、负数、365、
  小数、bool、空评估留出也覆盖。合法 `[0,1] / [2]` 能训练并实际评估 day2，
  不是只检查两个列表的字符串。检查 train/eval worker 实际选日。
- 复现：真实 p4udt4 曾接受重叠配置，两个 split 都 reset 到 day000，首负荷
  均为0.7263993158146118kW；当前交付示例 `[0,1] / [2]` 本身没有重叠。

## 4. P2：未采集支路数据仍记录过载数0

位置：`envs/powergrid/control_env.py:808–817`，
`scripts/check_smartds.py:110–112`；检查 `scripts/benchmark_scale.py` 的输出契约。

- 修复：明确区分未采集、已采集但无额定值、已采集且无过载。未采集时最大值和
  计数为 null，并带 `branch_stats_collected=false`；聚合不得用 `or 0` 抹掉缺失。
  已采集但无可评级支路时，不得宣称整个网络没有过载。
- 验收：真实 light/full 两模式；空支路、全无额定值、部分无额定值、有过载和
  无过载帧；检查逐步 info、脚本 JSON 和摘要均不把缺失写成0或安全结论。
- 复现：light info 为 max=null、overload_count=0、unrated_count=0，缺 collected
  标记。报告声称的脚本 null+false 当前并未交付；基准脚本原本没有过载字段。

## 5. P2：性能读取分段仍重复计时

位置：`envs/powergrid/session.py:532–534,595–597,659`，
`scripts/benchmark_scale.py:171–197`。

- 根因：read_state 父计时包含 branch_state，基准将父子再相加，负余量钳零。
- 修复：定义互斥分段，或明确父子包含关系并从聚合中去重；保留原始余量，
  发现明显不守恒时显式报错/诊断，不用 `max(other_ms,0)` 掩盖。
- 验收：确定性嵌套计时测试，probe 内 solve 仍不重复；34Bus、9500 full/light
  实际基准至少数步，剔除预热后段和在合理测量误差内不大于墙钟耗时。
  报告机器、步数、口径，不把固定动作耗时当训练吞吐。
- 复现：9500 full步均295.967ms、分段和392.845ms；read204.990ms包含
  branch145.897ms。probe期间solve增量0这一已修路径应继续通过。

## 6. P2：UK 深检 CLI 与现有数据迁移未完成

位置：`scripts/download_external_case.py:142,157–158`，
`utils/external_cases.py:623–684`。

- 修复：增加 `--deep` 并传入verify；明确它与 `--verify` 的组合及非zip行为。
  对旧manifest提供明确迁移/重下载流程，不把当前未知文件直接自签为可信版本。
  在临时目录从已核对的上游压缩包重建清单，核验后由用户决定替换现存数据。
- 验收：CLI成功/失败退出码；新下载深检成功，展开文件篡改深检失败，缺依赖快检
  失败；旧manifest深检明确失败；真实UK数据深检完成后再标“真实完整性验收”。
- 已确认边界：深检API与新文件哈希门禁有效。当前 `--verify --deep` 退出2
  （参数不存在）；本机真实UK旧manifest缺逐文件哈希，API返回unavailable。
  UK A2/B原生物理基线已通过，不能替代文件完整性证明。

## 7. P2：真实时序未统一进入训练数据冻结记录

位置：`utils/training_setup.py:35–41`，
`envs/powergrid/control_env.py:30–48`，
`envs/powergrid/smartds_timeseries.py:286–297`。

- 根因：prepare_run仍冻结合成ScenarioDataset，而worker改用SMART-DS真实数据；
  真实dataset哈希仅含路径/master/日期/步数，未绑定实际曲线内容与外部清单。
- 修复：为SMART-DS冻结真实manifest内容摘要、设备曲线映射、单位、时间基准、
  日号与split、派生补丁版本；正式worker、运行快照和重放使用同一冻结入口。
  不把未使用的合成清单登记为实际数据来源；恢复时校验摘要与映射。
- 验收：真实训练的resolved配置、来源表、info及checkpoint绑定一致；在临时副本
  修改任一负荷/PV/温度曲线或映射后，摘要改变或恢复明确拒绝；保持数据内容不变
  仅搬目录时，不应仅因路径不同误判数据版本。保留合成数据路径的既有回归。
- 已确认边界：locate_entry→verify的外部文件清单校验有效；这里的缺口是运行产物
  对实际数据版本的冻结记录，不是声称任意篡改都能绕过现有verify。

## 回归入口与验收报告

在有依赖与外部数据的环境运行，诊断输出选择临时目录；只读worktree下将日志和
合成缓存重定向到可写临时目录。不要把本机权限错误或skip解释为业务修复成功。

```bash
python -m pytest tests/test_real_timeseries.py tests/test_smartds_import.py \
  tests/test_external_cases.py tests/test_local_observation.py \
  tests/test_scale_optimizations.py tests/test_powergrid_session.py \
  tests/test_divergence_semantics.py -o addopts="" -q -rs
python -m pip check
python scripts/check_runtime.py --train
```

本清单尚未新增针对七项缺陷的仓库测试。以上已有测试须补充相应失败复现；若执行
完整tests，分别报告通过、失败、skip数量和原因。OpenDSS接口/路径变更需运行时检查，
CUDA环境额外 `--require-cuda`。FP/EP训练、真实时序训练及checkpoint恢复需要实际执行。

复审基线：77个不同针对性测试最终通过，0skip；24步真实p4udt4训练与day2评估完成，
EP三组实测更新；UK A2/B差值为0；四种固定策略各96步均收敛但仍有越限/过载。
首次RDT编译曾有一次原生回调异常，重跑未复现，原因未确定。472PV既有480步训练
产物已核实，本次未重跑规模训练或417项全套；上述证据不支持长训练收敛或泛化结论。

本机临时复审材料（会被系统清理，正式修复需将必要回归落入tests）：
`/tmp/powerzoo-glm-rereview-4d8af3b5.md`、
`/tmp/powerzoo-review-training-findings.py`、
`/tmp/powerzoo-review-physical-findings.py`、
`/tmp/powerzoo-review-benchmark.json`、`/tmp/powerzoo-review-integrity.json`。

## 可直接交给 GLM 的提示词

> 请先读取此worktree的AGENTS.md、CLAUDE.md、docs/development.md及本文件。
> 用户要求后续新分支为sheldon123z/<topic>，Git author与committer均为
> sheldon123z <zhengxiaodongfl@gmail.com>，不添加工具署名/协作者尾注；
> 推送认证另行核验本人账号并取得授权，不改全局配置、历史提交或既有分支名。
> 对照本清单七项依次锁定失败复现并做最小修复，先FP优势和PV驱动恢复，
> 再补留出门禁、缺失统计、计时、深检CLI与实际数据冻结。逐项报告根因、
> 改动、测试命令、真实输出和未验证范围，不把固定策略/短训练当收敛证据。
> 此提示词本身不授权提交、推送、合并或覆盖其他会话的未提交修改。
