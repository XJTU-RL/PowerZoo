# PR60 整合修复与验证（基于60ac9ea6）

## 工作区与基线

GLM 的推进遵循用户此前授权的任务，指定工作区为
`.claude/worktrees/sheldon123z-review-fixes-4d8af3b5`，HEAD为 `60ac9ea6`，工作区干净。
本次没有对其权限作否定，也未改写该工作区或此前本地修复原稿。

- 整合基线：`60ac9ea6`（GLM最新，包含 `9b5ceedc` 的split摘要修复及后续报告）。
- 整合分支：`sheldon123z/pr60-integrated`。
- 整合目录：`.claude/worktrees/sheldon123z-pr60-integrated`。
- 早期原稿：`sheldon123z/pr60-followup@9028105f` 的未提交补丁，保持不变。
- 整合阶段仅手工应用可兼容增量，未执行Git merge/cherry-pick/rebase。用户随后明确授权提交、推送更新现有PR60；仍禁止合并。
- 此前确认的署名及worktree规则纳入本PR；原 `sheldon123z/review-rules` 工作区保持不变。独立分支/worktree上的代码可直接提交推送，新建PR先询问。

## 重叠设计的具体处理

1. 保留GLM的 `dataset_sha256`（train）及 `dataset_sha256_validation`，不采用早期
   本地原稿的“两个split共用一个摘要”。真实p4udt4上，两种split的摘要均与
   `60ac9ea6` 原始代码计算结果逐字一致，见 `/tmp/pr60-integrated-digest-compatibility.json`。
2. 两个分片摘要共享一次内容扫描，避免重复读取曲线。每个worker核对自己split的
   期望值；当前split摘要缺失时明确失败，不悄悄跳过校验。prepare_run核对已冻结的
   train、validation及外部清单摘要，不将旧快照静默改成当前值。
3. checkpoint的 `smartds_dataset` 记录稳定的完整split摘要表、日划分及逐设备映射。
   train/validation使用同一张完整表，因此合法评估切换能通过；改变日期、映射或
   内容会失败。info中的dataset摘要仍代表当前split。
4. 保留本地的LFC mask兼容、PV引擎实际驱动恢复、惰性日期读取、可信ZIP清单迁移、
   源目录/运行副本一致性和对应测试；GLM后续报告正文全部保留，只追加整合状态入口。
5. 实质设计冲突已解决，无需覆盖任一原稿。缺少新数据绑定字段的旧checkpoint不会
   静默放行；若需迁移，应先独立核验其原始resolved配置，再补齐元数据，不改历史模型。

## 修复映射

| 审查项 | 本地结果与回归 |
| --- | --- |
| LFC 连续动作 None mask 首步崩溃 | 统一处理整体 None、逐环境 None、逐 agent None；warmup/insert/eval 共用提取逻辑。部分 worker 缺失 mask 明确报错，保留混合离散约束。真实 LFC 包装器+向量层+actor buffer 的1/2 worker首步测试通过，联合仿真内核用替身，不声称已跑 MATLAB |
| MAPPO FP 分组优势与日志索引 | 保留并验证 GLM 的成员切片、原编号日志回填；增加真实 critic `[T,N,A,1]` 布局回归。非连续组、active mask、EP/FP、参数实际更新测试通过 |
| 原生 constant-Q PV 恢复 | 用引擎 `ActiveDSSElement.ToJSON()` 的实际生效 PF/kvar 属性取代数值猜测，支持参数顺序、运行期 Edit、显式接管、reset及子集。恒Q、fixed-pf、先PF后Q/先Q后PF均回归，不再用“pf getter 小于1”推断模式 |
| SMART-DS reset 反复预处理全部日期 | episodes 改为惰性 Sequence；len不读数据，单天只生成96行并缓存只读上下文；逐行处理避免同时保存96份逐设备字典。300天配置查len为0次row，访问一天为96次，重复访问无增加 |
| 日期交集/非法日期 | 保留 GLM 日号/重复/交集/split/评估非空检查，补容器类型和未知划分键检查 |
| light 缺测报零 | 验证 GLM 的统计三态；真实light info为collected=false，计数和最大值为null；full模式保留真实数值及未评级数量 |
| benchmark 嵌套计时 | 验证 GLM 的read/branch互斥聚合与未钳零余量。此前原稿的9500 full/light固定动作实测步均333.665/104.982ms，互斥分段和282.863/58.591ms，余量50.802/46.391ms，anomaly均null；这些是并行测试期间的固定动作量测，不是训练吞吐或严格性能对比 |
| `--verify --deep` 与可信清单迁移 | 验证已有CLI；新增 `--migrate-manifest`，只接受登记大小/哈希的本地原始ZIP，展开文件集合及逐文件内容须匹配，依赖闭包完整后仅原子替换manifest。篡改、缺文件、额外文件及错误ZIP均拒绝且不改原清单。真实UK在临时副本迁移后深检ok，源manifest未改 |
| 真实数据、划分及映射冻结 | 保留GLM真实内容摘要和外部manifest记录；修复训练摘要带到验证split的冲突：沿用GLM的train/validation独立摘要，checkpoint绑定完整摘要表、日划分和设备映射。供数绑定会话临时副本，源目录修改不影响已建立供数；编译副本再次校验冻结摘要；拒绝已有冻结值被prepare_run静默覆盖。physics.manifest显式记录日划分、逐设备映射和摘要，并参与checkpoint一致性检查 |

说明：官方 SMART-DS 的 dss_folder 为布局根，原有 model_sha256 已覆盖其中的
profiles/solar_data；本修复没有把这些曲线描述成原本完全未受哈希保护。

## 本轮验证与限制

整合后的59项定向测试通过，包含GLM原有split不等断言、本地训练/评估manifest一致性、
验证摘要错误或缺失拒绝、两个split内容扫描只一次等回归。本轮20个相关测试文件去重执行，371 passed、0 skip、185.58s；覆盖配置、LFC契约、
MAPPO、PV、真实时序、外部算例及既有六类物理任务。末次追加的旧checkpoint拒绝断言
另做定向复核，日志见下文。早期原稿的测试结果不冒充本轮结果。

环境为Mac CPU、Python3.12.8、DSS-Python0.15.7、torch2.9.1。共享环境pip check仍有缺包
与锁定版本冲突，未修安装环境；MATLAB完整联合仿真、Windows、CUDA和长规模训练不在
本轮验证范围。LFC回归使用真实包装器/向量层/actor buffer，物理内核用替身，不能当作
MATLAB联合仿真通过。

外部源数据经被忽略的data/external符号链接只读引用；仿真使用DSSSession临时副本，
UK可信迁移使用临时副本。主工作区及两份源工作区的HEAD、status、diff摘要和新增文件哈希
在整合前后对比，证据为 `/tmp/pr60-integration-source-snapshots.json`。

## 合并前建议

- 先审阅相对60ac9ea6的整合增量，不叠加旧的9028105f补丁，避免重复应用或覆盖split修复。
- PR远端旧head及其绿色CI不能替代本地新补丁验证；用户另行授权提交/推送后，应更新PR正文
  并重跑目标Python3.10、依赖锁定版本及平台CI。
- 原始model_sha256已经覆盖官方SMART-DS布局根的曲线文件；这里补齐的是实际数据划分、
  映射、来源及重放记录，不声称原先曲线完全未受哈希保护。
- 短训练/潮流收敛不等于控制收益达标、长训练收敛或跨场景泛化。

## 常用复现入口

```bash
python -m pytest tests/test_pr60_regressions.py tests/test_real_timeseries.py \
  tests/test_powergrid_session.py tests/test_mappo_group_fp.py \
  tests/test_external_cases.py tests/test_scale_optimizations.py \
  -o addopts="" -q -rs
python scripts/check_runtime.py --train
python -m pip check
```

本轮日志前缀 `/tmp/pr60-integrated-`；临时日志可能被系统清理，长期回归保留在tests。

## 收尾验证记录

- `/tmp/pr60-integrated-regression.log`：371 passed、0 skip（185.58s）。
- `/tmp/pr60-integrated-targeted.log`：整合早期59项定向通过。
- `/tmp/pr60-integrated-runtime.log`：POWERZOO_RUNTIME_OK，双DSS绑定、1/2 worker、HAPPO24步。
- `/tmp/pr60-integrated-realtrain.log`：真实p4udt4 HAPPO group 24步、day2留出评估完成。
- `/tmp/pr60-integrated-restore.log`：上述checkpoint恢复后8步及评估完成；保存的physics.manifest
  含train/validation摘要表、36个负荷和6个PV的明确曲线映射。
- `/tmp/pr60-integrated-digest-compatibility.json`：两个分片摘要分别与60ac9ea6逐字相等。
- `/tmp/pr60-integrated-pip.log`：依赖检查失败，含h5py/imageio缺包、torch2.9.1与项目2.6.0
  锁定不符及共享环境其他约束冲突；未安装或调整依赖。
- `/tmp/pr60-integrated-final.log`：收尾分片/旧checkpoint安全性2项定向复核通过（17.43s）。
- `git diff --check`、Python AST、Markdown链接及完整补丁语法检查；源工作区保全摘要检查。

本轮没有重新执行长规模472PV训练或MATLAB/CUDA验收，也没有将371项结果扩大为全仓库通过。
如后续GLM继续在指定worktree新增提交，应以新的HEAD再次核对本补丁；不要把基于60ac9ea6的
增量盲目覆盖到新版本。本轮结束前指定worktree仍为干净60ac9ea6，未发现未解决的设计冲突。

## PR60 最终更新验证

用户已授权提交并推送更新现有PR60，不授权合并。GLM的37b7494c至9028105f七项修复及
后续30aa2129、9b5ceedc、60ac9ea6完整保留；本次增量及行为变化见上文，不归为重复开发。

最终本地验证分两组：tests排除tests/envs为482 passed，tests/envs为56 passed，均0 skip；
两组目录不重叠，合计538项。命令均为python -m pytest，禁用覆盖率附加选项，并明确重定向
日志至/tmp。日志 `/tmp/pr60-release-full.log`、`/tmp/pr60-release-envs.log`。
另外复核CI语法/未定义名称检查，目标平台验证以本次新提交的实际CI结果为准。

PR的存量源分支保留原名，避免重命名导致GitHub关闭打开的PR；没有新建工具前缀分支。
新提交author/committer为sheldon123z，推送用经gh当前用户接口核验的本人凭据，未改全局身份。
