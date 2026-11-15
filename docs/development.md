# 分支与 worktree 开发

主目录保留 `main`，每项开发使用独立分支和目录。worktree 共享 Git 对象和历史，
文件、索引和当前分支分别独立。不要在同一个工作目录中并行切换任务分支。

## 新建任务

新任务分支统一为 `sheldon123z/<topic>`，新 worktree 放在仓库内
`.claude/worktrees/<分支名将 / 替换为 ->`。既有分支、提交及目录保留原名，
下文历史实施记录中的旧名称不作为新任务命名模板。
以下为 PowerShell 示例，任务名称应描述实际工作。
常规新任务从已验证的 `main` 创建；Windows 修复尚未合入时，从明确依赖的任务分支创建。
不同任务分别验收。在独立任务分支和worktree上，新写或修复的代码及配套更改
可以直接提交、推送；新建PR先询问用户，不自动合并。

```powershell
git -C D:\Exps\PowerZoo status --short --branch
git -C D:\Exps\PowerZoo worktree list
git -C D:\Exps\PowerZoo worktree add -b sheldon123z/config-pipeline D:\Exps\PowerZoo\.claude\worktrees\sheldon123z-config-pipeline main
Set-Location D:\Exps\PowerZoo\.claude\worktrees\sheldon123z-config-pipeline
conda activate powerzoo-win
python -m pip install --no-deps -e .
```

`pip install -e .` 会把当前 Conda 环境的包路径指向这个 worktree。共用环境时，
切换任务后需重新执行该命令；同时开发不同依赖组合时，为各 worktree 创建不同 Conda 环境。
从项目根目录启动训练，以明确配置和输出路径。

## 提交署名与推送身份

GitHub 用户名已通过连接器的当前用户接口核实为 `sheldon123z`（用户 ID
`42455304`）；`zhengxiaodongfl@gmail.com` 来自现有 Git 配置与仓库历史提交，
并非根据用户名猜测。连接器未返回邮箱，本次未验证该邮箱在 GitHub 的验证状态。
后续新提交 author 和 committer 的姓名均使用 GitHub 用户名，邮箱沿用上述已配置邮箱。
不添加工具/机器人署名、`Co-authored-by` 或生成工具标记；提交信息描述实际改动。

仅在当前仓库本地设置，或使用单次命令配置；不修改全局 Git 配置。本规则文件的
修改不会自动改写当前 Git 配置。以后获准提交前可在任务 worktree 执行：

```powershell
git config --local user.name sheldon123z
git config --local user.email zhengxiaodongfl@gmail.com
git var GIT_AUTHOR_IDENT
git var GIT_COMMITTER_IDENT
```

worktree 的本地配置可能由同仓库共享，设置前检查现状；身份环境变量可能覆盖配置，
以上两条 `git var` 的最终结果都应为本人身份。不得为修正署名擅自 amend、rebase
或批量改写历史提交，也不为应用新命名规则自动重命名既有分支。

提交署名与推送认证是两件事。推送前核对 remote、实际 Git 传输使用的凭据/认证账号
及任务范围，账号应为 `sheldon123z`；组织 `XJTU-RL` 是仓库所有者，不是本人的用户名。
连接器登录成功不代表本机 Git/gh 凭据可用，`user.name` 也不能改变推送账号。
凭据不可用时停止推送并报告，不输出 token。遇到平台强制命名前缀、署名或服务端
合并身份时明确说明限制，不声称仓库规则能够覆盖。规则文件本身不会自动设置Git身份，
每次提交须核对最终author/committer；已有历史名称与署名保留。

更新已有PR时保留存量源分支，正常快进推送即可；不要为套用新命名规则自动重命名
打开PR的head分支，GitHub会关闭该PR。若确需改名，先说明影响并询问，不擅自关闭重建。

## 历史实施记录

本次 Windows 环境任务使用分支 `codex/windows-environment`，目录为
`D:\Exps\PowerZoo-worktrees\windows-environment`。

## 验证与提交

```powershell
python -m pip check
python scripts/check_runtime.py --train
python -m pytest tests/test_configuration_contract.py tests/test_config_pipeline.py tests/test_unified_config.py -o addopts="" -q
git diff --check
git diff
git status --short
```

CUDA 环境另运行 `python scripts/check_runtime.py --train --require-cuda`。
提交说明需包含修改原因、实际验证和剩余限制。审查通过后再合并进 `main`；
不要将独立研发分支 `fea/vvc` 合并进 `main`。清理 worktree 前确认变更已经提交并保留，
不使用强制删除丢弃未提交工作。

## 开发顺序

1. 安装与运行基线：依赖、DSS 原生库、Windows 路径、spawn 多进程、短训练。
2. RL 环境契约：单/多智能体接口、动作与观测、终止与截断、求解异常。
3. 参数流程：统一已有配置加载入口、覆盖优先级、未知参数校验和最终配置快照。

第一阶段不修改奖励函数、电气约束或训练算法语义。后续阶段分别讨论目标与验收。

## SmartGrid 数据与运行任务

从 `01926ba2` 建立 `codex/smartgrid-data-runtime`，目录为
`D:\Exps\PowerZoo-worktrees\smartgrid-data-runtime`。环境依赖不变，RL/物理模型改动在此分支单独验收。
除通用运行检查外，执行 `python scripts/check_smartgrid.py --train --require-cuda` 和
`python -m pytest tests/test_smartgrid_dataset_runtime.py -o addopts="" -q`。

## 公共训练入口整理

从 `3f47f9f4` 建立 `codex/training-entrypoints`，目录
`D:\Exps\PowerZoo-worktrees\training-entrypoints`。所有环境共用 `scripts/train.py`；
检查脚本为 `scripts/check_training.py`，归档与参数流说明见
[入口整理说明](training-entrypoints.md)。本分支继承此前修复，不自动合并到main。


## 配置精简任务

从 `1c0f050e` 建立 `codex/config-simplicity`，目录为
`D:\Exps\PowerZoo-worktrees\config-simplicity`。单次引用的训练预设合并回环境文件，
真实共享的 smoke 设置保留；字段删除与保留依据实际消费者，详见[配置说明](configuration.md)。
训练验证沿用公共检查脚本，不添加另一套长期训练入口。

## 物理层与任务重构

从 `fab44772` 建立 `codex/physics-refactor`，目录为
`D:\Exps\PowerZoo-worktrees\physics-refactor`。阶段方案见[设计](physical-consistency-plan.md)，
实际完成范围、命令、基线和已知限制见[实施记录](physics-refactor-progress.md)。
先校准SmartGrid，再逐一迁移其他任务；不将未通过电气/收益/交互验收的旧环境标为可用。

### 继续增加算例与任务

日常开发优先继续对应任务；需要新分支时，从已验收且明确依赖的提交创建
`sheldon123z/<topic>`，worktree 使用本文的新目录规则。物理层历史研发分支见上文记录。
本轮不自动合并到main；此前环境/config修复已经沿本分支提交历史继承，无需再复制旧环境目录。

当前 GLM 修订的返工清单与验收步骤见
[4d8af3b5 复审修复建议](review-remediation-4d8af3b5.md)。

仅换馈线、负荷水平、控制分组、价格或故障时，优先新增现有任务的**独立算例YAML**：

1. 在`node_systems/<case>/`保存静态DSS入口与依赖，明确原生设备、接线、额定值和控制假设。
   不在运行时重写设备定义，不在YAML重复容量表；新增设备类型先补原生响应/单位/能力边界测试。
2. 在`configs/envs_cfgs/<family>_<case>.yaml`选择DSS、数据和任务专属绑定。
   `--env <family>_<case>`选择该文件，公共解析器将其归一到任务家族；不需要改第二个系统名称映射。
   同一算例反复引用的内容才提取引用。列表整体替换，映射逐键合并，空映射不会清空父值。
3. 运行任务复用`DataConfig/ScenarioDataset`，恢复使用故障manifest；两者都通过`prepare_run`冻结路径与SHA256。
   外部数据先显式换算成对应格式，不通过截断、循环、补电压或猜测基准让环境运行。
4. 写最小实验文件，`--print-config/--explain/--validate-config`检查最终值；再做实际native基线、短训练和保存模型的恢复回放。

目标、事件或决策顺序变化才新建任务类。任务直接组合`DSSSession`，在需要时复用
`DeviceControlEnv`的固定拓扑运行流程；不要继承SmartGrid的CMDP或强迫所有任务共享同一种奖励。
配置定义集中在任务dataclass，`utils/environment_registry.py`登记配置/包装器及交互类型。
同步协作任务可接公共adapter和HAPPO，但必须满足团队收益契约；非合作顺序任务需像定价任务一样
明确阶段观测、独立收益与匹配的收集器，不能仅改`state_type=FP`或算法名称。
公共层尚未提供频率、暂态或保护协调能力，这些目标需要先建设匹配的物理模型。

当前复现命令和边界分别见[SmartGrid实施记录](physics-refactor-progress.md)、[VVC](vvc-physics.md)、
[区域调度](district-physics.md)、[恢复](dsr-physics.md)、[顺序定价](pricing-physics.md)。

## LFC（OHASAC 论文算例）开发约定

- 环境家族 `lfc`（`envs/lfc/`）：OpenDSS 配电网 + MATLAB/Simulink 输电网联合仿真，
  忠实移植自 OHASAC 论文实现；算例 `13Bus`（10 智能体）/ `34Bus`（20 智能体）。
- **MATLAB 依赖边界**：联合仿真层需本机 MATLAB R2024a~R2025b（含 Simscape
  Electrical / Specialized Power Systems）+ `matlabengine` 对应版本 + DISPLAY
  （Xvfb 即可）。无 MATLAB 的环境（CI）可正常 import 做配置与数据校验，物理
  联合仿真验收经 `python scripts/check_lfc.py --matlab` 分层执行。
- **环境防御**（worker 起引擎前的净化，见 `envs/lfc/lfc_env.py`）：
  LD_LIBRARY_PATH 复原 + 剥离 cv2 注入的 QT_*/SDL_*；LFC worker 强制 spawn。
- 数据只读：worker 构造时复制 `systems/<Bus>` 到临时工作区，生成物不落源目录。
- 算法 `ohasac`（`runners/ohasac_runner.py`）：HASAC + 次序网络（OrderAgent，
  `algorithms/actors/order_agent.py`），次序决策经基类 `_agent_order` 钩子注入；
  次序缓冲与 `train.eval_interval` 的耦合关系见 `configs/algos_cfgs/ohasac.yaml`。
