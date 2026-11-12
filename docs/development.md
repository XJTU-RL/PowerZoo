# 分支与 worktree 开发

主目录保留 `main`，每项开发使用独立分支和目录。worktree 共享 Git 对象和历史，
文件、索引和当前分支分别独立。不要在同一个工作目录中并行切换任务分支。

## 新建任务

以下为 PowerShell 示例。实际目录可以调整，分支和目录名应描述当前任务。
常规新任务从已验证的 `main` 创建；Windows 修复尚未合入时，从明确依赖的任务分支创建。
旧提交含 Windows 非法文件名，本次 `config-pipeline` 从环境修复提交 `28e25a7c` 创建。
两个任务分别提交，不自动合并或推送。

```powershell
git -C D:\Exps\PowerZoo status --short --branch
git -C D:\Exps\PowerZoo worktree list
git -C D:\Exps\PowerZoo worktree add -b codex/config-pipeline D:\Exps\PowerZoo-worktrees\config-pipeline codex/windows-environment
Set-Location D:\Exps\PowerZoo-worktrees\config-pipeline
conda activate powerzoo-win
python -m pip install --no-deps -e .
```

`pip install -e .` 会把当前 Conda 环境的包路径指向这个 worktree。共用环境时，
切换任务后需重新执行该命令；同时开发不同依赖组合时，为各 worktree 创建不同 Conda 环境。
从项目根目录启动训练，以明确配置和输出路径。

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

日常开发先继续或基于`codex/physics-refactor`的已验收提交创建新的`codex/<topic>`分支和同级worktree。
本轮不自动合并到main；此前环境/config修复已经沿本分支提交历史继承，无需再复制旧环境目录。

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
