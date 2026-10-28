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
