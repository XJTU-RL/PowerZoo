# PowerZoo 开发约定

- 使用中文沟通。新增 Python 代码使用 4 个空格缩进，支持 Python 3.10。
- `main` 保持稳定；所有代码、配置和文档修改在独立 worktree 的任务分支完成。
- 分支使用 `codex/<topic>`，worktree 放在本仓库 `.Codex/worktrees/<分支名将 / 替换为 ->`，并忽略该目录。
- 开始任务前检查 `git status`、`git worktree list` 和已有分支，优先继续对应任务的 worktree。
- 不覆盖用户修改，不自动合并或推送。`fea/vvc` 是独立研发分支，不合并进 `main`。
- Windows 使用 `environment.windows.yml` 或 `scripts/setup_windows.ps1`。依赖版本以
  `requirements.txt` 为准；Windows 完整约束记录在 `constraints/windows-py310.txt`。
- 改动安装、路径、OpenDSS 接口或多进程时，运行 `python -m pip check` 和
  `python scripts/check_runtime.py --train`，CUDA 版本额外使用 `--require-cuda`。
- 运行相关 pytest 测试；报告失败和跳过的原因，不把 skip 当作运行成功的证据。
- 仿真检查使用临时目录或结果目录，避免覆盖版本控制下的 DSS 模型和研究数据。
- 环境依赖修复与强化学习奖励、动作、观测、参数流程的重构分开提交和验收。
- 具体流程见 `docs/development.md`。
