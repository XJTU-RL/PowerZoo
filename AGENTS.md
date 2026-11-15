# PowerZoo 开发约定

- 使用中文沟通。新增 Python 代码使用 4 个空格缩进，支持 Python 3.10。
- `main` 保持稳定；所有代码、配置和文档修改在独立 worktree 的任务分支完成。
- 新分支使用 `sheldon123z/<topic>`，不得使用 `codex/` 前缀；新 worktree 放在本仓库 `.claude/worktrees/<分支名将 / 替换为 ->`，并忽略该目录。已有分支和 worktree 保留原名，不为应用新规则自动重命名。
- 后续提交的 Git author 和 committer 均使用 `sheldon123z <zhengxiaodongfl@gmail.com>`；不添加工具或机器人署名、`Co-authored-by` 或生成工具标记。只在仓库本地或单次命令设置身份，不修改 Git 全局配置，不重写历史提交。提交前检查 `git var GIT_AUTHOR_IDENT` 和 `git var GIT_COMMITTER_IDENT`，防止环境变量覆盖。
- 推送认证账号与提交署名分别核验：GitHub 用户应为 `sheldon123z`，不得把组织名 `XJTU-RL`、本地显示名或连接器身份当作 Git 推送凭据的证明。凭据不可用或平台强制前缀/署名时如实报告，不冒用身份或虚构已覆盖限制。
- 开始任务前检查 `git status`、`git worktree list` 和已有分支，优先继续对应任务的 worktree。
- 不覆盖用户修改。新写或修复的代码及配套更改在独立任务分支和 worktree 上，可直接提交、推送；创建新 PR 前先询问用户，更新已有 PR 按已授权范围继续。不得自动合并；`fea/vvc` 是独立研发分支，不合并进 `main`。
- Windows 使用 `environment.windows.yml` 或 `scripts/setup_windows.ps1`。依赖版本以
  `requirements.txt` 为准；Windows 完整约束记录在 `constraints/windows-py310.txt`。
- 改动安装、路径、OpenDSS 接口或多进程时，运行 `python -m pip check` 和
  `python scripts/check_runtime.py --train`，CUDA 版本额外使用 `--require-cuda`。
- 运行相关 pytest 测试；报告失败和跳过的原因，不把 skip 当作运行成功的证据。
- 仿真检查使用临时目录或结果目录，避免覆盖版本控制下的 DSS 模型和研究数据。
- 环境依赖修复与强化学习奖励、动作、观测、参数流程的重构分开提交和验收。
- 具体流程见 `docs/development.md`。
- GLM 继续本轮工作前，先读 `docs/pr60-followup-validation.md` 的整合结果、验证证据与剩余边界；`docs/review-remediation-4d8af3b5.md` 是历史复审清单，不代替当前授权和验收状态。
