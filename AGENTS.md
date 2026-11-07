# PowerZoo 开发约定

- 使用中文沟通。新增 Python 代码使用 4 个空格缩进，支持 Python 3.10。
- `main` 保持稳定；所有代码、配置和文档修改在独立 worktree 的任务分支完成。
- 分支按任务命名，例如 `feat/<topic>`、`fix/<topic>` 或 `docs/<topic>`。按本机工作区约定放置 worktree，并忽略其目录。
- 今后新增、修改或发布的项目内容统一以 PowerZoo 和任务本身为主。清除涉及内容中编码智能体的品牌标识、署名、生成声明和工具专属标记，包括代码、注释、测试、文档、配置、界面、素材和发布说明。
- 分支名、提交信息、Issue、PR、审查评论及自动化工作流使用项目和任务术语。发布前检查生成或复制的内容，移除随工具带入的徽章、水印、链接、签名和归属元数据。
- 开始任务前检查 `git status`、`git worktree list` 和已有分支，优先继续对应任务的 worktree。
- 不覆盖用户修改。改动完成、验证通过并提交后，可直接推送任务分支。创建 PR 或合并仍需明确授权。
- `fea/vvc` 是独立研发分支，不合并进 `main`。
- Windows 使用 `environment.windows.yml` 或 `scripts/setup_windows.ps1`。依赖版本以
  `requirements.txt` 为准；Windows 完整约束记录在 `constraints/windows-py310.txt`。
- 改动安装、路径、OpenDSS 接口或多进程时，运行 `python -m pip check` 和
  `python scripts/check_runtime.py --train`，CUDA 版本额外使用 `--require-cuda`。
- 运行相关 pytest 测试；报告失败和跳过的原因，不把 skip 当作运行成功的证据。
- 仿真检查使用临时目录或结果目录，避免覆盖版本控制下的 DSS 模型和研究数据。
- 环境依赖修复与强化学习奖励、动作、观测、参数流程的重构分开提交和验收。
- 具体流程见 `docs/development.md`。
