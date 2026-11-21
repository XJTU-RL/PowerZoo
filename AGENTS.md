# PowerZoo 开发约定

- 使用中文沟通。新增 Python 代码使用 4 个空格缩进，支持 Python 3.10。
- 主工作区保留 `main` 且只读；所有代码、配置、文档和 CI 修改在独立 worktree 的任务分支完成。同一任务继续其已有 worktree，不同任务分别建立分支和 worktree。主目录只做只读检查或确认干净后的快进同步，不直接提交开发改动。
- 新分支使用 `sheldon123z/<topic>`，不得使用 `codex/` 前缀；新 worktree 放在本仓库 `.claude/worktrees/<分支名将 / 替换为 ->`，并忽略该目录。已有分支和 worktree 保留原名，不为应用新规则自动重命名。
- 后续提交的 Git author 和 committer 均使用 `sheldon123z <zhengxiaodongfl@gmail.com>`；不添加工具或机器人署名、`Co-authored-by` 或生成工具标记。只在仓库本地或单次命令设置身份，不修改 Git 全局配置，不重写历史提交。提交前检查 `git var GIT_AUTHOR_IDENT` 和 `git var GIT_COMMITTER_IDENT`，防止环境变量覆盖。
- 推送认证账号与提交署名分别核验：GitHub 用户应为 `sheldon123z`，不得把组织名 `XJTU-RL`、本地显示名或连接器身份当作 Git 推送凭据的证明。凭据不可用或平台强制前缀/署名时如实报告，不冒用身份或虚构已覆盖限制。
- 开始任务前检查 `git status`、`git worktree list`、已有分支和远端基线，优先继续对应任务的 worktree；命令通过绝对工作目录或 `git -C <worktree>` 锚定，核对当前分支后再修改。更新已有 PR 前检查最新 head、目标分支和他人增量，禁止强推或擅自重写历史。
- 清理只针对自己的任务 worktree；先检查状态、未跟踪文件和保存情况，不强制丢弃改动。禁止在主工作区执行 `git clean -ffdx`，任何清理先预览，不能删除其他在途 worktree。
- 不覆盖用户修改。新写或修复的代码及配套更改在独立任务分支和 worktree 上，可直接提交、推送；创建新 PR 前先询问用户，更新已有 PR 按已授权范围继续。不得自动合并；`fea/vvc` 是独立研发分支，不合并进 `main`。
- Windows 使用 `environment.windows.yml` 或 `scripts/setup_windows.ps1`。依赖版本以
  `requirements.txt` 为准；Windows 完整约束记录在 `constraints/windows-py310.txt`。
- 改动安装、路径、OpenDSS 接口或多进程时，运行 `python -m pip check` 和
  `python scripts/check_runtime.py --train`，CUDA 版本额外使用 `--require-cuda`。
- 按改动范围运行相关 pytest；公共训练器、状态恢复、数据契约或依赖等影响面大时扩大验证。报告失败和跳过原因，不把 skip 当作成功；纯文档改动检查内容、链接与差异，无需无变化地重复全套训练。
- 数据来源、划分、映射与指纹使用实际消费入口的同一事实源；模拟数据与真实数据分开标注，缺测不得伪造为零。固定策略、真实训练、checkpoint 恢复和研究效果分别报告，不通过修改断言或基线掩盖缺陷。
- 提交信息与 PR 标题使用中文简述；PR 正文包含“变更摘要”“Mock 与真实数据状态”“验证记录”“风险与回滚”，明确实际通过、失败、未运行及回滚方式。
- 仿真检查使用临时目录或结果目录，避免覆盖版本控制下的 DSS 模型和研究数据。
- 环境依赖修复与强化学习奖励、动作、观测、参数流程的重构分开提交和验收。
- 涉及开发方向、接入范围或验收状态的变更时，同一任务分支在 `docs/development-roadmap.html` 对应泳道追加一个短节点。区分讨论、计划、调整、候选验收与合并，保留历史，不记录猜测完成百分比或私有链接、数据和路径；无实质进展无需更新。
- 具体流程见 `docs/development.md`。
- GLM 继续本轮工作前，先读 `docs/pr60-followup-validation.md` 的整合结果、验证证据与剩余边界；`docs/review-remediation-4d8af3b5.md` 是历史复审清单，不代替当前授权和验收状态。
