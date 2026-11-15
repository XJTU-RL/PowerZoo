# PowerZoo 开发纪律适配依据

按用户要求，参考本机实际电力现货交易平台工作区
`/Users/xiaodongzheng/exps/dljy-workspace/CLAUDE.md` 及
`.claude/rules/common-block.md`，对应本地规则仓提交
`fcde0b7723c4c08bfcffbb066ab22d579df26dfa`。本次只读取该平台，未修改其文件。

采用并适配：主工作区只读；一任务一分支/worktree；命令锚定目录；更新前核对
远端和他人增量；只清理自己的 worktree；验证按影响面扩大；真实数据与模拟明确
区分；规则保持稳定，具体运行证据放验收文档；PR 用四章节说明变更、数据、验证、
风险与回滚。PowerZoo 的通用事实源为 AGENTS.md，CLAUDE.md 保留入口及专属工作流。

保留 PowerZoo 既有 Python/Windows/OpenDSS 验证、研究数据保护和 fea/vvc 隔离要求。
按用户较新的明确授权：独立任务分支和 worktree 可直接提交、推送；新 PR 先询问；
已有 PR 按授权更新；不自动合并。新分支使用 sheldon123z/，提交 author/committer
均为 sheldon123z <zhengxiaodongfl@gmail.com>，不加工具或协作者尾注。

未照搬平台的六仓结构、项目看板与 issue 编号、端口、Slack 通知、地区业务规则、
自动开 PR/合并、自动同步 main 的 merge 指令或跳过 hooks 的做法。它们不是本仓
授权。已有 PR 的存量分支保留原名，不为新命名规则关闭重建 PR。

具体实现与验证状态见 [PR60 整合记录](pr60-followup-validation.md)。
