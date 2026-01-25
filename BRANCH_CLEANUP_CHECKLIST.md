# 分支清理标记清单 (Branch Cleanup Checklist)

这是一个快速参考清单，用于标记和处理所有分支。

---

## 🔴 立即删除 (DELETE IMMEDIATELY)

| 分支名 | 状态 | 原因 | 操作命令 |
|--------|------|------|----------|
| `claude/training-config-frontend-wXO8f` | 已合并 | PR #32已在2026-01-12合并到main | `git push origin --delete claude/training-config-frontend-wXO8f` |

---

## 🟢 建议合并 (RECOMMENDED TO MERGE)

### CI/CD 更新 (低风险)

| PR# | 分支名 | 更新内容 | 合并命令 |
|-----|--------|----------|----------|
| #27 | `dependabot/github_actions/actions/checkout-6` | actions/checkout v4→v6 | `gh pr merge 27 --merge` |
| #26 | `dependabot/github_actions/actions/setup-python-6` | actions/setup-python v5→v6 | `gh pr merge 26 --merge` |
| #25 | `dependabot/github_actions/codecov/codecov-action-5` | codecov/codecov-action v4→v5 | `gh pr merge 25 --merge` |

### Python 依赖更新 (低风险)

| PR# | 分支名 | 更新内容 | 合并命令 |
|-----|--------|----------|----------|
| #30 | `dependabot/pip/numpy-gte-1.23.0-and-lt-1.27.0` | numpy版本范围扩展 | `gh pr merge 30 --merge` |
| #29 | `dependabot/pip/torch-gte-2.6.0-and-lt-2.9.0` | torch版本范围扩展 | `gh pr merge 29 --merge` |
| #28 | `dependabot/pip/torchvision-gte-0.21.0-and-lt-0.24.0` | torchvision版本范围扩展 | `gh pr merge 28 --merge` |

---

## 🟡 需要测试后合并 (TEST BEFORE MERGE)

| PR# | 分支名 | 更新内容 | 测试步骤 |
|-----|--------|----------|----------|
| #31 | `dependabot/pip/gymnasium-1.1.1` | gymnasium 0.29.1→1.1.1 (大版本跳跃) | 见下方详细测试步骤 |

### Gymnasium PR #31 测试步骤:
```bash
# 1. 切换到分支
git fetch origin
git checkout dependabot/pip/gymnasium-1.1.1

# 2. 安装依赖
pip install -e .

# 3. 运行测试
pytest tests/

# 4. 测试主要环境
python examples/multi_agent/scripts/happo_powerzoo.py --test

# 5. 如果测试通过，合并
gh pr merge 31 --merge

# 6. 返回main分支
git checkout main
```

---

## 🔒 保持独立 (KEEP INDEPENDENT - DO NOT MERGE)

| 分支名 | 目的 | 最后更新 | 保护措施 |
|--------|------|----------|----------|
| `fea/vvc` | Volt-Var Control功能独立研发 | 2025-04-09 | 设置分支保护规则 |

**重要说明**: 
- 此分支是独立的研究项目，基于不同的代码结构
- 不应合并到main，应继续独立发展
- 建议在GitHub设置中为此分支添加保护规则
- 可以定期cherry-pick main分支的重要bug修复

---

## ✅ 保持现状 (KEEP AS IS)

| 分支名 | 状态 | 说明 |
|--------|------|------|
| `main` | 活跃 | 主开发分支 |
| `copilot/evaluate-branch-merge-status` | 当前工作 | 本次分支评估任务，完成后合并删除 |

---

## 📊 统计摘要

- **总分支数**: 11个
- **需要立即删除**: 1个 (已合并但未删除)
- **建议合并**: 6个 (低风险Dependabot更新)
- **需测试后合并**: 1个 (gymnasium更新)
- **独立发展**: 1个 (fea/vvc)
- **正常活跃**: 2个 (main + 当前工作分支)

---

## 🚀 快速执行方案

### 方案A: 使用自动化脚本
```bash
cd /home/runner/work/PowerZoo/PowerZoo
./scripts/branch_cleanup.sh
```

### 方案B: 手动执行
```bash
# 1. 删除已合并分支
git push origin --delete claude/training-config-frontend-wXO8f

# 2. 批量合并低风险PR (需要GitHub CLI)
gh pr merge 27 --merge
gh pr merge 26 --merge
gh pr merge 25 --merge
gh pr merge 30 --merge
gh pr merge 29 --merge
gh pr merge 28 --merge

# 3. 手动测试并合并gymnasium PR (参见上面的测试步骤)

# 4. 清理本地引用
git fetch --all --prune

# 5. 验证结果
git branch -r --sort=-committerdate
```

---

## ⚠️ 重要提醒

1. **在合并前备份**: 虽然所有这些更新都是低风险的，但建议在合并前确认CI通过
2. **Gymnasium需要特别注意**: 这是一个较大版本升级，必须先测试
3. **fea/vvc分支**: 绝对不要尝试合并或删除
4. **定期维护**: 建议每月检查一次分支状态，及时清理

---

**创建日期**: 2026-01-25  
**维护者**: GitHub Copilot Agent  
**相关文档**: BRANCH_MANAGEMENT.md
