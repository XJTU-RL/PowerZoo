# 2026-11-06 主分支合并检查

检查基线为主分支 `4d0c0f61`，待合并开发提交截至 `1f65db4d`，共16个提交。

## 合并前修复

- 统一临时模型根目录的规范路径，防止合法DSS文件因目录别名被拒绝；新增先复现后验证的路径回归。
- 补齐123Bus引用的静态负荷文件并加入精确忽略白名单，原生回归核对额定P/Q与无时序绑定。
- 修复活动源码未定义的向量包装类与可选wandb引用。CI安装固定完整依赖、运行全部活动测试及原生运行检查，不吞掉测试失败。
- 将重复的 `huggingface_space/` 合并至 `huggingface_spaces/powerzoo/`：19个文件逐字节一致，保留12个Space应用，更新生成器与同步路径。同步显式使用main初始分支。
- 更新worktree目录约定并忽略仓库内 `.Codex/worktrees/`。

## 本轮实测

平台为macOS ARM64、Python 3.11.15、PyTorch 2.6.0 CPU、DSS-Python 0.15.7 / C-API 0.14.5。

| 检查 | 结果 |
| --- | --- |
| `python -m pip check` | 无依赖冲突 |
| `python -m pytest tests -o addopts='' -q` | 395项通过，零失败、零跳过 |
| `python scripts/check_training.py` | VVC、SmartGrid、DSR、Stackelberg、District各240步/10次更新并完成评估；顺序定价恢复回放24步 |
| `python scripts/check_runtime.py --train` | 原生双DSS接口、单/双worker和真实24步HAPPO更新通过 |
| Windows工作流列明的4组回归在本机执行 | 84项通过 |
| 活动源码flake8 E9/F63/F7/F82 | 0错误；排除归档、依赖、数据与嵌套worktree |
| `python -m build` | sdist与wheel成功构建 |
| Hugging Face应用 | 12个入口语法通过，主站19文件内容保全，工作流YAML解析及路径核对通过 |
| `git diff --check` | 通过 |

测试XML与短训练报告位于运行worktree的 `results/validation/`，不提交大体积运行输出。
本轮没有在Windows、Linux或CUDA机器重新执行；修订后的GitHub Actions尚未触发。
Space没有新增浏览器交互验收或远端部署。本轮验证不代表所有算法训练、策略收敛或所有电气场景可行。
