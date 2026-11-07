# Windows 本机验证记录

验证日期：2026-10-27。源码基于 `4d0c0f610b57d74c9d4f99f8c203593f25c1921a`，
修改位于 `codex/windows-environment`，工作目录为
`D:\Exps\PowerZoo-worktrees\windows-environment`。

## 已安装环境

| 项目 | 本机结果 |
| --- | --- |
| Conda 环境 | `powerzoo-win` |
| Python | `D:\Development\CondaEnvs\powerzoo-win\python.exe`，3.10.22 x64 |
| GPU | NVIDIA GeForce RTX 3080，12 GB，驱动 610.88 |
| PyTorch | `2.6.0+cu124`，CUDA 12.4，实际使用 `cuda:0` |
| NumPy / SB3 | 1.26.4 / 2.6.0 |
| DSS-Python / 原生后端 | 0.15.7 / 0.14.5 |
| OpenDSSDirect.py | 0.9.4 |
| 中文编码 | Conda 环境变量 `PYTHONUTF8=1` |

完整包版本记录在 `constraints/windows-py310.txt`。该文件只记录名称与版本，
不包含本机 wheel 路径或 Conda 构建机路径。Torch 约束保留 `2.6.0`，
允许安装脚本明确选择官方 `+cpu` 或 `+cu124` 构建。
安装使用 conda-forge 创建环境，未接受 defaults 频道的额外服务条款。

## 已执行检查

| 检查 | 结果 |
| --- | --- |
| 完整安装脚本 `scripts/setup_windows.ps1 -TorchFlavor cu124` | 成功；在已创建环境上验证依赖安装、editable 安装及运行检查 |
| `python -m pip check` | 无依赖冲突 |
| DSS-Python 与 OpenDSSDirect 原生求解 | 分别新建上下文并求解三相负荷电路，均收敛 |
| CPU / CUDA 张量计算 | 均通过实际矩阵计算 |
| VVC IEEE13，单 worker | reset 后执行 4 步，观测形状 `[1, 5, 46]`，观测及奖励有限 |
| VVC IEEE13，两个 spawn worker | reset 后执行 4 步，观测形状 `[2, 5, 46]` |
| CPU HAPPO | 24 步，完成一次更新并释放日志和环境资源 |
| CUDA HAPPO | 24 步，完成一次更新并释放日志和环境资源 |
| README 普通训练入口 `train.py` | 240 步、10 次更新，保存检查点后正常退出 |
| Windows 回归及现有配置测试 | 57 passed，0 failed，0 skipped |
| `python -m build --sdist --no-isolation` | 成功构建源码包 |

VVC 与 CLI 检查使用临时复制的 `node_systems/13Bus` 及其自带负荷曲线，
临时路径包含空格。CLI 检查通过 `--system_name` 指定这份副本，其余参数采用 README
短训练命令。没有补造负荷、PV 或温度数据，没有修改奖励函数或电气参数。

运行日志保存在本机 `results/validation/`，该目录按仓库规则不提交：

- `setup-windows.log`：完整 PowerShell 安装脚本及最终检查。
- `windows-runtime.log`：原生接口、多进程、CPU/CUDA 短训练。
- `pytest-windows.log`：57 项测试的结果。
- `cli-windows.log`：正常 CLI 的 240 步训练。
- `build-windows.log`：源码包构建。

## 本次修复的启动问题

1. 仓库含带字面 `*` 的文件名，Windows clone 后 checkout 失败；改为合法文件名。
2. 旧 VVC 路径查找依赖目录名为 `PowerZoo`，在 Windows 任意名称 worktree 中可能死循环；
   改为复用项目已有路径解析器。
3. DSS `compile` 命令未引用文件路径，含空格的目录无法正常加载；补充引号。
4. IEEE13 主 DSS 引用了仓库缺失的母线坐标文件，真实编译报错；从标准算例恢复，
   来源记录在 `node_systems/13Bus/README.md`。
5. SmartGrid 无条件导入 Unix 专用 `fcntl`，导致 Windows 配置模块也不能导入；
   改用 `filelock` 原生跨平台锁，并验证另一进程持锁时不会越过锁写文件。
6. DSS worker 输入文件未及时关闭；全局训练诊断对象还会重新打开上次已关闭的日志。
   分别修复文件生命周期与 writer 引用，连续 CPU/CUDA 训练及临时目录清理已通过。
7. `requirements.txt`、`setup.py` 与 Conda 文件互不一致，且包含机器相关 Linux 构建；
   收敛运行依赖、补齐代码实际导入的包，并新增 Windows 安装入口及 CI。

## 验证边界与后续工作

- 原主目录仍保留 `main` 的 clone 状态；当前使用上面的修复 worktree。
  在 Windows 文件名修复合入 `main` 之前，从旧 `main` 新建 worktree 仍会遇到检出错误。
  分支流程见 `docs/development.md`。
- SmartGrid 的 episode 生成器要求 `month_*_0000_to_2359.csv`、
  `irrad_*_0500_to_1659.csv` 和 `temperature_*_0500_to_1659.csv`；
  仓库自带的旧 `LoadShape` 文件不等于这套输入。完整 SmartGrid 训练尚未验收。
- 单智能体旧底层环境与新版 `LoadProfile`、`Circuits` 构造参数不一致，
  需要下一阶段修复接口与路径；本次未将其描述为已可训练。
- `useS=True` 的 CuPy 敏感度路径未安装和验收；基线使用 `useS=False`。
  Gradio / Plotly 界面、W&B 和社区检测插件也不在本次基线中。
- pytest 日志中出现一次 DSS 原生 DLL 初始化期间的 Windows 异常堆栈；
  该次进程退出码为 0，57 项测试及独立原生求解均通过。若之后出现非零退出码，仍应按失败排查。
- 短训练会输出解释方差偏低的诊断信息；单元素张量的样本标准差还会让诊断日志出现
  NaN 警告，对应 `utils/happo_diagnostics.py` 的统计实现。这里仅验收运行链路，
  不据此判断学习质量，也未修改算法统计或训练语义。
- Linux、完整测试集、所有算法与系统组合尚未执行。新增 Windows CI 已写入仓库，
  远端任务需提交推送后才能运行，本机通过不等于远端 CI 已运行。

复验命令：

```powershell
Set-Location D:\Exps\PowerZoo-worktrees\windows-environment
conda activate powerzoo-win
python -m pip check
python scripts/check_runtime.py --train --require-cuda
python -m pytest tests/test_windows_runtime.py tests/test_config_pipeline.py tests/test_unified_config.py -o addopts="" -q
```
