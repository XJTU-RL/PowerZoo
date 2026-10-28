# PowerZoo

PowerZoo 是基于 Python、PyTorch 和 OpenDSS 的电力系统强化学习研究框架。
代码包含自研多智能体训练器，以及面向 Stable-Baselines3 的单智能体接口。

当前主要环境：

| 环境参数 | 代码目录 | 用途 |
| --- | --- | --- |
| `vvc` | `envs/vvc/` | 经典电压无功控制；13/34/123共用原生物理层并已短训练验收 |
| `smartgrid` | `envs/smartgrid/` | 模块化电网控制，含 PV、奖励和日志组件 |
| `dsr` | `envs/dsr/` | 配电网恢复 |
| `stackelberg_13bus/34bus/123bus` | `envs/stackelberg/` | Stackelberg 博弈控制 |
| `district_dispatch` | `envs/district_dispatch/` | 真实区域DER协同调度，原生计量与成本；HAPPO/PPO短训练验收 |

`powerzoo` 是 `vvc` 的历史别名。历史文档中的 PowerZoo_LLM 通常指 SmartGrid
相关实现，并不代表安装后自动具备独立的大语言模型服务。
算法以 `runners/__init__.py` 中的注册表为准；包含 HAPPO、MAPPO、SHOM、
MADDPG、HASAC、QMix 和两时间尺度 VVC 等。不同环境与算法的组合需要分别验证。

配置遵循“共享入口、环境专用字段、实验只写差异”。同一环境的设置与推荐训练节可放在
同一个 YAML，仅对真正重复使用的内容提取引用；未知或未实现参数直接报错。
使用与实际生效边界见[配置说明](docs/configuration.md)。

当前短训练验收不等同于全部任务的物理与学习目标正确。对DSR、District和Stackelberg进一步审查发现的
设备映射、能量时钟、目标函数与奖励契约问题，以及分阶段建设建议，见
[物理一致性与场景环境方案](docs/physical-consistency-plan.md)；SmartGrid、VVC和区域调度已迁移并验收，恢复/定价任务仍阻断旧实现训练，见[实施记录](docs/physics-refactor-progress.md)。
区域设备归属、真实P/Q计量、TOU成本与局部观测见[区域调度说明](docs/district-physics.md)。

## Windows 安装

基线为 **Windows x64 + Python 3.10 + NumPy 1.26.4 + PyTorch 2.6.0**。
使用独立 Conda 环境，所有命令从项目根目录执行。

### 一键安装与验证

先安装 Miniconda 或 Miniforge，确保当前 PowerShell 能执行 `conda`。

```powershell
git clone https://github.com/XJTU-RL/PowerZoo.git
Set-Location PowerZoo
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

上面安装 CPU 版本。NVIDIA 显卡使用 CUDA 12.4 版 PyTorch：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -TorchFlavor cu124
```

该脚本创建或复用 `powerzoo-win`，安装依赖与当前 worktree 的 editable 包，
运行 `pip check`、原生 DSS 求解、VVC 环境步进、两个 spawn worker 和一次 HAPPO 短训练。
任一步失败会返回错误，不会把失败当成成功。
`cu124` 验证还会额外完成一次 CUDA 上的 HAPPO 更新。

历史提交中存在字面文件名 `.github/instructions/*.instructions.md`，Windows 无法检出。
本次修复将它改名为 `general.instructions.md`；旧远端提交仍可能在 clone 后报告
checkout 失败。本机当前开发目录为 `D:\Exps\PowerZoo-worktrees\physics-refactor`，
分支为 `codex/physics-refactor`；环境修复单独保留在 `codex/windows-environment`。

CUDA 版本需要兼容的 NVIDIA 驱动。使用官方 PyTorch wheel 时不需要额外安装完整
CUDA Toolkit；OpenDSS 潮流求解仍在 CPU 上执行，CUDA 用于 PyTorch 网络计算。
GPU wheel 较大，下载时间通常长于 CPU 版本。

### 使用 YAML 手动安装

```powershell
conda env create -f environment.windows.yml
conda activate powerzoo-win
python -m pip install --no-deps -e .
python -m pip check
python scripts/check_runtime.py --train
```

YAML 安装按 pip 索引选择 PyTorch 构建；需要明确指定 CPU/CUDA 时优先使用安装脚本。
手动切换到 CUDA 版：

```powershell
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python scripts/check_runtime.py --train --require-cuda
```

`requirements.txt` 是运行依赖的唯一版本来源；`setup.py` 读取同一文件。
`requirements-dev.txt` 放测试和开发工具，`requirements-viz.txt` 放可选浏览器界面。
`requirements-windows.txt` 合并运行、开发依赖，并应用
`constraints/windows-py310.txt` 中的 Windows / Python 3.10 完整版本约束。
Conda 配置启用 `PYTHONUTF8=1`，避免 Windows 默认编码影响中文注释和日志。

### OpenDSS 到底安装什么

实际代码使用：

- `envs/vvc/vvc/circuit.py`、`envs/smartgrid/circuit_system/runtime.py`：`import dss`；
- `envs/dsr/core/circuit.py`：`import opendssdirect`。

因此环境同时安装 `dss-python==0.15.7`、其原生后端 `dss-python-backend==0.14.5`，
以及 `opendssdirect-py==0.9.4`。Windows wheel 自带 DSS 原生引擎，
本项目的这些运行路径不需要 OpenDSS 桌面程序、COM 注册或 DSS 子模块源码编译。
参见 [DSS-Python 官方说明](https://dss-extensions.org/DSS-Python/)。
`cupy` 只用于 VVC 中可选的敏感度加速路径，不是首次启动的依赖。

## Linux 与其他安装方式

`environment.yml` 已改为可移植的 Conda 配置，不再包含某台机器导出的 Linux build 编号
或整套 NVIDIA / Atari 依赖：

```bash
conda env create -f environment.yml
conda activate PowerZoo
python -m pip install --no-deps -e .
python scripts/check_runtime.py --train
```

已有 Python 3.10 环境时，也可以使用：

```bash
python -m pip install -e ".[dev]"
```

可选训练界面、环境渲染页面需要 `python -m pip install -e ".[viz]"`。
当前以源码仓库的 editable 安装为开发方式；DSS 模型、配置和数据仍位于仓库目录，
普通 wheel 不作为独立分发这些数据的安装方式。

## 首次运行与验证范围

```powershell
conda activate powerzoo-win
python scripts/check_runtime.py --train
```

成功时输出以 `POWERZOO_RUNTIME_OK` 开头的 JSON，包含实际解释器、Torch/CUDA、DSS
版本、观测形状和短训练结果。检查使用临时复制的 IEEE13 模型及仓库自带负荷曲线，
不会生成替代研究输入的合成数据，也不会覆盖仓库中的 DSS 文件。
这验证安装与执行链路，不代表算法已收敛或控制效果已得到验证。
本机版本、硬件、已执行检查及限制见 [Windows 验证记录](docs/windows-runtime.md)。

Windows 首次开发优先使用 `vvc + 13Bus + HAPPO`。普通训练可从下列短任务开始：

```powershell
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml --validate-config
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml
```

结果按训练器配置写入 `results/`。部分旧环境实现会在 `node_systems/` 下生成 worker DSS
和 episode 负荷文件；不要将这些运行产物作为模型修改提交。

### 需要单独处理的现有问题

- SmartGrid 已接入可复现的合成数据和统一文件入口，无需缺失的 month/irrad/temperature CSV 即可训练。
  单、多智能体共用一套独立 DSS context 内核；实际验证范围见 [SmartGrid 数据与运行设计](docs/smartgrid-data-runtime.md)。
- 已将30个退役入口、旧配置、专属工具及旧API测试归档到 `archive/legacy-training/`。集中控制基线使用
  `examples/single_agent/scripts/train_smartgrid.py`，与 MARL 共用环境配置和仿真。
- 其他电路、研究算法和浏览器UI组合仍需逐项验证，不以短训练代表收敛。

```powershell
python scripts/train.py --config configs/experiments/smartgrid_synthetic_smoke.yaml
python scripts/check_smartgrid.py --train --require-cuda
python examples/single_agent/scripts/train_smartgrid.py --steps 240
```

五类环境共用 `scripts/train.py` 和各自实验 YAML；目前正式训练验收为VVC、SmartGrid、District，
使用各自的 `check_vvc/check_smartgrid/check_district.py` 运行完整专项检查。
DSR与Stackelberg旧实现会在正式训练前拒绝。结构与历史清理清单见[入口整理说明](docs/training-entrypoints.md)。

## 测试

```powershell
python -m pytest tests/test_windows_runtime.py -o addopts="" -q
python -m pytest tests/test_configuration_contract.py tests/test_config_pipeline.py tests/test_unified_config.py -o addopts="" -q
```

完整测试为 `python -m pytest`，默认 `pytest.ini` 会生成覆盖率报告。
旧测试中存在将运行异常转为 skip 的情况，跳过不等于环境通过。
Windows 安装与运行另由 `.github/workflows/windows-runtime.yml` 验证，失败会使该任务失败。

## 持续开发：每项任务使用 worktree 分支

`main` 保持稳定，每项修改使用独立 worktree 和 `codex/<topic>` 分支：
以下新建命令以 Windows 修复已经审查并合入 `main` 为前提；当前开发在 `codex/training-entrypoints`，继承已验证的环境与SmartGrid修复；
未合并前，新任务应从此任务分支建立worktree。

```powershell
git -C D:\Exps\PowerZoo worktree list
git -C D:\Exps\PowerZoo worktree add -b codex/your-topic D:\Exps\PowerZoo-worktrees\your-topic codex/training-entrypoints
Set-Location D:\Exps\PowerZoo-worktrees\your-topic
conda activate powerzoo-win
python -m pip install --no-deps -e .
```

editable 安装会指向当前 worktree，共用环境时切换任务后需要重新安装。
修改依赖的并行任务应使用不同 Conda 环境。检查、提交、审查后再合并；
`fea/vvc` 保持独立，不合并进 `main`。详见 [开发流程](docs/development.md)
和 [协作约定](AGENTS.md)。

## 项目结构

```text
PowerZoo/
├── algorithms/       # Actor、Critic 和两时间尺度算法
├── common/           # Buffer、归一化和公共日志
├── configs/
│   ├── algos_cfgs/       # 多智能体算法参数
│   ├── envs_cfgs/        # 环境与运行配置
│   ├── systems/          # 电力系统与设备配置
│   └── training/         # 训练策略配置
├── envs/             # vvc、smartgrid、dsr、stackelberg、district_dispatch
├── examples/         # SmartGrid 集中控制入口及公共 CLI 兼容路径
├── models/           # 网络实现
├── node_systems/     # IEEE DSS 模型与部分负荷输入
├── runners/          # 自研训练运行器
├── scripts/          # 公共 train.py、Windows 安装与实际运行检查
├── archive/          # 已退役训练脚本/配置，附逐文件清单
├── tests/            # pytest 测试
└── training_ui/      # 可选训练界面
```

配置只通过 `utils/config_pipeline.py` 解析；旧 `unified_config_loader.py` 是兼容导入位置。
CLI、UI、系统引用、环境预设和实验覆盖采用同一规则，启动前可以检查最终值及来源：

```powershell
python scripts/train.py --env vvc --algo happo --set env.voltage_min=0.94 --explain env.voltage_min
python scripts/train.py --env smartgrid --env-config configs/envs_cfgs/smartgrid_pv_plans/smartgrid_aggressive.yaml --print-config
```

`train.episode_length` 是采样长度，`env.max_episode_steps` 是单局长度。新 CLI 使用
`--set 完整路径=值`，错误拼写和歧义会报错。每次训练保存完整 YAML 与参数来源记录。
详见 [配置指南](docs/configuration.md) 和 [代码审计与验证范围](docs/configuration-audit.md)。
QMix、2TS、旧 DAN-HAPPO、SN-MAPPO 的历史配置与公共 runner 契约不同，公共入口明确拒绝；
注册在代码中不代表已经适配或验证。SB3 旧入口也不属于本次已验证的公共 MARL 路径。

## 引用

Please cite the following paper if you use this repo for scientific research:

[1] X. Zheng, S. Yu, H. Cao, T. Shi, S. Xue, and T. Ding, “Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network,” IEEE Transactions on Smart Grid, pp. 1–1, Feb. 2025, doi: 10.1109/TSG.2025.3540416.
```
@article{zhengSensitivityBasedHeterogeneousOrdered2025,
  author  = {Xiaodong Zheng and Shixuan Yu and Hui Cao and Tianzhuo Shi and Shuangsi Xue and Tao Ding},
  title   = {Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2025},
  month   = {Feb},
  doi     = {10.1109/TSG.2025.3540416},
  issn    = {1949-3061},
  url     = {https://ieeexplore.ieee.org/document/10879343},
  urldate = {2025-02-12}
}
```

