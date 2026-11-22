# PowerZoo

<img src="docs/assets/powerzoo-logo.png" alt="PowerZoo 标识" width="112">

PowerZoo 是面向电力系统强化学习的 Python 研究框架。它用 OpenDSS 仿真电网，通过统一的 YAML 配置启动多智能体训练或集中控制基线。

[查看交互演示](https://xjtu-rl.github.io/PowerZoo/) · [配置指南](docs/configuration.md) · [VVC 算例库](docs/vvc-case-library.md)

## 快速开始

先安装 Miniconda 或 Miniforge，再克隆仓库。以下命令均在仓库根目录执行。

```sh
git clone https://github.com/XJTU-RL/PowerZoo.git
cd PowerZoo
```

**Windows（PowerShell）**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
conda activate powerzoo-win
```

默认安装 CPU 版本。使用兼容的 NVIDIA 驱动时，可向安装脚本传入 `-TorchFlavor cu124`。[Windows 环境与验证说明](docs/windows-runtime.md)

**Linux（Conda，CPU）**

```sh
conda env create -f environment.yml
conda activate PowerZoo
python -m pip install --no-deps -e .
python scripts/check_runtime.py --train
```

运行基线为 Python 3.10。看到 `POWERZOO_RUNTIME_OK` 表示原生 DSS 求解、VVC 工作进程和一次 HAPPO 更新通过。OpenDSS 通过 Python 原生绑定运行，不需要安装 OpenDSS 桌面程序。

### 运行 IEEE 13 节点 VVC

先解析实验配置，再运行 HAPPO 短训练。该预设使用合成时序，用于核对训练链路。

```sh
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml --validate-config
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml
```

同一场景的集中控制基线使用 Stable-Baselines3 PPO：

```sh
python scripts/train_centralized.py --config configs/experiments/vvc_13bus_smoke.yaml --backend sb3_ppo --steps 96
```

输出默认写到 `results/`。短训练检查入口、仿真和参数更新，不代表策略收敛或满足电网约束。

## 场景与训练入口

下表列出已有短训练证据的组合。运行其它组合前，请查看各场景的验证范围。

| 任务 | 示例配置 | 已验证的入口 |
| --- | --- | --- |
| [电压无功控制（VVC）](docs/vvc-physics.md) | [IEEE 13](configs/experiments/vvc_13bus_smoke.yaml)、[IEEE 34](configs/experiments/vvc_34bus_smoke.yaml)、[IEEE 123](configs/experiments/vvc_123bus_smoke.yaml) | HAPPO、集中式 PPO |
| [离散 VVC 多智能体基线](docs/discrete-marl-vvc.md) | [IEEE 34 局部设备观测](configs/experiments/vvc_34bus_discrete_marl_quickstart.yaml) | IPPO、QMIX、VDN、COMA |
| [含光伏配电网控制（SmartGrid）](docs/smartgrid-data-runtime.md) | [合成时序短训](configs/experiments/smartgrid_synthetic_smoke.yaml) | HAPPO、集中式 PPO |
| [有源配电网故障恢复（DSR）](docs/dsr-physics.md) | [小网络](configs/experiments/dsr_small_smoke.yaml)、[IEEE 13 派生场景](configs/experiments/dsr_13bus_smoke.yaml) | HAPPO、集中式 PPO |
| [区域协同调度（District Dispatch）](docs/district-physics.md) | [三区域实验](configs/experiments/district_dispatch_smoke.yaml) | HAPPO、集中式 PPO |
| [顺序定价与需求响应（Stackelberg）](docs/pricing-physics.md) | [小网络](configs/experiments/stackelberg_smoke.yaml)、[IEEE 13 派生场景](configs/experiments/stackelberg_13bus_smoke.yaml) | `sequential_ppo`，由 `scripts/train.py` 启动 |

可用 `--print-config` 查看解析后的配置，或用 `--set 完整路径=值` 覆盖参数。`--validate-config` 会检查已声明的 VVC 多智能体组合的静态兼容性，无需启动仿真；正式训练还会复核运行环境的设备布局。用 `python scripts/check_capabilities.py --list` 查看能力声明，详见 [VVC 接入与验收](docs/vvc-acceptance-contract.md)。

### 参考馈线与论文复现

- [VVC 算例库](docs/vvc-case-library.md)提供 IEEE 34 和 IEEE 123 的参考底板与独立储能叠加层，可生成直接交给训练入口的实验 YAML。来源和文件摘要见[算例清单](scenarios/vvc/cases.yaml)。
- [OHASAC 负荷频率控制](docs/lfc-ieee9-reproduction.md)：三机九节点输电网与 IEEE 13 配电网的联合仿真已核对，需要 MATLAB、Simulink 和相应 Python Engine。论文完整训练的验证范围以复现说明为准。
- [PISTG-LDCPPO](research/pistg_ldcpo/)：IEEE 33 节点配电网中多储能协调的独立复现。[CBESS 快速配置](configs/experiments/cbess_ldcppo_quickstart.yaml)使用集中入口的 `ldcppo` 后端；复现实验与结果见[实验报告](docs/pistg-repro-results.md)及[复现偏离说明](docs/pistg-repro-deviations.md)。

[训练入口说明](docs/training-entrypoints.md) · [演示数据与复现](docs/showcase/data/README.md) · [论文索引](papers-repro/) · [贡献指南](docs/development.md)

## 引用

如果在研究中使用 PowerZoo，请引用 [Zheng 等，IEEE Transactions on Smart Grid](https://doi.org/10.1109/TSG.2025.3540416)：

```bibtex
@article{zhengSensitivityBasedHeterogeneousOrdered2025,
  author  = {Xiaodong Zheng and Shixuan Yu and Hui Cao and Tianzhuo Shi and Shuangsi Xue and Tao Ding},
  title   = {Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2025},
  doi     = {10.1109/TSG.2025.3540416}
}
```
