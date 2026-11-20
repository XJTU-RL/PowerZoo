# OHASAC 三机九节点 × 13Bus 联合仿真

论文：[Order-based Heterogeneous Agents Reinforcement Learning Method with the Coordination of Distribution Network and Transmission Network for Load Frequency Control](https://doi.org/10.1109/TPWRS.2025.3598951)。

## 模型与数据

- 输电侧：Simulink IEEE 三机九节点系统。三台同步机分别接在 Bus 1、2、3；模型根层包含 Bus 1–9。Bus 2、3 是潮流的 PV 节点，不是光伏单元。
- 配电侧：本次验证使用 OpenDSS 13Bus `IEEE13Nodeckt_duty.dss`，含 7 台光伏和 3 台储能。训练/评估使用的 5 个 Simulink 模型文件均与 [论文实现](https://github.com/rokey0001/MARL_LFC/tree/63f466186b0649896bda1013d296df2dcf682017/harl/envs/LFC/IEEE9BusSystem) 对应文件逐字节一致。
- 该 LFC 电路不是标准 VVC 13Bus：230 kV 电源经 230/75 kV 变压器接入；光伏为 `PV671/PV634/PV6452/PV646/PV692/PV6113/PV6521`。
- 储能接在 680、675、670，由论文内核用 `Generator.batt*` 和 `Battery.csv` 更新功率和能量，并非 OpenDSS 原生 `Storage`。额定功率 12.5/10/4.455 MW，额定能量 100/100/50 MWh，初始 SOC 均为 50%。
- 负荷使用随包的 `systems/13Bus/loadshape/LoadShape1.CSV` 至 `LoadShape3.CSV` 归一化倍率；30 步设置下生成 233 个窗口。未核实为实测，不是 VVC 场景清单。exp1 使用固定辐照（`use_irrad: false` 和 `Irrad_constant`），不是禁用光伏。
- 包内另有 34Bus 配电模型，不能由本次 13Bus 的运行结果推定 34Bus 已完成联合仿真验收。

## 运行检查

在仓库根目录运行。需要已安装的 MATLAB 和对应 Python Engine、有许可证的 Simulink/Simscape Electrical（含所用模块）、DSS-Python 与项目 Python 依赖。本次在 Linux、MATLAB R2025b 上验证；无桌面会话时可使用 Xvfb。

```bash
xvfb-run -a python scripts/check_lfc.py --matlab
```

预期出现 `[PASS] matlab` 和 `LFC_CHECK_RESULT=OK`。不传 `--matlab` 时不得将结构和 OpenDSS 检查视为联合仿真通过。

本次使用 Linux 上的 MATLAB R2025b、DSS-Python 0.15.7 和 `de8ce3c7` 的未修改源码：OpenDSS 负荷曲线检查与电路求解通过，真实 MATLAB 联合仿真一步返回有限团队奖励 −1.0687。另以相同模型三步探测核对了三个发电机的有功出力和转速，以及配电侧 OpenDSS 电源有功/无功与 Simulink 配电等效负荷输入的一致性。

配对检查固定光伏动作 0.5，将储能动作设为 0 和 +0.5，各运行三步。第一步 Simulink 收到的配电接口有功为 −2.492 和 −15.967 MW（负值表示配电侧向输电侧送电）；第二步的接口频率分别为 60.03119 和 60.08778 Hz，三台同步机的有功与转速也出现差异。这是动作响应检查，不是控制效果对比。

**首步时钟限制：** 上游代码及当前实现均在 reset 和第一次 step 设置同一个 1 s 暂停目标。实际暂停时刻依次为 1.0333 s（reset）、1.0500 s、2.0333 s、3.0333 s。第一步已写入新的配电有功与无功，但采样到的输电状态仍与 reset 相同，第二步才观测到差异。不能把 reset 后的第一次动作视作完整的一秒响应。此处保留上游时序，不在复现说明中把它当作已修复行为。

## 复做三步动作探测

以下命令各自启动 MATLAB 引擎，不加载权重、不训练。6 步上限仅用于探测前三步，不替代论文的 30 步回合。

```bash
for battery in 0 0.5; do
  xvfb-run -a python - "$battery" <<'PYPROBE' || exit
import json
import sys
import numpy as np
from envs.lfc.lfc_config import LFCConfig
from envs.lfc.lfc_env import LFCEnv

env = LFCEnv(LFCConfig(system="13Bus", dss_file="IEEE13Nodeckt_duty.dss",
                       max_episode_steps=6), rank=0)
try:
    env.seed(17)
    np.random.seed(17)
    env.reset()
    core = env.env
    for i in range(4):
        if i:
            action = np.array([[0.5]] * env.pv_num
                              + [[float(sys.argv[1])]] * env.bat_num)
            obs, _, reward, done, _, _ = env.step(action)
            assert np.isfinite(obs).all() and np.isfinite(reward).all()
            assert not np.any(done)
        gen = np.asarray(core.eng.workspace["gen_ob"], dtype=float)[-1].ravel()
        freq = np.asarray(core.eng.workspace["out_f_load1"], dtype=float).ravel()[-1]
        t = float(core.eng.get_param(core.env_name, "SimulationTime"))
        pq = [float(core.eng.get_param(core.env_name + "/load1" + k, "value"))
              for k in ("p", "q")]
        source = core.circuit.dss.ActiveCircuit.CktElements("Vsource.SOURCE")
        assert len(gen) == 6 and np.isfinite(np.r_[gen, freq, t, pq]).all()
        np.testing.assert_allclose(pq, -1000 * np.asarray(source.TotalPowers)[:2])
        print(json.dumps(dict(step=i, time_s=t, frequency_hz=float(freq),
                              interface_W_var=pq, generator_W=gen[:3].tolist(),
                              generator_speed_pu=gen[3:].tolist())))
finally:
    env.close()
PYPROBE
done
```

对照两条轨迹的初态、接口 P/Q 和第二步响应。不要求不同 MATLAB/OpenDSS 版本逐位相同。该接口将配电侧功率写入输电模型；OpenDSS 仍按既定电源求解，不代表输配电压与频率双向闭环，也不用于证明控制收益。

## 训练入口（未用于本次验收）

```bash
xvfb-run -a python scripts/train.py --config configs/experiments/lfc_13bus_ohasac_quickstart.yaml
xvfb-run -a python scripts/train.py --config configs/experiments/lfc_13bus_ohasac_full.yaml
```

快速档 180 步、1 训练/1 评估 worker；完整配置默认 15,000 步、每回合 30 步、3 训练/1 评估 worker，每个 worker 独立启动 MATLAB。完整配置默认 CUDA。这些是启动配置，不是训练通过记录。

## 验收范围

上述结果核对的是三机九节点模型、13Bus 配电侧和联合仿真接口。尚未在本次验收核对 15,000 步训练的完成产物、论文曲线或算法收敛，不能等同于论文结果复现。当前 13Bus 训练配置见 [实验配置](../configs/experiments/lfc_13bus_ohasac_full.yaml)。
