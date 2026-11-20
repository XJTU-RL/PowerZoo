# IEEE 123 → IEEE9 Bus 8 联合探针（待实际 MATLAB 验证）

本实验仅为新的研究验收，不改论文模型、不注册训练环境。Bus 8 的
动态负荷通过 `load1p` / `load1q` 接受等值功率，是与同母线固定
`Load C 100 MW 35 MVAr` 并联的另一元件。不要再次向动态支路加入
100 MW / 35 MVAr。脚本不改变固定负荷、其他区域负荷、断路器或
Simulink 拓扑。

`scenarios/lfc/ieee123-native/coupling_v1.json` 固定模型摘要与待测始点：
论文 13 节点流程 seed 17、rank 0 在 1.033333 s 的复位接口命令为
7,851.739821876 kW / 12,942.530488712 kvar，当时 GSP 频率约
59.965001 Hz。这是此前观测的初始条件，不是稳态证明。123 配电侧
的原生静态始点约 3,615.265 kW / 1,311.528 kvar，算术背景是二者
相减；不对配电设备或潮流结果放大。实际命令采用这个始点叠加实时
测得的 123 馈线 P/Q 增量，kW/kvar 到 W/var 转换一次。

## 按以下顺序执行

需要已有 MATLAB R2025b、匹配 Python Engine、有有效许可证的
Simulink/Simscape Electrical，以及已有 DSS-Python。无桌面时使用
Xvfb；不得把缺少许可证、引擎或测量通道当作通过。模型与所需辅助脚本
复制到单独的临时目录，在带超时的验证进程运行，检查原始模型哈希。

```sh
python -m unittest tests.test_lfc_123_native tests.test_lfc_123_coupling_probe
xvfb-run -a timeout 240 python -m scripts.check_lfc_123_coupling
# 仅在 warmup_passed 后另外执行短联动：
xvfb-run -a timeout 480 python -m scripts.check_lfc_123_coupling --coupled
```

1. 默认只跑 20 s 的恒定动态负荷，取最后 3 s 的实际功率传感器
   Pm、Qm 与 GSP 频率。命令回读不能替代功率传感器。若 Simscape
   日志缺少 Pm、Qm 或时序覆盖，停止并报告缺项；不填零、不放宽门槛。
2. 显式请求 `--coupled` 时重跑预热检查，然后重复同一初始条件。
   每次 Simulink 停在新的通信边界时由原生 IEEE 123 求解一帧，随后
   将实测净输入增量送给 Bus 8 动态负荷。固定基步长是 1/60 s，
   通信间隔 1 s，实际暂停允许模型断言最多迟两个基步。
   结束后还要复核本次预热的真实 P/Q 及频率，以及每帧的传感器值。
3. 短探针使用固定储能脉冲，不根据频率闭环，不验证控制收益。
   配电侧仍是刚性电源，输出报告不得描述为双向耦合、训练通过或论文复现。
   时序负荷和 OEDI 数据不是本实验的一部分。

当前只运行了不启动 MATLAB 的单元测试。必须实际验证 R2025b 能从
既有 Simscape 日志把 Bus 8 动态支路的实际 P/Q 以指定的通道和单位
导出，才能升级验收状态。若缺通道，优先报告阻塞，不改原始模型。
