# IEEE34 原生连续储能与 MADDPG／MATD3

这是连续控制接口与离策略采样验证，不引入 SHOM 排序，也不依赖离散
MARL 基线。默认 VVC 仍允许策略控制原有机械设备和储能。

## 接口边界

- `env.continuous_storage_only: true` 要求 `env.bat_act_num: .inf`。每台原生
  Storage 对应一个 `Box(-1, 1, (1,))` actor。归一化指令按设备额定功率映射，
  实际限幅、SOC 和时间推进由 OpenDSS 会话处理。
- IEEE34 的电容器和调压器保持编译后的设置，仍参与仿真和观测，不再占用
  动作槽位。PV 保持模型设定。新模式必须有原生储能；普通 VVC 的动作路径不变。
- MADDPG 和 MATD3 通过公共训练入口采集原生连续动作并更新 replay；MATD3
  保留双 critic 和延迟 actor 更新。仅在新储能模式改变随机预热取样方式；
  其它离策略模式继续使用原有可用动作规则。HADDPG／HATD3 未接入本模式，
  在静态阶段拒绝，不据此取消其其它连续配置的声明。
- 储能模式的策略重载使用物理清单核对模型、逐设备动作及观测语义。已验证
  同配置、同工况的确定性轨迹重放；没有承诺优化器、replay buffer 的精确断点续训。

## 复现范围

```sh
python scripts/train.py --config configs/experiments/vvc_34bus_continuous_storage_quickstart.yaml
python scripts/train.py --algo matd3 --config configs/experiments/vvc_34bus_continuous_storage_quickstart.yaml
python -m pytest -q -o addopts= tests/test_vvc_continuous_storage.py
```

IEEE34 两台原生储能使用合成场景及 OpenDSS 求解。检查不同非档位命令的功率
实测、SOC 方向、参数更新、MATD3 延迟更新和保存前后的轨迹一致性。该预设
按原 runner 口径另做 4 个预热交互，随后采集配置的 8 个训练交互。短跑不用于
收益、算法排名或跨馈线泛化结论。
