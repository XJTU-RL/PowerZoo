# OpenDSS节点系统详尽使用说明

## 目录
1. [系统概述](#系统概述)
2. [目录结构说明](#目录结构说明)
3. [DSS脚本文件详解](#dss脚本文件详解)
4. [负荷曲线系统](#负荷曲线系统)
5. [快速入门指南](#快速入门指南)
6. [高级配置](#高级配置)
7. [故障排除](#故障排除)
8. [最佳实践](#最佳实践)

---

## 系统概述

本项目包含了多个标准IEEE配电系统测试馈线的OpenDSS模型，专门用于电力系统仿真和分析。这些模型支持日负荷模式仿真，并集成了分布式能源（如电池储能系统）。

### 支持的测试系统
- **IEEE 13节点测试馈线** - 小型不平衡配电系统
- **IEEE 34节点测试馈线** - 中型配电系统，包含电压调节器
- **IEEE 123节点测试馈线** - 大型不平衡配电系统
- **IEEE 8500节点测试馈线** - 超大型配电系统
- **IEEE 9500节点测试馈线** - 最大规模测试系统

---

## 目录结构说明

```
/node_systems/
├── 13Bus/                    # IEEE 13节点系统
│   ├── IEEE13Nodeckt_daily.dss    # 主脚本文件
│   ├── loadshape.dss              # 负荷曲线定义
│   ├── IEEELineCodes.dss          # 线路参数
│   ├── IEEE13Node_BusXY.csv       # 节点坐标
│   └── loadshape/                 # 负荷曲线数据目录
│       └── 000/                   # 场景000的数据
├── 34Bus/                    # IEEE 34节点系统
│   ├── ieee34Mod1_daily.dss       # 主脚本文件
│   ├── loadshape.dss              # 负荷曲线定义
│   ├── IEEELineCodes.dss          # 线路参数
│   ├── ieee34_BusXY.csv           # 节点坐标
│   └── loadshape/                 # 负荷曲线数据目录
│       ├── data_without_noise/    # 无噪声数据
│       └── data_with_gaussian_noise/ # 含高斯噪声数据
├── 123Bus/                   # IEEE 123节点系统
│   ├── IEEE123Master_daily.dss    # 主脚本文件
│   ├── IEEE123Loads_daily.dss     # 负荷定义
│   ├── loadshape.dss              # 负荷曲线定义
│   └── loadshape/                 # 负荷曲线数据目录
├── 8500-Node/                # IEEE 8500节点系统
│   ├── Master_daily.dss           # 主脚本文件
│   ├── loadshape.dss              # 负荷曲线定义
│   ├── Loads_daily.dss            # 负荷定义
│   └── 其他配置文件...
└── 9500-Node/                # IEEE 9500节点系统
    └── 相关配置文件...
```

---

## DSS脚本文件详解

### 1. 主脚本文件结构

每个系统的主脚本文件（如`ieee34Mod1_daily.dss`）都遵循以下标准结构：

#### 1.1 电路初始化
```dss
Clear                          ! 清除之前的电路定义
New Circuit.IEEE34             ! 创建新电路
~ basekv=69 pu=1.0001         ! 设置基准电压和标幺值
~ phases=3 bus1=sourcebus      ! 设置相数和源母线
~ Angle=30                     ! 设置相角
~ MVAsc3=200000 MVASC1=210000  ! 设置短路容量
```

#### 1.2 设备定义
```dss
! 变压器定义
New Transformer.SubXF Phases=3 Windings=2 XHL=8
~ wdg=1 bus=sourcebus conn=delta kv=69 kva=25000 %r=0.5
~ wdg=2 bus=800 conn=wye kv=24.9 kva=25000 %r=0.5

! 线路定义
New Line.L1 Phases=3 Bus1=800 Bus2=802 LineCode=300 Length=2.58

! 负荷定义（关键：daily参数）
New Load.S860 Bus1=860 Phases=3 Conn=Wye Model=1 kV=24.9 
~ kW=20 kvar=16 daily=loadshape_S860
```

#### 1.3 分布式能源（电池）
```dss
! 电池定义 - 初始化为断开状态
New Generator.batt1 bus1=860 kV=24.9 kW=0 pf=0.95 conn=Wye Model=1
New Generator.batt2 bus1=840 kV=24.9 kW=0 pf=0.95 conn=Wye Model=1
```

#### 1.4 仿真设置
```dss
! 设置电压基准
Set voltagebases=[69, 24.9, 4.16, 0.48]
Calcvoltagebases

! 导入母线坐标
BusCoords ieee34_BusXY.csv

! 设置日负荷仿真模式
Set mode=Daily number=1 hour=0 stepsize=3600 sec=0
```

### 2. 关键参数说明

#### 2.1 `daily` 参数
- **作用**: 指定负荷的日负荷曲线
- **格式**: `daily=loadshape_name`
- **示例**: `daily=loadshape_S860` 表示使用名为`loadshape_S860`的负荷曲线

#### 2.2 仿真模式参数
- **mode=Daily**: 设置为日负荷仿真模式
- **number=1**: 仿真天数（1天）
- **hour=0**: 起始小时（0点开始）
- **stepsize=3600**: 时间步长（3600秒=1小时）
- **sec=0**: 起始秒数

---

## 负荷曲线系统

### 1. 负荷曲线定义文件（loadshape.dss）

#### 1.1 基本语法
```dss
New Loadshape.loadshape_S860 npts=24 sinterval=3600 mult=(file=./loadshape/000/S860.csv)
```

**参数说明**:
- `loadshape_S860`: 负荷曲线名称
- `npts=24`: 数据点数（24小时）
- `sinterval=3600`: 时间间隔（秒）
- `mult=(file=...)`: 从CSV文件读取倍数数据

#### 1.2 不同系统的时间配置

**34Bus和13Bus系统**:
```dss
npts=24 sinterval=3600    ! 24小时，每小时一个数据点
```

**123Bus系统**:
```dss
npts=5 sinterval=17280    ! 5个数据点，每个点4.8小时
```

### 2. 负荷曲线数据文件

#### 2.1 数据格式
CSV文件包含标准化的负荷倍数（0-1之间的小数）：

**34Bus系统示例** (`S860.csv`):
```
0.352673492605233    # 0:00时刻的负荷倍数
0.346985210466439    # 1:00时刻的负荷倍数
0.329920364050057    # 2:00时刻的负荷倍数
...
0.346985210466439    # 23:00时刻的负荷倍数
```

**123Bus系统示例** (`S1a.csv`):
```
0.43     # 第1个时间段
0.426    # 第2个时间段
0.423    # 第3个时间段
0.425    # 第4个时间段
0.427    # 第5个时间段
```

#### 2.2 数据目录结构
```
loadshape/
├── data_without_noise/     # 无噪声数据
│   ├── 000/               # 场景000
│   ├── 001/               # 场景001
│   └── ...
├── data_with_gaussian_noise/  # 含高斯噪声数据
│   ├── 000/
│   └── ...
└── scale.txt              # 缩放因子文件
```

---

## 快速入门指南

### 1. 环境准备

#### 1.1 安装OpenDSS
```bash
# Windows
# 下载并安装OpenDSS官方版本

# Linux (使用Python接口)
pip install opendssdirect[extras]
```

#### 1.2 验证安装
```python
import opendssdirect as dss
print(dss.Basic.Version())
```

### 2. 运行第一个仿真

#### 2.1 使用OpenDSS命令行
```bash
# 进入13Bus目录
cd /path/to/node_systems/13Bus

# 运行OpenDSS
opendssdirect

# 在OpenDSS中执行
redirect IEEE13Nodeckt_daily.dss
solve
show voltages
```

#### 2.2 使用Python脚本
```python
import opendssdirect as dss
import os

# 设置工作目录
os.chdir('/path/to/node_systems/13Bus')

# 加载并运行仿真
dss.run_command('redirect IEEE13Nodeckt_daily.dss')
dss.run_command('solve')

# 获取结果
voltages = dss.Circuit.AllBusVmagPu()
print(f"节点电压: {voltages}")
```

### 3. 基本操作示例

#### 3.1 修改负荷
```python
# 修改特定负荷的功率
dss.Loads.Name('671')  # 选择负荷671
dss.Loads.kW(1500)     # 设置有功功率为1500kW
dss.Loads.kvar(800)    # 设置无功功率为800kvar

# 重新求解
dss.run_command('solve')
```

#### 3.2 控制电池
```python
# 激活电池
dss.Generators.Name('batt1')  # 选择电池1
dss.Generators.kW(100)        # 设置输出功率100kW
dss.Generators.PF(0.95)       # 设置功率因数

# 重新求解
dss.run_command('solve')
```

#### 3.3 获取仿真结果
```python
# 获取所有母线电压
voltages = dss.Circuit.AllBusVmagPu()
bus_names = dss.Circuit.AllBusNames()

# 获取线路功率流
dss.run_command('show powers kva elements')

# 获取损耗
losses = dss.Circuit.Losses()
print(f"系统损耗: {losses[0]/1000:.2f} kW")
```

---

## 高级配置

### 1. 自定义负荷曲线

#### 1.1 创建新的负荷曲线数据
```python
import numpy as np
import pandas as pd

# 生成24小时负荷曲线
hours = np.arange(24)
load_curve = 0.3 + 0.4 * np.sin(2 * np.pi * (hours - 6) / 24) + 0.3 * np.random.normal(0, 0.05, 24)
load_curve = np.clip(load_curve, 0.1, 1.0)  # 限制在0.1-1.0之间

# 保存为CSV
pd.DataFrame(load_curve).to_csv('custom_load.csv', header=False, index=False)
```

#### 1.2 在DSS中定义新负荷曲线
```dss
! 在loadshape.dss中添加
New Loadshape.custom_loadshape npts=24 sinterval=3600 mult=(file=./loadshape/000/custom_load.csv)

! 在主脚本中使用
New Load.custom_load Bus1=860 Phases=3 Conn=Wye Model=1 kV=24.9 
~ kW=50 kvar=30 daily=custom_loadshape
```

### 2. 批量仿真

#### 2.1 多场景仿真脚本
```python
import opendssdirect as dss
import os
import pandas as pd

def run_scenario_analysis(base_path, scenarios):
    results = []
    
    for scenario in scenarios:
        # 设置工作目录
        os.chdir(f"{base_path}/34Bus")
        
        # 修改loadshape.dss中的路径
        modify_loadshape_path(scenario)
        
        # 运行仿真
        dss.run_command('redirect ieee34Mod1_daily.dss')
        dss.run_command('solve')
        
        # 收集结果
        voltages = dss.Circuit.AllBusVmagPu()
        losses = dss.Circuit.Losses()
        
        results.append({
            'scenario': scenario,
            'min_voltage': min(voltages),
            'max_voltage': max(voltages),
            'total_losses': losses[0]/1000
        })
    
    return pd.DataFrame(results)

# 运行分析
scenarios = ['000', '001', '002', '003', '004']
results = run_scenario_analysis('/path/to/node_systems', scenarios)
print(results)
```

### 3. 电池优化控制

#### 3.1 基于电压的电池控制
```python
def voltage_based_battery_control():
    # 获取当前电压
    voltages = dss.Circuit.AllBusVmagPu()
    bus_names = dss.Circuit.AllBusNames()
    
    # 找到电压最低的母线
    min_voltage_idx = voltages.index(min(voltages))
    min_voltage_bus = bus_names[min_voltage_idx]
    
    # 如果电压过低，激活附近的电池
    if min(voltages) < 0.95:
        # 激活电池支撑电压
        dss.Generators.Name('batt1')
        dss.Generators.kW(200)  # 输出200kW
        print(f"电压过低({min(voltages):.3f})，激活电池支撑")
    
    # 如果电压过高，电池吸收功率
    elif max(voltages) > 1.05:
        dss.Generators.Name('batt1')
        dss.Generators.kW(-100)  # 吸收100kW
        print(f"电压过高({max(voltages):.3f})，电池吸收功率")
    
    # 重新求解
    dss.run_command('solve')
```

---

## 故障排除

### 1. 常见错误及解决方案

#### 1.1 文件路径错误
**错误信息**: `File not found: ./loadshape/000/S860.csv`

**解决方案**:
1. 检查当前工作目录是否正确
2. 确认CSV文件是否存在
3. 检查路径分隔符（Windows使用\，Linux使用/）

```python
# 检查文件是否存在
import os
if os.path.exists('./loadshape/000/S860.csv'):
    print("文件存在")
else:
    print("文件不存在，请检查路径")
```

#### 1.2 收敛问题
**错误信息**: `Solution did not converge`

**解决方案**:
1. 检查负荷数据是否合理
2. 调整求解器参数
3. 检查网络连接性

```dss
! 调整求解器参数
Set maxiterations=100
Set tolerance=0.0001
Set algorithm=newton
```

#### 1.3 电压越限
**问题**: 节点电压超出正常范围

**解决方案**:
1. 检查负荷曲线数据
2. 调整变压器分接头
3. 添加电压调节设备

### 2. 调试技巧

#### 2.1 逐步调试
```python
# 逐步加载组件
dss.run_command('clear')
dss.run_command('new circuit.test')
# 只加载线路
dss.run_command('redirect IEEELineCodes.dss')
# 检查是否正常
dss.run_command('solve')
```

#### 2.2 输出详细信息
```dss
! 显示详细求解信息
set verbose=y
solve
show solution
```

---

## 最佳实践

### 1. 项目组织

#### 1.1 目录结构建议
```
project/
├── base_models/          # 基础模型文件
├── scenarios/            # 不同场景配置
├── results/              # 仿真结果
├── scripts/              # 自动化脚本
└── docs/                 # 文档
```

#### 1.2 版本控制
- 使用Git管理DSS文件
- 为重要修改创建标签
- 记录参数变更历史

### 2. 性能优化

#### 2.1 大规模仿真优化
```python
# 禁用不必要的输出
dss.run_command('set verbose=n')
dss.run_command('set showexport=n')

# 使用更快的求解算法
dss.run_command('set algorithm=newton')
dss.run_command('set maxiterations=50')
```

#### 2.2 内存管理
```python
# 定期清理内存
dss.run_command('clear')

# 批量处理时重启DSS引擎
if scenario_count % 100 == 0:
    dss.Basic.ClearAll()
```

### 3. 结果验证

#### 3.1 基本检查
```python
def validate_results():
    # 检查电压范围
    voltages = dss.Circuit.AllBusVmagPu()
    if min(voltages) < 0.9 or max(voltages) > 1.1:
        print("警告：电压超出正常范围")
    
    # 检查功率平衡
    total_load = sum([dss.Loads.kW() for _ in dss.Loads.AllNames()])
    total_gen = sum([dss.Generators.kW() for _ in dss.Generators.AllNames()])
    losses = dss.Circuit.Losses()[0]/1000
    
    balance_error = abs(total_gen - total_load - losses)
    if balance_error > 1:  # 1kW误差
        print(f"警告：功率不平衡，误差{balance_error:.2f}kW")
```

#### 3.2 结果对比
```python
# 与基准案例对比
def compare_with_baseline(current_results, baseline_results):
    voltage_diff = np.array(current_results['voltages']) - np.array(baseline_results['voltages'])
    max_voltage_change = max(abs(voltage_diff))
    
    if max_voltage_change > 0.05:  # 5%变化
        print(f"警告：电压变化过大，最大变化{max_voltage_change:.3f}")
```

### 4. 文档和注释

#### 4.1 DSS文件注释规范
```dss
! ========================================
! IEEE 34节点测试馈线 - 日负荷仿真模型
! 创建日期: 2024-01-01
! 修改记录: 
!   - 2024-01-15: 添加电池储能系统
!   - 2024-01-20: 更新负荷曲线数据
! ========================================

! 电路基本参数设置
Clear
New Circuit.IEEE34 
~ basekv=69          ! 基准电压69kV
~ pu=1.0001          ! 标幺电压1.0001
~ phases=3           ! 三相系统
~ bus1=sourcebus     ! 源母线名称
```

#### 4.2 Python脚本文档
```python
def run_daily_simulation(system_path: str, scenario: str = '000') -> dict:
    """
    运行日负荷仿真
    
    Args:
        system_path: 系统文件路径
        scenario: 场景编号，默认'000'
    
    Returns:
        dict: 包含电压、损耗等仿真结果
    
    Example:
        >>> results = run_daily_simulation('/path/to/34Bus', '001')
        >>> print(f"最小电压: {results['min_voltage']:.3f}")
    """
    # 实现代码...
```

---

## 总结

本使用说明涵盖了OpenDSS节点系统的完整使用流程，从基础概念到高级应用。通过遵循本指南，您可以：

1. **快速上手** - 理解系统结构，运行第一个仿真
2. **深入应用** - 自定义负荷曲线，控制分布式能源
3. **专业分析** - 批量仿真，结果验证，性能优化
4. **可靠运行** - 故障排除，最佳实践，文档规范

建议按照以下学习路径逐步掌握：
1. 从13Bus小系统开始熟悉基本操作
2. 理解负荷曲线系统的工作原理
3. 学习电池控制和优化方法
4. 掌握批量仿真和结果分析技巧
5. 应用到实际项目中并持续优化

如有问题，请参考故障排除章节或查阅OpenDSS官方文档。