# IEEE 34节点配电网仿真脚本

本脚本用于运行IEEE 34节点配电网的OpenDSS仿真，支持duty模式（时间序列仿真），并生成全面的仿真结果分析图表。

## 功能特性

### 仿真功能
- 运行IEEE 34节点配电网duty模式仿真
- 支持24小时时间序列分析（8640个时间步，10秒间隔）
- 包含光伏系统、负载和电池储能的综合建模
- 自动处理OpenDSS编译和求解过程

### 分析图表
1. **电压分析图** (`voltage_analysis.png`)
   - 主要母线电压时间曲线
   - 电压分布直方图
   - 电压热力图
   - 电压统计箱线图

2. **功率分析图** (`power_analysis.png`)
   - 负载功率时间曲线
   - 总负载与光伏出力对比
   - 功率因数分析
   - 光伏发电功率曲线

3. **光伏系统分析图** (`pv_analysis.png`)
   - 各光伏系统发电功率曲线
   - 光伏渗透率变化
   - 光伏发电功率分布
   - 光伏系统效率分析

4. **系统综合分析图** (`system_summary.png`)
   - 电压质量统计
   - 负载分布饼图
   - 系统损耗变化
   - 电压稳定性指标
   - 功率平衡分析
   - 关键指标汇总

### 数据输出
- `voltage_results.csv`: 所有母线电压时间序列数据
- `power_results.csv`: 所有负载功率时间序列数据
- `pv_results.csv`: 所有光伏系统功率时间序列数据
- 详细的仿真报告（控制台输出）

## 安装依赖

### 基础依赖（必需）
```bash
pip install -r requirements.txt
```

### OpenDSS Python接口（推荐）
```bash
# 方法1: 使用conda安装（推荐）
conda install -c conda-forge py_dss_interface

# 方法2: 使用pip安装
pip install py_dss_interface

# 方法3: 从源码安装
git clone https://github.com/PauloRadatz/py_dss_interface.git
cd py_dss_interface
pip install .
```

**注意**: 如果无法安装`py_dss_interface`，脚本会自动使用模拟数据运行，仍可生成完整的分析图表。

## 使用方法

### 基本运行
```bash

python run_ieee34_duty_simulation.py
```

### 文件结构
```
dss_simulations/
├── run_ieee34_duty_simulation.py  # 主仿真脚本
├── requirements.txt               # 依赖包列表
├── README.md                     # 说明文档
└── simulation_results/           # 结果输出目录（自动创建）
    ├── voltage_analysis.png
    ├── power_analysis.png
    ├── pv_analysis.png
    ├── system_summary.png
    ├── voltage_results.csv
    ├── power_results.csv
    └── pv_results.csv
```

## 脚本配置

### 主要参数
- **DSS文件路径**: `/home/zhengxiaodong/exps/PowerZoo/envs/smartgrid/node_systems_with_pv/34Bus/ieee34Mod1_duty.dss`
- **仿真时长**: 24小时
- **时间步长**: 10秒
- **总时间步**: 8640步
- **仿真模式**: duty（时间序列）

### 系统组成
- **母线数量**: 20个主要母线
- **负载数量**: 包含集中负载和分布式负载
- **光伏系统**: 3个光伏发电单元（PV834, PV890, PV864）
- **储能系统**: 2个电池储能单元
- **变压器**: 多个配电变压器

## 输出说明

### 图表说明
1. **电压质量分析**
   - 监控系统电压是否在合理范围内（0.95-1.05 pu）
   - 识别电压越限情况和薄弱节点
   - 分析电压稳定性和波动情况

2. **功率平衡分析**
   - 展示负载需求与光伏发电的匹配情况
   - 计算净负载和系统功率平衡
   - 分析功率因数和无功功率需求

3. **光伏系统性能**
   - 评估光伏发电效率和出力特性
   - 计算光伏渗透率对系统的影响
   - 分析光伏发电的时间分布特征

### 关键指标
- **电压质量**: 平均电压、最值、越限统计
- **能量统计**: 负载能量、光伏发电量、渗透率
- **系统损耗**: 平均损耗、损耗率
- **稳定性**: 电压标准差、功率波动

## 故障排除

### 常见问题
1. **OpenDSS接口问题**
   - 确保已正确安装OpenDSS软件
   - 检查py_dss_interface版本兼容性
   - 如无法安装，脚本会自动使用模拟数据

2. **文件路径问题**
   - 确认DSS文件路径正确
   - 检查相关的loadshape和其他依赖文件

3. **内存不足**
   - 可以减少仿真时间步数
   - 限制记录的母线和负载数量

### 性能优化
- 对于大规模仿真，可以调整时间步长
- 可以选择性记录关键母线和设备的数据
- 使用并行计算加速仿真过程

## 扩展功能

### 自定义分析
脚本采用模块化设计，可以轻松添加新的分析功能：
- 谐波分析
- 故障分析
- 优化算法集成
- 实时数据接口

### 参数调整
可以修改脚本中的参数来适应不同的仿真需求：
- 仿真时长和步长
- 记录的设备范围
- 图表样式和内容
- 输出格式

## 技术支持

如有问题或建议，请检查：
1. OpenDSS官方文档
2. py_dss_interface项目页面
3. 脚本内的详细注释
4. 仿真日志和错误信息

---

**版本**: 1.0  
**更新日期**: 2024年  
**兼容性**: Python 3.7+, OpenDSS 9.0+