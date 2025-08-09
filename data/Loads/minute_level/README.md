# 分钟级负荷数据说明

本目录包含从小时级负荷数据插值生成的分钟级负荷数据文件。

## 文件说明

- `LoadShape1_minute_level.csv` - 第一个负荷形状的分钟级数据
- `LoadShape2_minute_level.csv` - 第二个负荷形状的分钟级数据  
- `LoadShape3_minute_level.csv` - 第三个负荷形状的分钟级数据
- `interpolation_report.txt` - 插值过程的详细报告

## 数据规格

- **原始数据**: 8,760个点 (小时级, 1年365天×24小时)
- **插值数据**: 525,600个点 (分钟级, 1年365天×24小时×60分钟)
- **插值方法**: 三次样条插值 (CubicSpline)
- **数据格式**: CSV文件，每行一个负荷系数值
- **数据范围**: 0.0 - 1.0 (归一化的负荷系数)

## 插值质量

根据插值报告，所有文件的插值质量为**优秀**:
- 平均绝对误差 (MAE): 0.000000
- 相关系数: 1.000000
- 插值数据完美保持了原始负荷的变化趋势

## 使用方法

```python
import pandas as pd
import numpy as np

# 读取分钟级负荷数据
load_data = pd.read_csv('LoadShape1_minute_level.csv', header=None)
load_values = load_data.iloc[:, 0].values

# 数据包含一年的分钟级负荷系数
print(f"数据点数: {len(load_values)}")  # 525600
print(f"数据范围: {load_values.min():.6f} - {load_values.max():.6f}")

# 获取特定时间的负荷值
# 例如：第1天第2小时第30分钟的负荷值
day = 1      # 第1天 (从1开始)
hour = 2     # 第2小时 (从1开始) 
minute = 30  # 第30分钟 (从1开始)

index = (day-1) * 24 * 60 + (hour-1) * 60 + (minute-1)
load_value = load_values[index]
print(f"第{day}天第{hour}小时第{minute}分钟的负荷系数: {load_value:.6f}")
```

## 注意事项

1. 插值后的数据保持了原始负荷的变化趋势和周期性特征
2. 使用了边界增强处理，确保年初年末的平滑过渡
3. 所有负值已被修正为0（负荷系数应为非负）
4. 建议结合实际使用场景验证插值结果的合理性
5. 与光伏辐照度数据匹配使用时，确保时间索引的对应关系正确

## 重新生成数据

如需重新生成或修改插值参数，可使用以下命令：

```bash
# 处理所有文件
python data/Loads/interpolate_loads.py

# 处理单个文件
python data/Loads/interpolate_loads.py --single LoadShape1.CSV

# 应用平滑处理
python data/Loads/interpolate_loads.py --smooth 0.1

# 生成对比图表
python data/Loads/interpolate_loads.py --plot
```

---
生成时间: 2025-07-30
插值工具: PowerZoo LoadInterpolator