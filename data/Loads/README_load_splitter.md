# 负荷数据分割工具

这个工具可以将一年的分钟级负荷数据按月份和时间段进行分割，方便进行时间序列分析和建模。

## 功能特点

- 📅 **按月份分割**: 将一年的数据分割为12个月的独立文件
- ⏰ **时间段筛选**: 可以指定每天的特定时间段（如工作时间、高峰时段等）
- 📊 **数据统计**: 提供详细的数据统计信息
- 🎯 **灵活配置**: 支持命令行参数和编程接口两种使用方式
- 📈 **可视化支持**: 包含数据可视化示例

## 文件说明

- `load_data_splitter.py`: 主要的分割工具类
- `example_usage.py`: 使用示例和可视化代码
- `README_load_splitter.md`: 本说明文档

## 快速开始

### 1. 命令行使用

```bash
# 基本用法 - 分割所有月份的全天数据
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv

# 只处理1月份，时间段为早上5点到下午17点
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --month 1 --start-hour 5 --end-hour 17

# 处理所有月份的工作时间（8:00-18:00）
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --start-hour 8 --end-hour 18

# 显示数据统计信息
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --stats

# 生成可视化图表和统计分析
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --visualize

# 指定输出目录
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --output-dir ./custom_output
```

### 2. 编程接口使用

```python
from load_data_splitter import LoadDataSplitter

# 创建分割器
splitter = LoadDataSplitter("minute_level/LoadShape1_minute_level.csv")

# 加载数据
splitter.load_data()

# 提取1月份早上5点到下午17点的数据
jan_data = splitter.split_by_month_and_time(
    month=1, 
    start_hour=5, 
    end_hour=17
)

# 处理所有月份的工作时间数据
all_data = splitter.split_all_months(
    start_hour=8,
    end_hour=18
)

# 获取统计信息
stats = splitter.get_data_statistics()
print(f"总数据点: {stats['total_points']}")
```

## 参数说明

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_file` | str | - | 输入的CSV文件路径（必需） |
| `--month` | int | - | 指定月份 (1-12)，不指定则处理所有月份 |
| `--start-hour` | int | 0 | 开始小时 (0-23) |
| `--end-hour` | int | 23 | 结束小时 (0-23) |
| `--start-minute` | int | 0 | 开始分钟 (0-59) |
| `--end-minute` | int | 59 | 结束分钟 (0-59) |
| `--output-dir` | str | - | 输出目录，默认为输入文件同目录下的monthly_splits |
| `--stats` | flag | False | 显示数据统计信息 |
| `--visualize` | flag | False | 生成数据可视化图表和统计分析 |

### 时间段配置

时间段配置支持跨天设置。例如：
- 夜间时段：`--start-hour 22 --end-hour 6` (22:00 到次日 06:00)
- 工作时间：`--start-hour 8 --end-hour 17` (08:00 到 17:00)
- 精确分钟：`--start-hour 8 --start-minute 30 --end-hour 17 --end-minute 30` (08:30 到 17:30)

## 输出格式

生成的CSV文件只包含纯数据值，每行一个负荷值（浮点数），无表头和时间戳。

示例输出文件内容：
```csv
0.535958075026019
0.5365259166242062
0.5370968688134178
0.5376707851367923
0.5382475191374687
```

**注意**: 数据按时间顺序排列，第一行对应指定时间段的开始时刻，每行代表一分钟的负荷数据。

文件命名规则：`month_MM_HHMM_to_HHMM.csv`

例如：
- `month_01_0500_to_1700.csv`: 1月份 05:00-17:00 的数据
- `month_12_0800_to_1800.csv`: 12月份 08:00-18:00 的数据

## 使用示例

### 示例1: 提取工作日高峰时段

```bash
# 早高峰 (7:00-9:00)
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --start-hour 7 --end-hour 9

# 晚高峰 (17:00-19:00)
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --start-hour 17 --end-hour 19
```

### 示例2: 提取特定月份数据

```bash
# 夏季月份 (6-8月) 的空调负荷高峰时段
for month in 6 7 8; do
    python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --month $month --start-hour 14 --end-hour 16
done
```

### 示例3: 批量处理不同时间段

```python
# 定义多个时间段
time_periods = {
    "早高峰": {"start_hour": 7, "end_hour": 9},
    "午间": {"start_hour": 11, "end_hour": 13},
    "晚高峰": {"start_hour": 17, "end_hour": 19},
    "夜间": {"start_hour": 22, "end_hour": 6}
}

splitter = LoadDataSplitter("minute_level/LoadShape1_minute_level.csv")
splitter.load_data()

for period_name, config in time_periods.items():
    for month in range(1, 13):
        data = splitter.split_by_month_and_time(
            month=month,
            output_dir=f"periods/{period_name}",
            **config
        )
```

## 数据可视化

### 1. 命令行可视化

使用 `--visualize` 参数可以直接生成可视化图表和统计分析：

```bash
# 生成单月份可视化分析
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --month 3 --start-hour 8 --end-hour 18 --visualize

# 生成全年数据可视化分析
python load_data_splitter.py minute_level/LoadShape1_minute_level.csv --start-hour 9 --end-hour 17 --visualize
```

可视化功能包含：
- **月度平均负荷对比**: 显示各月份的平均负荷水平
- **小时负荷模式**: 展示不同月份的日内负荷变化模式
- **负荷分布直方图**: 分析负荷值的分布特征
- **月度变异系数**: 评估各月份负荷的波动性
- **统计分析报告**: 生成详细的统计数据CSV文件
- **季节性分析**: 按季节分析负荷特征

输出文件：
- `load_analysis.png`: 可视化图表
- `statistical_analysis.csv`: 详细统计数据
- `seasonal_analysis.csv`: 季节性分析结果

### 2. 编程接口可视化

运行 `example_usage.py` 可以查看更多可视化示例：

```bash
python example_usage.py
```

## 注意事项

1. **数据格式**: 输入文件应为纯数值CSV文件，每行一个负荷值
2. **时间假设**: 默认假设数据从2024年1月1日00:00开始，按分钟间隔排列
3. **内存使用**: 大文件可能占用较多内存，建议在处理超大文件时分批处理
4. **时区**: 所有时间均按本地时间处理，不考虑时区转换

## 依赖库

```bash
pip install pandas numpy matplotlib pathlib
```

## 错误处理

工具包含完善的错误处理机制：
- 文件不存在检查
- 参数范围验证
- 数据格式验证
- 空数据处理

## 扩展功能

可以根据需要扩展以下功能：
- 支持其他时间间隔（小时级、秒级）
- 添加数据预处理功能（平滑、去噪等）
- 支持多种输出格式（Excel、JSON等）
- 集成更多统计分析功能

## 联系方式

如有问题或建议，请联系数据分析团队。