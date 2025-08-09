# 光伏数据处理工具集

本目录包含用于处理MIDC气象数据并转换为OpenDSS兼容格式的完整工具链。

## 核心脚本

### 1. 数据格式转换

#### `PVdata_processing.py` - 主要数据转换脚本
将MIDC格式的气象数据转换为OpenDSS PV系统可用格式。

```bash
# 处理单个文件
python PVdata_processing.py --input txt/202501.txt --output csv/

# 批量处理txt目录下所有文件
python PVdata_processing.py --batch --input txt/ --output csv/
```

#### `daily_processor.py` - 按日分割数据
将原始数据按月份和日期分别存储。

```bash
# 处理单个文件并按日分割
python daily_processor.py --input txt/202501.txt --output daily_data/

# 批量处理
python daily_processor.py --batch --input txt/ --output daily_data/
```

#### `extract_pv_data_from_csv.py` - 时间段数据提取
从日数据中提取指定时间段的光伏数据。

```bash
# 按天分割所有月份数据（默认8-17点）
python extract_pv_data_from_csv.py --split-by-day

# 自定义时间段（6-18点）
python extract_pv_data_from_csv.py --split-by-day --start-hour 6.0 --end-hour 18.0

# 指定输出目录
python extract_pv_data_from_csv.py --split-by-day --output-dir hourly_segments/
```

### 2. OpenDSS集成

#### `pv_opendss_integrator.py` - PV数据与OpenDSS集成
将处理后的PV数据集成到OpenDSS项目中。

```bash
# 集成指定日期的数据
python pv_opendss_integrator.py --date 2025-01-15 --dss-dir ../node_systems/34Bus_PV/

# 批量集成多个日期
python pv_opendss_integrator.py --batch --start-date 2025-01-01 --end-date 2025-01-31 --dss-dir ../node_systems/34Bus_PV/

# 查看可用日期
python pv_opendss_integrator.py --list-dates
```

### 3. 数据可视化

#### `data_visualizer.py` - 单文件可视化分析
对单个数据文件进行综合可视化分析。

```bash
# 分析单个文件
python data_visualizer.py --input txt/202501.txt --output visualization_results/

# 生成完整报告
python data_visualizer.py --input txt/202501.txt --output visualization_results/ --full-report
```

#### `batch_visualizer.py` - 批量可视化处理
批量处理多个数据文件并生成综合分析。

```bash
# 批量处理txt目录下所有文件
python batch_visualizer.py --input txt/ --output batch_visualizations/

# 生成HTML索引页面
python batch_visualizer.py --input txt/ --output batch_visualizations/ --create-index
```

## 数据流程

```
原始MIDC数据(txt/) 
    ↓ PVdata_processing.py
OpenDSS格式数据(csv/)
    ↓ daily_processor.py  
按日分割数据(daily_data/)
    ↓ extract_pv_data_from_csv.py
时间段数据(hourly_segments/)
    ↓ pv_opendss_integrator.py
OpenDSS项目集成
```

## 输出目录结构

- `csv/` - OpenDSS兼容格式的转换结果
- `daily_data/` - 按月份和日期组织的数据
- `hourly_segments/` - 按时间段提取的数据
- `batch_visualizations/` - 批量可视化结果

## 数据字段说明

详细的数据字段定义请参考 `filed_definition.md` 文件。主要包括：
- 太阳辐射数据（全球水平辐射、直射辐射等）
- 温度数据（干球温度、湿球温度等）
- 风速风向数据
- 云量和湿度数据

## 注意事项

1. 确保输入数据格式符合MIDC标准
2. 处理大文件时建议使用批量模式
3. 可视化功能需要安装matplotlib、seaborn等依赖
4. OpenDSS集成需要确保目标DSS项目目录结构正确

## 依赖包

```bash
pip install pandas numpy matplotlib seaborn pathlib
```