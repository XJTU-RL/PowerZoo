# 光伏数据处理工具集

本目录包含用于处理MIDC气象数据并转换为OpenDSS兼容格式的完整工具链。

原始输入保存在 `txt/`，脚本生成的 CSV、DSS、统计报告和批量图表不再纳入版本控制。
新检出仓库后，先生成数据再运行提取或集成工具；不能直接运行缺少配套 CSV 的旧 DSS 输出。
以下命令均从本目录运行。生成物用于数据准备，不是已经通过当前任务物理验收的算例。

## 核心脚本

### 1. 数据格式转换

#### `PVdata_processing.py` - 主要数据转换脚本
将MIDC格式的气象数据转换为OpenDSS PV系统可用格式。

```bash
# 处理单个文件
python PVdata_processing.py txt/202501.txt -o csv/

# 批量处理txt目录下所有文件
python PVdata_processing.py txt/ --batch -o csv/
```

#### `daily_processor.py` - 按日分割数据
将原始数据按月份和日期分别存储。

```bash
# 处理单个文件并按日分割
python daily_processor.py txt/202501.txt -o daily_data/

# 批量处理
python daily_processor.py txt/ --batch -o daily_data/
```

#### `extract_pv_data_from_csv.py` - 时间段数据提取
从日数据中提取指定时间段的光伏数据。

```bash
# 按天分割所有月份数据（默认8-17点）
python extract_pv_data_from_csv.py --base-dir daily_data/ --output-dir hourly_segments/ --split-by-day

# 自定义时间段（6-18点）
python extract_pv_data_from_csv.py --base-dir daily_data/ --output-dir hourly_segments/ --split-by-day --start-hour 6.0 --end-hour 18.0

# 指定输出目录
python extract_pv_data_from_csv.py --base-dir daily_data/ --split-by-day --output-dir hourly_segments/
```

### 2. OpenDSS集成

#### `pv_opendss_integrator.py` - PV数据与OpenDSS集成
将处理后的PV数据集成到OpenDSS项目中。
旧工具启动时就会初始化目标目录，连列出日期也应显式传入临时目录，避免使用旧机器的默认绝对路径。

```bash
# 查看已生成数据的日期
python pv_opendss_integrator.py --pv-data-dir . --dss-project-dir /tmp/powerzoo-model-copy/ --list-dates

# 集成会写入目标模型目录；请传入临时副本，勿覆盖 node_systems 中的源模型。
python pv_opendss_integrator.py --pv-data-dir . --date 2025-01-15 --dss-project-dir /tmp/powerzoo-model-copy/
```

### 3. 数据可视化

#### `data_visualizer.py` - 单文件可视化分析
对单个数据文件进行综合可视化分析。

```bash
# 分析单个文件
python data_visualizer.py txt/202501.txt -o visualization_results/

```

#### `batch_visualizer.py` - 批量可视化处理
批量处理多个数据文件并生成综合分析。

```bash
# 批量处理txt目录下所有文件
python batch_visualizer.py txt/ -o batch_visualizations/

```

## 数据流程

```
原始MIDC数据(txt/)
    ├─ PVdata_processing.py → 月度转换结果(csv/)
    └─ daily_processor.py → 按日分割数据(daily_data/)
         ├─ extract_pv_data_from_csv.py → 时间段数据(hourly_segments/)
         └─ pv_opendss_integrator.py → OpenDSS项目副本
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
