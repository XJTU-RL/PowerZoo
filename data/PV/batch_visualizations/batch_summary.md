
# PV数据批量可视化分析报告

## 处理概览
- 处理时间: 2025-07-27 16:50:46
- 输入目录: txt
- 输出目录: batch_visualization_fixed
- 成功处理: 4 个文件
- 处理失败: 0 个文件

## 成功处理的文件

### 202501.txt
- 输出目录: `batch_visualization_fixed/202501`
- 处理时间: 2025-07-27 16:50:14
- 状态: ✓ 成功

### 202502.txt
- 输出目录: `batch_visualization_fixed/202502`
- 处理时间: 2025-07-27 16:50:25
- 状态: ✓ 成功

### 202503.txt
- 输出目录: `batch_visualization_fixed/202503`
- 处理时间: 2025-07-27 16:50:35
- 状态: ✓ 成功

### 202504.txt
- 输出目录: `batch_visualization_fixed/202504`
- 处理时间: 2025-07-27 16:50:46
- 状态: ✓ 成功


## 输出目录结构
```
batch_visualization_fixed/
├── batch_summary.md          # 本摘要报告

├── 202501/              # 202501.txt 的分析结果
│   ├── solar_radiation/      # 太阳辐射分析
│   ├── temperature_analysis/ # 温度分析
│   ├── weather_conditions/   # 天气条件
│   ├── correlation_analysis/ # 相关性分析
│   ├── statistical_summary/  # 统计摘要
│   └── README.md            # 详细报告

├── 202502/              # 202502.txt 的分析结果
│   ├── solar_radiation/      # 太阳辐射分析
│   ├── temperature_analysis/ # 温度分析
│   ├── weather_conditions/   # 天气条件
│   ├── correlation_analysis/ # 相关性分析
│   ├── statistical_summary/  # 统计摘要
│   └── README.md            # 详细报告

├── 202503/              # 202503.txt 的分析结果
│   ├── solar_radiation/      # 太阳辐射分析
│   ├── temperature_analysis/ # 温度分析
│   ├── weather_conditions/   # 天气条件
│   ├── correlation_analysis/ # 相关性分析
│   ├── statistical_summary/  # 统计摘要
│   └── README.md            # 详细报告

├── 202504/              # 202504.txt 的分析结果
│   ├── solar_radiation/      # 太阳辐射分析
│   ├── temperature_analysis/ # 温度分析
│   ├── weather_conditions/   # 天气条件
│   ├── correlation_analysis/ # 相关性分析
│   ├── statistical_summary/  # 统计摘要
│   └── README.md            # 详细报告

```

## 使用说明
1. 每个数据文件的分析结果保存在独立的子目录中
2. 每个子目录包含完整的可视化分析图表
3. 查看各子目录中的 `README.md` 获取详细分析说明
4. 所有图表均为高分辨率PNG格式，适合报告使用

## 分析内容
每个文件的分析包括：
- **太阳辐射分析**: 全球水平辐射、直射辐射、散射辐射的时间序列和分布
- **温度分析**: 多点温度对比、日变化模式、与辐射的相关性
- **天气条件**: 风速风向、云量、湿度分布
- **相关性分析**: 关键变量间的相关矩阵热力图
- **统计摘要**: 描述性统计和数据分布箱线图
