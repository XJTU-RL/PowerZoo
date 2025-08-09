# Usage Examples - Load Data Analysis Tools

This document provides practical examples for using the load data analysis tools.

## 🚀 Quick Start

### Step 1: Navigate to the tools directory
```bash
cd data/Loads
```

### Step 2: Analyze existing minute-level data
```bash
# Analyze LoadShape1 data
python run_temporal_analysis.py --data-file minute_level/LoadShape1_minute_level.csv --label "Building Load"

# Analyze with custom output directory
python run_temporal_analysis.py --data-file minute_level/LoadShape2_minute_level.csv --label "Industrial Load" --output-dir industrial_analysis
```

### Step 3: Generate sample data analysis
```bash
# Generate and analyze sample data
python run_temporal_analysis.py --generate-sample --label "Demonstration Data"
```

## 📊 Advanced Usage Examples

### 1. Complete Workflow Example
```bash
# Assume you have hourly data files (LoadShape1.CSV, etc.)

# Step 1: Interpolate hourly data to minute-level
python unified_load_interpolator.py --plot

# Step 2: Analyze temporal patterns for each dataset
python run_temporal_analysis.py --data-file minute_level/LoadShape1_minute_level.csv --label "Residential" --output-dir residential_analysis

python run_temporal_analysis.py --data-file minute_level/LoadShape2_minute_level.csv --label "Commercial" --output-dir commercial_analysis

python run_temporal_analysis.py --data-file minute_level/LoadShape3_minute_level.csv --label "Industrial" --output-dir industrial_analysis

# Step 3: Batch analysis for comparison
python loadshape_analysis_enhanced.py . --charts
```

### 2. Research Workflow
```bash
# Generate multiple sample datasets for research
python run_temporal_analysis.py --generate-sample --label "Winter Pattern" --output-dir research/winter_analysis

python run_temporal_analysis.py --generate-sample --label "Summer Pattern" --output-dir research/summer_analysis

# Batch comparison
python loadshape_analysis_enhanced.py research --charts --output research/comparison_results
```

### 3. Quality Assessment Workflow
```bash
# Interpolate with different smoothing factors
python unified_load_interpolator.py --smooth 0.0 --output-dir minute_level/no_smooth
python unified_load_interpolator.py --smooth 0.1 --output-dir minute_level/light_smooth
python unified_load_interpolator.py --smooth 0.3 --output-dir minute_level/heavy_smooth

# Compare results
python run_temporal_analysis.py --data-file minute_level/no_smooth/LoadShape1_minute_level.csv --label "No Smoothing" --output-dir comparison/no_smooth

python run_temporal_analysis.py --data-file minute_level/light_smooth/LoadShape1_minute_level.csv --label "Light Smoothing" --output-dir comparison/light_smooth

python run_temporal_analysis.py --data-file minute_level/heavy_smooth/LoadShape1_minute_level.csv --label "Heavy Smoothing" --output-dir comparison/heavy_smooth
```

## 📈 Understanding the Results

### Daily Analysis Results
- **Peak Hour**: Hour of day with maximum average load
- **Valley Hour**: Hour of day with minimum average load  
- **Load Factor**: Ratio of average to peak load (higher = more efficient)
- **Daily Load Swing**: Difference between peak and valley loads

### Monthly Analysis Results
- **Peak Month**: Month with highest average load
- **Valley Month**: Month with lowest average load
- **Seasonal Variation**: Difference between peak and valley months
- **Annual Load Factor**: Overall system efficiency metric

### Key Visualizations
1. **Daily Load Profile**: 24-hour average pattern with variability bands
2. **Daily Heatmap**: Hour-by-month load intensity visualization
3. **Monthly Trends**: Seasonal load variations and load factors
4. **Polar Plots**: Circular visualization of daily/seasonal patterns
5. **Box Plots**: Load distribution analysis by hour/month
6. **Comprehensive Dashboard**: Combined overview of all patterns

## 🔍 Troubleshooting

### Common Issues

**Issue**: "ImportError: No module named 'data_processor'"
**Solution**: Make sure you're running scripts from within the `data/Loads/` directory.

**Issue**: "ValueError: Expected 525600 data points"
**Solution**: Ensure your minute-level data file has exactly 525,600 rows (1 year of minute data).

**Issue**: "FileNotFoundError: Data file not found"
**Solution**: Check that the file path is correct relative to the `data/Loads/` directory.

### Performance Tips

1. **Large Dataset Analysis**: Use `--output-dir` to organize results when analyzing multiple datasets
2. **Memory Optimization**: The tools are optimized for 525K data points but monitor memory usage for very large datasets
3. **Parallel Processing**: The batch analyzer uses parallel processing - adjust `--workers` parameter if needed

## 📋 Output Structure

Each analysis creates a structured output directory:

```
output_directory/
├── comprehensive_analysis.png          # Complete dashboard
├── daily_analysis/
│   ├── daily_load_profile.png         # 24-hour average pattern
│   ├── daily_load_heatmap.png         # Hour-by-month heatmap
│   ├── daily_box_plot.png             # Hourly distribution
│   └── daily_polar_plot.png           # Circular daily pattern
├── monthly_analysis/
│   ├── monthly_trends.png             # Monthly averages & load factors
│   ├── seasonal_patterns.png          # Seasonal analysis
│   ├── monthly_statistics.png         # Statistical comparison
│   └── monthly_box_plot.png           # Monthly distribution
└── reports/
    ├── temporal_analysis_results.json # Complete data (machine-readable)
    ├── temporal_analysis_report.txt   # Summary report (human-readable)
    ├── daily_patterns.csv             # Daily pattern data
    ├── monthly_patterns.csv           # Monthly pattern data
    └── analysis_log.log               # Processing log
```

## 🎯 Best Practices

1. **Data Validation**: Always check that input data has the expected number of points (525,600 for minute-level)
2. **Consistent Labeling**: Use descriptive labels to distinguish between different load types or scenarios
3. **Output Organization**: Use meaningful output directory names for easy comparison
4. **Quality Metrics**: Review interpolation quality metrics before proceeding with analysis
5. **Comparative Analysis**: Use batch processing tools for systematic comparison of multiple datasets

## 📞 Support

For questions or issues:
1. Check this documentation and README.md
2. Review the analysis log files for detailed error messages
3. Ensure all required Python packages are installed: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`