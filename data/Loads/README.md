# Load Data Analysis and Processing Tools

This directory contains comprehensive tools for analyzing and processing load data, including interpolation, temporal trend analysis, and batch processing capabilities.

**🎯 All tools are now self-contained within the `data/Loads/` directory - no external dependencies on utils or examples folders.**

## 📁 File Structure

```
data/Loads/
├── __init__.py                          # Package initialization
├── data_processor.py                   # Data processing utilities
├── visualization_utils.py              # Visualization tools
├── temporal_trend_analyzer.py          # Main temporal analysis engine
├── run_temporal_analysis.py            # Temporal analysis runner script
├── unified_load_interpolator.py        # Unified interpolation tool
├── loadshape_analyzer.py              # Basic load analysis tool
├── loadshape_analysis_enhanced.py     # Batch processing tool
├── README.md                           # This documentation
└── minute_level/                       # Interpolated minute-level data
    ├── LoadShape1_minute_level.csv
    ├── LoadShape2_minute_level.csv
    └── LoadShape3_minute_level.csv
```

## 🔧 Available Tools

### 1. Unified Load Interpolator (`unified_load_interpolator.py`)
**Purpose**: Convert hourly load data (8760 points) to minute-level data (525600 points) using advanced cubic spline interpolation.

**Features**:
- High-quality cubic spline interpolation with enhanced boundary handling  
- Comprehensive validation metrics (MAE, RMSE, correlation, R²)
- Batch processing with parallel execution support
- Optional smoothing to reduce high-frequency noise
- Detailed quality assessment and reporting

**Usage Examples**:
```bash
# Process all LoadShape*.* files
python data/Loads/unified_load_interpolator.py

# Process with light smoothing
python data/Loads/unified_load_interpolator.py --smooth 0.1

# Process single file
python data/Loads/unified_load_interpolator.py --single LoadShape1.CSV

# Generate comparison chart
python data/Loads/unified_load_interpolator.py --plot

# Custom directories
python data/Loads/unified_load_interpolator.py --input-dir custom/input --output-dir custom/output
```

### 2. Temporal Trend Analyzer (`temporal_trend_analyzer.py`)
**Purpose**: Analyze minute-level load data to extract daily and monthly patterns and trends.

**Features**:
- Daily pattern analysis (24-hour load profiles)
- Monthly trend analysis (seasonal variations)
- Advanced statistical analysis and pattern recognition
- Professional visualization with 8 chart types
- Comprehensive reporting in multiple formats

**Usage Examples**:
```bash
# Analyze with sample data
python run_temporal_analysis.py --generate-sample --label "Sample Data"

# Analyze existing minute-level data
python run_temporal_analysis.py --data-file minute_level/LoadShape1_minute_level.csv

# Custom output directory
python run_temporal_analysis.py --generate-sample --output-dir results/temporal_analysis
```

### 3. Batch Loadshape Analyzer (`loadshape_analysis_enhanced.py`)
**Purpose**: Batch analysis of multiple load data files with parallel processing.

**Features**:
- Automatic detection of data granularity (hourly/minute-level)
- Parallel processing for improved performance
- Comparative analysis between multiple files
- Statistical visualization and reporting

**Usage Examples**:
```bash
# Analyze all files in directory
python data/Loads/loadshape_analysis_enhanced.py data/Loads

# With comparison charts
python data/Loads/loadshape_analysis_enhanced.py data/Loads --charts

# Custom output directory
python data/Loads/loadshape_analysis_enhanced.py data/Loads -o results/batch_analysis
```

## 📊 Output Structure

### Interpolation Results (`data/Loads/minute_level/`)
```
minute_level/
├── LoadShape1_minute_level.csv          # Interpolated data
├── LoadShape2_minute_level.csv          # ...
├── comprehensive_interpolation_report.txt  # Detailed report
└── interpolation_comparison_*.png       # Comparison charts
```

### Temporal Analysis Results (`data/Loads/temporal_analysis/`)
```
temporal_analysis/
├── daily_analysis/
│   ├── daily_load_profile.png           # 24-hour average pattern
│   ├── daily_load_heatmap.png          # Hour-by-month heatmap
│   ├── daily_box_plot.png              # Hourly load distribution
│   └── daily_polar_plot.png            # Circular daily pattern
├── monthly_analysis/
│   ├── monthly_trends.png              # Monthly load trends
│   ├── seasonal_patterns.png           # Seasonal analysis
│   ├── monthly_statistics.png          # Statistical comparison
│   └── monthly_box_plot.png            # Monthly distribution
├── temporal_analysis_results.json      # Complete results data
├── daily_patterns.csv                  # Daily pattern data
├── monthly_patterns.csv                # Monthly pattern data
└── temporal_analysis_report.txt        # Human-readable report
```

## 🎯 Typical Workflow

1. **Start with hourly data**: Place LoadShape*.CSV files in `data/Loads/`

2. **Interpolate to minute-level**:
   ```bash
   python unified_load_interpolator.py --plot
   ```

3. **Analyze temporal patterns**:
   ```bash
   python run_temporal_analysis.py --data-file minute_level/LoadShape1_minute_level.csv
   ```

4. **Batch analysis for comparison**:
   ```bash
   python loadshape_analysis_enhanced.py . --charts
   ```

## 📈 Quality Metrics

### Interpolation Quality Assessment
- **EXCELLENT**: Correlation > 0.99, MAE < 0.01, R² > 0.98
- **GOOD**: Correlation > 0.95, MAE < 0.02, R² > 0.90  
- **FAIR**: Correlation > 0.90, MAE < 0.05, R² > 0.80
- **NEEDS IMPROVEMENT**: Below fair thresholds

### Statistical Metrics Provided
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Correlation Coefficient
- R-squared (Coefficient of Determination)
- Peak-valley ratios and load factors

## 🛠️ Dependencies

Required Python packages:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

## 📝 Notes

- All tools follow PowerZoo project coding standards
- Use tabs for indentation, English for chart labels
- All outputs are saved with high-resolution (300 DPI) for publication quality
- Tools are optimized for large datasets (525K+ data points)
- Comprehensive error handling and logging included

## 🚀 Advanced Features

- **Parallel Processing**: Automatic multi-core utilization for batch operations
- **Memory Optimization**: Efficient handling of large datasets
- **Intelligent Sampling**: Smart data reduction for visualization without losing patterns
- **Enhanced Boundary Handling**: Seamless year-end transitions in interpolation
- **Professional Reporting**: Publication-ready charts and comprehensive documentation