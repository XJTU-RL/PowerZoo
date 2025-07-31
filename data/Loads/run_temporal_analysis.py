#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal Trend Analysis Runner Script
时间趋势分析运行脚本

This script demonstrates how to use the TemporalTrendAnalyzer to analyze
minute-level load data for daily and monthly patterns.

Author: PowerZoo Team
Date: 2025-07-30
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Import the temporal trend analyzer
from temporal_trend_analyzer import TemporalTrendAnalyzer


def generate_sample_data() -> np.ndarray:
	"""
	Generate sample minute-level load data for demonstration
	
	Returns:
		numpy array with 525600 data points (1 year of minute-level data)
	"""
	print("Generating sample minute-level load data...")
	
	# Create realistic load pattern with daily and seasonal variations
	minutes_per_year = 525600
	time_array = np.arange(minutes_per_year)
	
	# Base load with seasonal variation
	day_of_year = (time_array // 1440) % 365  # Day of year (0-364)
	seasonal_factor = 0.8 + 0.4 * np.cos(2 * np.pi * day_of_year / 365)  # Summer peak
	
	# Daily pattern
	minute_of_day = time_array % 1440  # Minute of day (0-1439)
	daily_pattern = (
		0.6 +  # Base load
		0.3 * np.cos(2 * np.pi * (minute_of_day - 480) / 1440) +  # Daily cycle, peak at 8AM
		0.1 * np.cos(4 * np.pi * (minute_of_day - 600) / 1440)    # Secondary peak
	)
	
	# Weekly pattern (lower on weekends)
	day_of_week = ((time_array // 1440) % 7)  # Day of week (0-6)
	weekly_factor = np.where(day_of_week < 5, 1.0, 0.85)  # Lower on weekends
	
	# Combine patterns and add noise
	load_data = seasonal_factor * daily_pattern * weekly_factor
	load_data += 0.02 * np.random.normal(0, 1, minutes_per_year)  # Add noise
	load_data = np.maximum(load_data, 0.1)  # Ensure minimum load
	
	# Scale to typical MW values
	load_data = load_data * 100  # Scale to ~100 MW range
	
	print(f"Generated {len(load_data)} data points")
	print(f"Load range: {np.min(load_data):.3f} - {np.max(load_data):.3f} MW")
	
	return load_data


def main():
	"""Main function"""
	parser = argparse.ArgumentParser(
		description="Temporal Trend Analysis Tool",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Analyze with sample data
  python run_temporal_analysis.py --generate-sample --label "Sample Data"
  
  # Analyze existing minute-level data
  python run_temporal_analysis.py --data-file minute_level/LoadShape1_minute_level.csv
  
  # Custom output directory
  python run_temporal_analysis.py --generate-sample --output-dir temporal_results
		"""
	)
	
	parser.add_argument('--data-file', type=str, 
					   help='Path to minute-level load data file (CSV format)')
	parser.add_argument('--generate-sample', action='store_true',
					   help='Generate and analyze sample data')
	parser.add_argument('--label', type=str, default='Load Data',
					   help='Label for the analysis (default: Load Data)')
	parser.add_argument('--output-dir', type=str, default='temporal_analysis',
					   help='Output directory for results (default: temporal_analysis)')
	
	args = parser.parse_args()
	
	try:
		# Determine current directory
		current_dir = Path(__file__).parent
		
		if args.generate_sample:
			# Generate sample data
			load_data = generate_sample_data()
			data_label = f"Sample {args.label}"
		elif args.data_file:
			# Load data from file
			data_path = Path(args.data_file)
			
			# If path is relative, make it relative to current directory
			if not data_path.is_absolute():
				data_path = current_dir / data_path
			
			if not data_path.exists():
				print(f"Error: Data file not found: {data_path}")
				return 1
			
			print(f"Loading data from: {data_path}")
			
			# Try to load the data
			try:
				df = pd.read_csv(data_path, header=None)
				load_data = df.iloc[:, 0].values
			except Exception as e:
				print(f"Error loading data file: {e}")
				return 1
			
			# Validate data length
			if len(load_data) != 525600:
				print(f"Warning: Expected 525600 data points, got {len(load_data)}")
				if len(load_data) < 525600:
					print("Error: Insufficient data for temporal analysis")
					return 1
			
			data_label = f"{data_path.stem} - {args.label}"
		else:
			print("Error: Either --data-file or --generate-sample must be specified")
			return 1
		
		# Set output directory
		output_dir = current_dir / args.output_dir
		
		print(f"\n{'='*60}")
		print(f"TEMPORAL TREND ANALYSIS")
		print(f"{'='*60}")
		print(f"Data Label: {data_label}")
		print(f"Data Points: {len(load_data):,}")
		print(f"Output Directory: {output_dir}")
		print(f"{'='*60}\n")
		
		# Initialize analyzer
		analyzer = TemporalTrendAnalyzer(output_dir=str(output_dir))
		
		# Run temporal trend analysis
		print("Starting temporal trend analysis...")
		results = analyzer.analyze_temporal_trends(
			data=load_data,
			data_label=data_label,
			start_date="2023-01-01"
		)
		
		print(f"\n{'='*60}")
		print(f"ANALYSIS COMPLETED SUCCESSFULLY!")
		print(f"{'='*60}")
		
		# Print summary statistics
		if 'daily_analysis' in results:
			daily_stats = results['daily_analysis']['statistics']
			print(f"\nDaily Pattern Summary:")
			print(f"  Peak Hour: {daily_stats['peak_hour']:02d}:00")
			print(f"  Valley Hour: {daily_stats['valley_hour']:02d}:00")
			print(f"  Peak Load: {daily_stats['peak_load']:.3f} MW")
			print(f"  Valley Load: {daily_stats['valley_load']:.3f} MW")
			print(f"  Daily Load Factor: {daily_stats['load_factor']:.3f}")
		
		if 'monthly_analysis' in results:
			monthly_stats = results['monthly_analysis']['statistics']
			print(f"\nMonthly Pattern Summary:")
			print(f"  Peak Month: {monthly_stats['peak_month']}")
			print(f"  Valley Month: {monthly_stats['valley_month']}")
			print(f"  Annual Load Factor: {monthly_stats['annual_load_factor']:.3f}")
			print(f"  Seasonal Variation: {monthly_stats['seasonal_variation']:.3f} MW")
		
		# List generated files
		print(f"\nGenerated Files:")
		result_files = list(output_dir.rglob("*"))
		result_files = [f for f in result_files if f.is_file()]
		for file_path in sorted(result_files):
			rel_path = file_path.relative_to(output_dir)
			print(f"  {rel_path}")
		
		print(f"\n✓ All results saved to: {output_dir}")
		
		return 0
		
	except Exception as e:
		print(f"Error during analysis: {e}")
		import traceback
		traceback.print_exc()
		return 1


if __name__ == "__main__":
	sys.exit(main())