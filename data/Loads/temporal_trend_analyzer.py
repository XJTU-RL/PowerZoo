#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Temporal Trend Analyzer for PowerZoo project.
分钟级负荷数据时间趋势分析器

This module provides a complete solution for analyzing minute-level
load data to extract daily and monthly temporal patterns with
statistical insights and professional visualizations.

Author: PowerZoo Team
Date: 2025-07-30
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Import local modules
from data_processor import LoadDataProcessor
from visualization_utils import TrendVisualizer


class TemporalTrendAnalyzer:
	"""
	Comprehensive temporal trend analyzer for minute-level load data.
	
	Performs daily and monthly trend analysis with statistical insights,
	visualization generation, and comprehensive reporting capabilities.
	"""
	
	def __init__(self, output_dir: str = "data/Loads/temporal_analysis"):
		"""
		Initialize the temporal trend analyzer.
		
		Args:
			output_dir: Output directory for results
		"""
		self.output_dir = Path(output_dir)
		self.reports_dir = self.output_dir / "reports"
		
		# Initialize components
		self.data_processor = LoadDataProcessor()
		self.visualizer = TrendVisualizer(str(self.output_dir))
		
		# Create directories
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.reports_dir.mkdir(parents=True, exist_ok=True)
		
		# Setup logging
		self._setup_logging()
		
		# Analysis results storage
		self.daily_patterns = None
		self.monthly_patterns = None
		self.overall_stats = None
		self.analysis_metadata = {
			'analysis_timestamp': None,
			'data_points': None,
			'data_range': None,
		}
	
	def _setup_logging(self) -> None:
		"""Setup logging configuration."""
		log_file = self.reports_dir / 'analysis_log.log'
		
		# Create logger
		self.logger = logging.getLogger(f'{__name__}_{id(self)}')
		self.logger.setLevel(logging.INFO)
		
		# Clear existing handlers to avoid duplicates
		self.logger.handlers.clear()
		
		# Create handlers
		file_handler = logging.FileHandler(log_file)
		console_handler = logging.StreamHandler()
		
		# Create formatter
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		file_handler.setFormatter(formatter)
		console_handler.setFormatter(formatter)
		
		# Add handlers
		self.logger.addHandler(file_handler)
		self.logger.addHandler(console_handler)
	
	def analyze_temporal_trends(self, data: Union[np.ndarray, List, str], 
								data_label: str = "Load Data",
								start_date: str = "2023-01-01",
								output_dir: Optional[str] = None) -> Dict:
		"""
		Perform comprehensive temporal trend analysis.
		
		Args:
			data: Input load data (array, list, or file path)
			data_label: Label for the data
			start_date: Starting date for time index
			output_dir: Optional custom output directory
			
		Returns:
			dict: Complete analysis results
		"""
		if output_dir:
			self.output_dir = Path(output_dir)
			self.reports_dir = self.output_dir / "reports"
			self.output_dir.mkdir(parents=True, exist_ok=True)
			self.reports_dir.mkdir(parents=True, exist_ok=True)
			# Reinitialize visualizer with new output directory
			self.visualizer = TrendVisualizer(str(self.output_dir))
		
		self.logger.info(f"Starting temporal trend analysis for {data_label}")
		
		# Load and validate data
		load_data = self._load_data(data)
		self.logger.info(f"Data loaded: {len(load_data)} points")
		
		# Store metadata
		self.analysis_metadata.update({
			'analysis_timestamp': datetime.now().isoformat(),
			'data_points': len(load_data),
			'data_label': data_label,
			'start_date': start_date,
		})
		
		# Reshape data for analysis
		data_2d = self.data_processor.reshape_to_minutes(load_data)
		self.logger.info("Data reshaped to (365, 1440) format")
		
		# Extract patterns
		self.daily_patterns = self.data_processor.extract_daily_patterns(data_2d)
		self.monthly_patterns = self.data_processor.extract_monthly_patterns(data_2d)
		self.overall_stats = self.data_processor.calculate_statistical_metrics(load_data)
		
		self.logger.info("Pattern extraction completed")
		
		# Generate visualizations
		self._generate_all_visualizations()
		self.logger.info("Visualizations generated")
		
		# Generate reports
		analysis_results = self._generate_comprehensive_report()
		self.logger.info("Analysis report generated")
		
		# Export patterns to CSV
		self.export_patterns_to_csv()
		
		return analysis_results
	
	def _load_data(self, data: Union[np.ndarray, List, str]) -> np.ndarray:
		"""
		Load data from various input formats.
		
		Args:
			data: Input data
			
		Returns:
			np.ndarray: Processed data array
		"""
		if isinstance(data, str):
			# Load from file
			data_path = Path(data)
			if data_path.suffix.lower() == '.csv':
				df = pd.read_csv(data_path, header=None)
				data_array = df.iloc[:, 0].values  # Use first column as load data
			elif data_path.suffix.lower() == '.npy':
				data_array = np.load(data_path)
			else:
				raise ValueError(f"Unsupported file format: {data_path.suffix}")
		else:
			data_array = np.array(data)
		
		# Validate data length
		if len(data_array) != 525600:
			raise ValueError(f"Expected 525600 data points for minute-level analysis, got {len(data_array)}")
		
		return data_array
	
	def _generate_all_visualizations(self) -> None:
		"""Generate all visualization plots."""
		# Daily trend visualizations
		self.visualizer.plot_daily_trends(self.daily_patterns)
		
		# Monthly trend visualizations
		self.visualizer.plot_monthly_trends(self.monthly_patterns)
		
		# Combined analysis visualization
		self.visualizer.plot_combined_analysis(
			self.daily_patterns, self.monthly_patterns, self.overall_stats
		)
	
	def _generate_comprehensive_report(self) -> Dict:
		"""
		Generate comprehensive analysis report.
		
		Returns:
			dict: Complete analysis results
		"""
		# Prepare daily insights
		daily_insights = self._analyze_daily_insights()
		
		# Prepare monthly insights
		monthly_insights = self._analyze_monthly_insights()
		
		# Prepare overall insights
		overall_insights = self._analyze_overall_insights()
		
		# Compile results
		analysis_results = {
			'metadata': self.analysis_metadata,
			'overall_statistics': self.overall_stats,
			'daily_analysis': {
				'patterns': self._serialize_patterns(self.daily_patterns),
				'insights': daily_insights,
				'statistics': self.daily_patterns['statistics']
			},
			'monthly_analysis': {
				'patterns': self._serialize_patterns(self.monthly_patterns),
				'insights': monthly_insights,
				'statistics': self.monthly_patterns['statistics']
			},
			'overall_insights': overall_insights,
		}
		
		# Save results to JSON
		self._save_json_report(analysis_results)
		
		# Generate text report
		self._generate_text_report(analysis_results)
		
		return analysis_results
	
	def _analyze_daily_insights(self) -> Dict[str, str]:
		"""Analyze daily patterns and generate insights."""
		patterns = self.daily_patterns
		
		# Find peak and valley characteristics
		peak_hour = np.argmax(patterns['hourly_means'])
		valley_hour = np.argmin(patterns['hourly_means'])
		peak_load = patterns['hourly_means'][peak_hour]
		valley_load = patterns['hourly_means'][valley_hour]
		
		# Calculate daily load factor
		daily_load_factor = np.mean(patterns['hourly_means']) / peak_load
		
		# Find most variable hours
		most_variable_hour = np.argmax(patterns['hourly_stds'])
		least_variable_hour = np.argmin(patterns['hourly_stds'])
		
		# Peak/valley timing analysis
		common_peak_hour = int(np.median(patterns['peak_hours']))
		common_valley_hour = int(np.median(patterns['valley_hours']))
		
		insights = {
			'peak_characteristics': f"Daily peak typically occurs at {peak_hour:02d}:00 with {peak_load:.2f} MW average load",
			'valley_characteristics': f"Daily valley typically occurs at {valley_hour:02d}:00 with {valley_load:.2f} MW average load",
			'load_factor': f"Daily load factor is {daily_load_factor:.3f}, indicating {'high' if daily_load_factor > 0.7 else 'moderate' if daily_load_factor > 0.5 else 'low'} utilization",
			'variability': f"Hour {most_variable_hour:02d}:00 shows highest variability (σ={patterns['hourly_stds'][most_variable_hour]:.2f}), hour {least_variable_hour:02d}:00 shows lowest (σ={patterns['hourly_stds'][least_variable_hour]:.2f})",
			'timing_consistency': f"Peak hours typically occur around {common_peak_hour:02d}:00, valley hours around {common_valley_hour:02d}:00",
			'load_swing': f"Daily load swing: {(peak_load - valley_load):.2f} MW ({(peak_load - valley_load)/valley_load*100:.1f}% of valley load)"
		}
		
		return insights
	
	def _analyze_monthly_insights(self) -> Dict[str, str]:
		"""Analyze monthly patterns and generate insights."""
		patterns = self.monthly_patterns
		months = ['January', 'February', 'March', 'April', 'May', 'June',
				  'July', 'August', 'September', 'October', 'November', 'December']
		
		# Find seasonal characteristics
		peak_month = np.argmax(patterns['monthly_means'])
		valley_month = np.argmin(patterns['monthly_means'])
		peak_load = patterns['monthly_means'][peak_month]
		valley_load = patterns['monthly_means'][valley_month]
		
		# Most/least variable months
		most_variable_month = np.argmax(patterns['monthly_stds'])
		least_variable_month = np.argmin(patterns['monthly_stds'])
		
		# Load factor analysis
		best_lf_month = np.argmax(patterns['monthly_load_factors'])
		worst_lf_month = np.argmin(patterns['monthly_load_factors'])
		
		# Seasonal trends
		summer_months = [5, 6, 7]  # Jun, Jul, Aug
		winter_months = [11, 0, 1]  # Dec, Jan, Feb
		summer_avg = np.mean([patterns['monthly_means'][i] for i in summer_months])
		winter_avg = np.mean([patterns['monthly_means'][i] for i in winter_months])
		
		insights = {
			'seasonal_peak': f"{months[peak_month]} shows highest average load ({peak_load:.2f} MW)",
			'seasonal_valley': f"{months[valley_month]} shows lowest average load ({valley_load:.2f} MW)",
			'seasonal_swing': f"Seasonal load variation: {(peak_load - valley_load):.2f} MW ({(peak_load - valley_load)/valley_load*100:.1f}% difference)",
			'variability_patterns': f"{months[most_variable_month]} is most variable (σ={patterns['monthly_stds'][most_variable_month]:.2f}), {months[least_variable_month]} is most stable (σ={patterns['monthly_stds'][least_variable_month]:.2f})",
			'load_factor_trends': f"Best load factor in {months[best_lf_month]} ({patterns['monthly_load_factors'][best_lf_month]:.3f}), worst in {months[worst_lf_month]} ({patterns['monthly_load_factors'][worst_lf_month]:.3f})",
			'summer_winter_comparison': f"Summer average: {summer_avg:.2f} MW, Winter average: {winter_avg:.2f} MW ({'Summer' if summer_avg > winter_avg else 'Winter'} dominant by {abs(summer_avg - winter_avg):.2f} MW)"
		}
		
		return insights
	
	def _analyze_overall_insights(self) -> Dict[str, str]:
		"""Analyze overall system characteristics."""
		stats = self.overall_stats
		
		# Classification based on standard metrics
		load_factor_category = (
			"High" if stats['load_factor'] > 0.7 else
			"Moderate" if stats['load_factor'] > 0.5 else "Low"
		)
		
		variability_category = (
			"High" if stats['coefficient_of_variation'] > 0.3 else
			"Moderate" if stats['coefficient_of_variation'] > 0.15 else "Low"
		)
		
		# Peak to average ratio
		peak_to_avg_ratio = stats['max'] / stats['mean']
		
		insights = {
			'system_utilization': f"System shows {load_factor_category.lower()} utilization with load factor of {stats['load_factor']:.3f}",
			'load_variability': f"Load variability is {variability_category.lower()} (CV = {stats['coefficient_of_variation']:.3f})",
			'peak_characteristics': f"Peak demand is {peak_to_avg_ratio:.2f}x the average load, indicating {'high peak stress' if peak_to_avg_ratio > 2.0 else 'moderate peak stress' if peak_to_avg_ratio > 1.5 else 'low peak stress'}",
			'distribution_shape': f"Load distribution shows {'positive' if stats['skewness'] > 0 else 'negative'} skew ({stats['skewness']:.3f}) and {'heavy' if stats['kurtosis'] > 1 else 'light'} tails (kurtosis = {stats['kurtosis']:.3f})",
			'operational_range': f"Operational range spans {stats['max'] - stats['min']:.2f} MW from {stats['min']:.2f} MW to {stats['max']:.2f} MW",
		}
		
		return insights
	
	def _serialize_patterns(self, patterns: Dict) -> Dict:
		"""Convert numpy arrays to lists for JSON serialization."""
		serialized = {}
		for key, value in patterns.items():
			if isinstance(value, np.ndarray):
				serialized[key] = value.tolist()
			elif isinstance(value, list) and any(isinstance(item, np.ndarray) for item in value):
				serialized[key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
			else:
				serialized[key] = value
		return serialized
	
	def _save_json_report(self, results: Dict) -> None:
		"""Save analysis results to JSON file."""
		json_file = self.reports_dir / 'temporal_analysis_results.json'
		with open(json_file, 'w') as f:
			json.dump(results, f, indent=2)
		self.logger.info(f"JSON report saved to {json_file}")
	
	def _generate_text_report(self, results: Dict) -> None:
		"""Generate human-readable text report."""
		report_file = self.reports_dir / 'temporal_analysis_report.txt'
		
		with open(report_file, 'w') as f:
			f.write("=" * 80 + "\n")
			f.write("COMPREHENSIVE TEMPORAL LOAD ANALYSIS REPORT\n")
			f.write("=" * 80 + "\n\n")
			
			# Metadata
			f.write("ANALYSIS METADATA\n")
			f.write("-" * 40 + "\n")
			metadata = results['metadata']
			f.write(f"Analysis Date: {metadata['analysis_timestamp']}\n")
			f.write(f"Data Label: {metadata['data_label']}\n")
			f.write(f"Data Points: {metadata['data_points']:,}\n")
			f.write(f"Date Range: Starting {metadata['start_date']}\n\n")
			
			# Overall Statistics
			f.write("OVERALL STATISTICS\n")
			f.write("-" * 40 + "\n")
			stats = results['overall_statistics']
			f.write(f"Mean Load: {stats['mean']:.2f} MW\n")
			f.write(f"Peak Load: {stats['max']:.2f} MW\n")
			f.write(f"Minimum Load: {stats['min']:.2f} MW\n")
			f.write(f"Standard Deviation: {stats['std']:.2f} MW\n")
			f.write(f"Load Factor: {stats['load_factor']:.3f}\n")
			f.write(f"Coefficient of Variation: {stats['coefficient_of_variation']:.3f}\n")
			f.write(f"Skewness: {stats['skewness']:.3f}\n")
			f.write(f"Kurtosis: {stats['kurtosis']:.3f}\n\n")
			
			# Daily Analysis Insights
			f.write("DAILY ANALYSIS INSIGHTS\n")
			f.write("-" * 40 + "\n")
			daily_insights = results['daily_analysis']['insights']
			for key, insight in daily_insights.items():
				f.write(f"• {insight}\n")
			f.write("\n")
			
			# Monthly Analysis Insights
			f.write("MONTHLY ANALYSIS INSIGHTS\n")
			f.write("-" * 40 + "\n")
			monthly_insights = results['monthly_analysis']['insights']
			for key, insight in monthly_insights.items():
				f.write(f"• {insight}\n")
			f.write("\n")
			
			# Overall Insights
			f.write("OVERALL SYSTEM INSIGHTS\n")
			f.write("-" * 40 + "\n")
			overall_insights = results['overall_insights']
			for key, insight in overall_insights.items():
				f.write(f"• {insight}\n")
			f.write("\n")
			
			# File locations
			f.write("OUTPUT FILES GENERATED\n")
			f.write("-" * 40 + "\n")
			f.write(f"• Comprehensive Analysis: {self.output_dir}/comprehensive_analysis.png\n")
			f.write(f"• Daily Load Profile: {self.output_dir}/daily_analysis/daily_load_profile.png\n")
			f.write(f"• Daily Patterns Heatmap: {self.output_dir}/daily_analysis/daily_load_heatmap.png\n")
			f.write(f"• Daily Box Plot: {self.output_dir}/daily_analysis/daily_box_plot.png\n")
			f.write(f"• Daily Polar Plot: {self.output_dir}/daily_analysis/daily_polar_plot.png\n")
			f.write(f"• Monthly Trends: {self.output_dir}/monthly_analysis/monthly_trends.png\n")
			f.write(f"• Seasonal Patterns: {self.output_dir}/monthly_analysis/seasonal_patterns.png\n")
			f.write(f"• Monthly Statistics: {self.output_dir}/monthly_analysis/monthly_statistics.png\n")
			f.write(f"• Monthly Box Plot: {self.output_dir}/monthly_analysis/monthly_box_plot.png\n")
			f.write(f"• JSON Results: {self.reports_dir}/temporal_analysis_results.json\n")
			f.write(f"• Daily Patterns CSV: {self.reports_dir}/daily_patterns.csv\n")
			f.write(f"• Monthly Patterns CSV: {self.reports_dir}/monthly_patterns.csv\n")
			f.write(f"• Analysis Log: {self.reports_dir}/analysis_log.log\n")
			
		self.logger.info(f"Text report saved to {report_file}")
	
	def export_patterns_to_csv(self) -> None:
		"""Export analysis patterns to CSV files for further analysis."""
		if self.daily_patterns is None or self.monthly_patterns is None:
			raise ValueError("No analysis results available. Run analyze_temporal_trends first.")
		
		# Export daily patterns
		daily_csv_file = self.reports_dir / 'daily_patterns.csv'
		daily_df = pd.DataFrame({
			'hour': range(24),
			'hourly_means': self.daily_patterns['hourly_means'],
			'hourly_stds': self.daily_patterns['hourly_stds'],
			'hourly_mins': self.daily_patterns['hourly_mins'],
			'hourly_maxs': self.daily_patterns['hourly_maxs'],
			'hourly_medians': self.daily_patterns['hourly_medians'],
		})
		daily_df.to_csv(daily_csv_file, index=False)
		
		# Export monthly patterns
		monthly_csv_file = self.reports_dir / 'monthly_patterns.csv'
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
				  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		monthly_df = pd.DataFrame({
			'month': months,
			'monthly_means': self.monthly_patterns['monthly_means'],
			'monthly_stds': self.monthly_patterns['monthly_stds'],
			'monthly_mins': self.monthly_patterns['monthly_mins'],
			'monthly_maxs': self.monthly_patterns['monthly_maxs'],
			'monthly_medians': self.monthly_patterns['monthly_medians'],
			'monthly_load_factors': self.monthly_patterns['monthly_load_factors'],
			'monthly_peak_demands': self.monthly_patterns['monthly_peak_demands'],
		})
		monthly_df.to_csv(monthly_csv_file, index=False)
		
		self.logger.info(f"Pattern data exported to CSV files: {daily_csv_file}, {monthly_csv_file}")


def main():
	"""
	Demo function showing how to use the TemporalTrendAnalyzer.
	"""
	# Create analyzer instance
	analyzer = TemporalTrendAnalyzer()
	
	# Generate sample data for demonstration
	print("Generating sample minute-level load data for demonstration...")
	np.random.seed(42)  # For reproducible results
	
	# Create realistic load pattern
	minutes_per_year = 525600
	time_index = np.arange(minutes_per_year)
	
	# Base load with daily and seasonal patterns
	base_load = 100
	daily_pattern = 20 * np.sin(2 * np.pi * (time_index % 1440) / 1440 - np.pi/2) + 10
	seasonal_pattern = 15 * np.sin(2 * np.pi * time_index / minutes_per_year)
	noise = np.random.normal(0, 5, minutes_per_year)
	
	sample_data = base_load + daily_pattern + seasonal_pattern + noise
	sample_data = np.maximum(sample_data, 20)  # Ensure minimum load
	
	print(f"Sample data created: {len(sample_data)} points")
	
	# Run analysis
	results = analyzer.analyze_temporal_trends(
		data=sample_data,
		data_label="Sample Load Data",
		start_date="2023-01-01"
	)
	
	print("\nAnalysis completed successfully!")
	print(f"Results saved to: {analyzer.output_dir}")
	print("\nGenerated files:")
	print("- Comprehensive analysis plots")
	print("- Daily trend visualizations")
	print("- Monthly trend visualizations")
	print("- Statistical reports (JSON and text)")
	print("- Pattern data (CSV)")
	print("- Analysis log")


if __name__ == "__main__":
	main()