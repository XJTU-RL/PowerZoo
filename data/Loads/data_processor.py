#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load Data Processor for Temporal Analysis
负荷数据处理器，用于时间序列分析

Author: PowerZoo Team
Date: 2025-07-30
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy import stats


class LoadDataProcessor:
	"""
	Load data processor for temporal analysis
	
	Processes minute-level load data to extract daily and monthly patterns
	with comprehensive statistical analysis.
	"""
	
	def __init__(self):
		"""Initialize the data processor"""
		self.minutes_per_hour = 60
		self.hours_per_day = 24
		self.days_per_year = 365
		self.minutes_per_day = 1440
		self.minutes_per_year = 525600
	
	def reshape_to_minutes(self, data: np.ndarray) -> np.ndarray:
		"""
		Reshape 1D minute data to 2D array (days x minutes_per_day)
		
		Args:
			data: 1D array of minute-level data (525600 points)
			
		Returns:
			2D array of shape (365, 1440)
		"""
		if len(data) != self.minutes_per_year:
			raise ValueError(f"Expected {self.minutes_per_year} data points, got {len(data)}")
		
		return data.reshape(self.days_per_year, self.minutes_per_day)
	
	def extract_daily_patterns(self, data_2d: np.ndarray) -> Dict:
		"""
		Extract daily load patterns and statistics
		
		Args:
			data_2d: 2D array of shape (365, 1440)
			
		Returns:
			Dictionary containing daily pattern analysis
		"""
		# Convert to hourly data for daily analysis
		hourly_data = self._convert_to_hourly(data_2d)  # Shape: (365, 24)
		
		# Calculate hourly statistics across all days
		hourly_means = np.mean(hourly_data, axis=0)
		hourly_stds = np.std(hourly_data, axis=0)
		hourly_mins = np.min(hourly_data, axis=0)
		hourly_maxs = np.max(hourly_data, axis=0)
		hourly_medians = np.median(hourly_data, axis=0)
		hourly_q25 = np.percentile(hourly_data, 25, axis=0)
		hourly_q75 = np.percentile(hourly_data, 75, axis=0)
		
		# Find daily peaks and valleys for each day
		daily_peaks = np.max(hourly_data, axis=1)
		daily_valleys = np.min(hourly_data, axis=1)
		daily_peak_hours = np.argmax(hourly_data, axis=1)
		daily_valley_hours = np.argmin(hourly_data, axis=1)
		
		# Calculate daily load factors
		daily_means = np.mean(hourly_data, axis=1)
		daily_load_factors = daily_means / daily_peaks
		
		# Weekly patterns (assuming data starts on Monday)
		day_of_week = np.arange(self.days_per_year) % 7
		weekly_patterns = []
		for day in range(7):
			mask = day_of_week == day
			if np.any(mask):
				weekly_patterns.append(np.mean(hourly_data[mask], axis=0))
			else:
				weekly_patterns.append(np.zeros(24))
		
		patterns = {
			'hourly_data': hourly_data,
			'hourly_means': hourly_means,
			'hourly_stds': hourly_stds,
			'hourly_mins': hourly_mins,
			'hourly_maxs': hourly_maxs,
			'hourly_medians': hourly_medians,
			'hourly_q25': hourly_q25,
			'hourly_q75': hourly_q75,
			'daily_peaks': daily_peaks,
			'daily_valleys': daily_valleys,
			'peak_hours': daily_peak_hours,
			'valley_hours': daily_valley_hours,
			'daily_load_factors': daily_load_factors,
			'weekly_patterns': np.array(weekly_patterns),
			'statistics': self._calculate_daily_statistics(hourly_means, daily_peaks, daily_valleys, daily_load_factors)
		}
		
		return patterns
	
	def extract_monthly_patterns(self, data_2d: np.ndarray) -> Dict:
		"""
		Extract monthly load patterns and statistics
		
		Args:
			data_2d: 2D array of shape (365, 1440)
			
		Returns:
			Dictionary containing monthly pattern analysis
		"""
		# Convert to daily averages
		daily_averages = np.mean(data_2d, axis=1)
		
		# Group by months (assuming 365 days, roughly 30.4 days per month)
		days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
		month_starts = np.cumsum([0] + days_per_month[:-1])
		
		monthly_means = []
		monthly_stds = []
		monthly_mins = []
		monthly_maxs = []
		monthly_medians = []
		monthly_load_factors = []
		monthly_peak_demands = []
		monthly_data = []
		
		for i, (start, length) in enumerate(zip(month_starts, days_per_month)):
			end = min(start + length, self.days_per_year)
			month_data = daily_averages[start:end]
			month_2d_data = data_2d[start:end]
			
			monthly_data.append(month_data)
			monthly_means.append(np.mean(month_data))
			monthly_stds.append(np.std(month_data))
			monthly_mins.append(np.min(month_data))
			monthly_maxs.append(np.max(month_data))
			monthly_medians.append(np.median(month_data))
			
			# Calculate monthly peak demand and load factor
			month_peak = np.max(month_2d_data)
			month_mean = np.mean(month_2d_data)
			monthly_peak_demands.append(month_peak)
			monthly_load_factors.append(month_mean / month_peak if month_peak > 0 else 0)
		
		# Seasonal analysis
		seasons = {
			'Spring': [2, 3, 4],  # Mar, Apr, May
			'Summer': [5, 6, 7],  # Jun, Jul, Aug
			'Fall': [8, 9, 10],   # Sep, Oct, Nov
			'Winter': [11, 0, 1]  # Dec, Jan, Feb
		}
		
		seasonal_averages = {}
		for season, months in seasons.items():
			seasonal_averages[season] = np.mean([monthly_means[m] for m in months])
		
		patterns = {
			'monthly_data': monthly_data,
			'monthly_means': np.array(monthly_means),
			'monthly_stds': np.array(monthly_stds),
			'monthly_mins': np.array(monthly_mins),
			'monthly_maxs': np.array(monthly_maxs),
			'monthly_medians': np.array(monthly_medians),
			'monthly_load_factors': np.array(monthly_load_factors),
			'monthly_peak_demands': np.array(monthly_peak_demands),
			'seasonal_averages': seasonal_averages,
			'statistics': self._calculate_monthly_statistics(monthly_means, monthly_peak_demands, seasonal_averages)
		}
		
		return patterns
	
	def calculate_statistical_metrics(self, data: np.ndarray) -> Dict:
		"""
		Calculate comprehensive statistical metrics for the load data
		
		Args:
			data: 1D array of load data
			
		Returns:
			Dictionary containing statistical metrics
		"""
		# Basic statistics
		mean_val = np.mean(data)
		std_val = np.std(data)
		min_val = np.min(data)
		max_val = np.max(data)
		median_val = np.median(data)
		
		# Percentiles
		percentiles = [5, 10, 25, 75, 90, 95]
		perc_values = np.percentile(data, percentiles)
		
		# Advanced statistics
		skewness = stats.skew(data)
		kurtosis = stats.kurtosis(data)
		
		# Load-specific metrics
		load_factor = mean_val / max_val if max_val > 0 else 0
		peak_valley_ratio = max_val / min_val if min_val > 0 else np.inf
		coefficient_of_variation = std_val / mean_val if mean_val > 0 else 0
		
		# Distribution analysis
		range_val = max_val - min_val
		iqr = perc_values[3] - perc_values[2]  # 75th - 25th percentile
		
		metrics = {
			'mean': mean_val,
			'std': std_val,
			'min': min_val,
			'max': max_val,
			'median': median_val,
			'range': range_val,
			'iqr': iqr,
			'load_factor': load_factor,
			'peak_valley_ratio': peak_valley_ratio,
			'coefficient_of_variation': coefficient_of_variation,
			'skewness': skewness,
			'kurtosis': kurtosis,
			'percentiles': {
				f'p{p}': v for p, v in zip(percentiles, perc_values)
			}
		}
		
		return metrics
	
	def _convert_to_hourly(self, data_2d: np.ndarray) -> np.ndarray:
		"""
		Convert minute-level data to hourly averages
		
		Args:
			data_2d: 2D array of shape (365, 1440)
			
		Returns:
			2D array of shape (365, 24) with hourly averages
		"""
		# Reshape to (365, 24, 60) and take mean over minutes
		reshaped = data_2d.reshape(self.days_per_year, self.hours_per_day, self.minutes_per_hour)
		return np.mean(reshaped, axis=2)
	
	def _calculate_daily_statistics(self, hourly_means: np.ndarray, 
								   daily_peaks: np.ndarray,
								   daily_valleys: np.ndarray,
								   daily_load_factors: np.ndarray) -> Dict:
		"""Calculate summary statistics for daily patterns"""
		
		peak_hour = np.argmax(hourly_means)
		valley_hour = np.argmin(hourly_means)
		
		stats = {
			'peak_hour': int(peak_hour),
			'valley_hour': int(valley_hour),
			'peak_load': float(hourly_means[peak_hour]),
			'valley_load': float(hourly_means[valley_hour]),
			'daily_load_swing': float(hourly_means[peak_hour] - hourly_means[valley_hour]),
			'load_factor': float(np.mean(daily_load_factors)),
			'peak_load_std': float(np.std(daily_peaks)),
			'valley_load_std': float(np.std(daily_valleys)),
			'most_variable_hour': int(np.argmax(np.std(hourly_means))),
			'least_variable_hour': int(np.argmin(np.std(hourly_means)))
		}
		
		return stats
	
	def _calculate_monthly_statistics(self, monthly_means: np.ndarray,
									 monthly_peaks: np.ndarray,
									 seasonal_averages: Dict) -> Dict:
		"""Calculate summary statistics for monthly patterns"""
		
		months = ['January', 'February', 'March', 'April', 'May', 'June',
				  'July', 'August', 'September', 'October', 'November', 'December']
		
		peak_month_idx = np.argmax(monthly_means)
		valley_month_idx = np.argmin(monthly_means)
		
		stats = {
			'peak_month': months[peak_month_idx],
			'valley_month': months[valley_month_idx],
			'peak_month_load': float(monthly_means[peak_month_idx]),
			'valley_month_load': float(monthly_means[valley_month_idx]),
			'seasonal_variation': float(np.max(monthly_means) - np.min(monthly_means)),
			'annual_load_factor': float(np.mean(monthly_means) / np.max(monthly_peaks)),
			'most_variable_month': months[np.argmax(np.std(monthly_means))],
			'seasonal_peaks': {season: float(avg) for season, avg in seasonal_averages.items()},
			'peak_season': max(seasonal_averages.items(), key=lambda x: x[1])[0],
			'valley_season': min(seasonal_averages.items(), key=lambda x: x[1])[0]
		}
		
		return stats