#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Utilities for Temporal Trend Analysis
时间趋势分析可视化工具

Author: PowerZoo Team
Date: 2025-07-30
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class TrendVisualizer:
	"""
	Professional visualization utilities for temporal trend analysis
	"""
	
	def __init__(self, output_dir: str = "results"):
		"""
		Initialize the visualizer
		
		Args:
			output_dir: Output directory for saving plots
		"""
		self.output_dir = Path(output_dir)
		self.daily_dir = self.output_dir / "daily_analysis"
		self.monthly_dir = self.output_dir / "monthly_analysis"
		
		# Create directories
		self.daily_dir.mkdir(parents=True, exist_ok=True)
		self.monthly_dir.mkdir(parents=True, exist_ok=True)
		
		# Color schemes
		self.colors = {
			'primary': '#2E86C1',
			'secondary': '#E74C3C',
			'accent': '#F39C12',
			'success': '#27AE60',
			'warning': '#F1C40F',
			'info': '#8E44AD',
			'light': '#BDC3C7',
			'dark': '#34495E'
		}
	
	def plot_daily_trends(self, daily_patterns: Dict) -> None:
		"""
		Generate comprehensive daily trend visualizations
		
		Args:
			daily_patterns: Dictionary containing daily pattern data
		"""
		# 1. Daily load profile
		self._plot_daily_load_profile(daily_patterns)
		
		# 2. Daily load heatmap
		self._plot_daily_heatmap(daily_patterns)
		
		# 3. Daily box plot
		self._plot_daily_box_plot(daily_patterns)
		
		# 4. Polar plot
		self._plot_daily_polar(daily_patterns)
	
	def plot_monthly_trends(self, monthly_patterns: Dict) -> None:
		"""
		Generate comprehensive monthly trend visualizations
		
		Args:
			monthly_patterns: Dictionary containing monthly pattern data
		"""
		# 1. Monthly trends
		self._plot_monthly_trends(monthly_patterns)
		
		# 2. Seasonal patterns
		self._plot_seasonal_patterns(monthly_patterns)
		
		# 3. Monthly statistics
		self._plot_monthly_statistics(monthly_patterns)
		
		# 4. Monthly box plot
		self._plot_monthly_box_plot(monthly_patterns)
	
	def plot_combined_analysis(self, daily_patterns: Dict, monthly_patterns: Dict, overall_stats: Dict) -> None:
		"""
		Generate combined analysis visualization
		
		Args:
			daily_patterns: Daily pattern data
			monthly_patterns: Monthly pattern data
			overall_stats: Overall statistical data
		"""
		self._plot_comprehensive_dashboard(daily_patterns, monthly_patterns, overall_stats)
	
	def _plot_daily_load_profile(self, patterns: Dict) -> None:
		"""Plot average daily load profile with confidence intervals"""
		fig, ax = plt.subplots(figsize=(12, 6))
		
		hours = np.arange(24)
		means = patterns['hourly_means']
		stds = patterns['hourly_stds']
		q25 = patterns['hourly_q25']
		q75 = patterns['hourly_q75']
		
		# Main line
		ax.plot(hours, means, linewidth=3, color=self.colors['primary'], 
				label='Average Load', zorder=3)
		
		# Confidence interval
		ax.fill_between(hours, means - stds, means + stds, 
						alpha=0.3, color=self.colors['primary'], 
						label='±1 Std Dev', zorder=1)
		
		# Quartile range
		ax.fill_between(hours, q25, q75, alpha=0.2, color=self.colors['secondary'],
						label='IQR (25th-75th)', zorder=2)
		
		# Peak and valley markers
		peak_hour = np.argmax(means)
		valley_hour = np.argmin(means)
		
		ax.scatter([peak_hour], [means[peak_hour]], color=self.colors['warning'], 
				  s=100, marker='^', label=f'Peak ({peak_hour}:00)', zorder=4)
		ax.scatter([valley_hour], [means[valley_hour]], color=self.colors['info'], 
				  s=100, marker='v', label=f'Valley ({valley_hour}:00)', zorder=4)
		
		ax.set_xlabel('Hour of Day')
		ax.set_ylabel('Load (MW)')
		ax.set_title('Daily Load Profile - Average Pattern with Variability')
		ax.set_xticks(range(0, 24, 2))
		ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(self.daily_dir / "daily_load_profile.png")
		plt.close()
	
	def _plot_daily_heatmap(self, patterns: Dict) -> None:
		"""Plot daily load patterns as heatmap"""
		fig, ax = plt.subplots(figsize=(14, 8))
		
		# Reshape hourly data for heatmap (days x hours)
		hourly_data = patterns['hourly_data']
		
		# Create month labels for y-axis
		days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
		month_starts = np.cumsum([0] + days_per_month[:-1])
		month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
					   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		
		# Sample data for visualization (every 7 days to avoid overcrowding)
		sample_indices = np.arange(0, len(hourly_data), 7)
		sampled_data = hourly_data[sample_indices]
		
		# Create heatmap
		im = ax.imshow(sampled_data, cmap='YlOrRd', aspect='auto', interpolation='bilinear')
		
		# Colorbar
		cbar = plt.colorbar(im, ax=ax)
		cbar.set_label('Load (MW)', rotation=270, labelpad=20)
		
		# Labels
		ax.set_xlabel('Hour of Day')
		ax.set_ylabel('Day of Year (sampled)')
		ax.set_title('Daily Load Patterns Heatmap - Hourly Load Throughout Year')
		
		# Ticks
		ax.set_xticks(range(0, 24, 2))
		ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
		
		# Y-axis with month indicators
		y_ticks = []
		y_labels = []
		for i, (start, name) in enumerate(zip(month_starts, month_names)):
			idx = np.searchsorted(sample_indices, start)
			if idx < len(sample_indices):
				y_ticks.append(idx)
				y_labels.append(name)
		
		ax.set_yticks(y_ticks)
		ax.set_yticklabels(y_labels)
		
		plt.tight_layout()
		plt.savefig(self.daily_dir / "daily_load_heatmap.png")
		plt.close()
	
	def _plot_daily_box_plot(self, patterns: Dict) -> None:
		"""Plot hourly load distribution box plots"""
		fig, ax = plt.subplots(figsize=(15, 8))
		
		hourly_data = patterns['hourly_data']
		
		# Create box plot
		box_data = [hourly_data[:, hour] for hour in range(24)]
		bp = ax.boxplot(box_data, patch_artist=True, notch=True)
		
		# Color the boxes
		colors = plt.cm.viridis(np.linspace(0, 1, 24))
		for patch, color in zip(bp['boxes'], colors):
			patch.set_facecolor(color)
			patch.set_alpha(0.7)
		
		# Styling
		ax.set_xlabel('Hour of Day')
		ax.set_ylabel('Load (MW)')
		ax.set_title('Hourly Load Distribution - Box Plots Showing Variability by Hour')
		ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(self.daily_dir / "daily_box_plot.png")
		plt.close()
	
	def _plot_daily_polar(self, patterns: Dict) -> None:
		"""Plot daily pattern in polar coordinates"""
		fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
		
		# Convert hours to radians
		hours = np.arange(24)
		theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
		
		means = patterns['hourly_means']
		stds = patterns['hourly_stds']
		
		# Main plot
		ax.plot(theta, means, linewidth=3, color=self.colors['primary'], 
				marker='o', markersize=6, label='Average Load')
		
		# Fill area
		ax.fill(theta, means, alpha=0.3, color=self.colors['primary'])
		
		# Error bars
		ax.errorbar(theta, means, yerr=stds, fmt='none', 
					ecolor=self.colors['secondary'], alpha=0.5, capsize=3)
		
		# Styling
		ax.set_theta_zero_location('N')  # 12 o'clock at top
		ax.set_theta_direction(-1)  # Clockwise
		ax.set_thetagrids(np.arange(0, 360, 15), 
						 [f'{h:02d}:00' for h in range(0, 24, 1)])
		ax.set_title('Daily Load Pattern - Polar Plot\n(12:00 AM at top, clockwise)', 
					pad=20)
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(self.daily_dir / "daily_polar_plot.png")
		plt.close()
	
	def _plot_monthly_trends(self, patterns: Dict) -> None:
		"""Plot monthly trend analysis"""
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
		
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
				  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		x = np.arange(12)
		
		# Monthly averages
		monthly_means = patterns['monthly_means']
		monthly_stds = patterns['monthly_stds']
		
		ax1.bar(x, monthly_means, yerr=monthly_stds, capsize=5,
				color=self.colors['primary'], alpha=0.7, 
				error_kw={'ecolor': self.colors['dark'], 'alpha': 0.8})
		
		ax1.set_ylabel('Average Load (MW)')
		ax1.set_title('Monthly Average Load with Standard Deviation')
		ax1.set_xticks(x)
		ax1.set_xticklabels(months)
		ax1.grid(True, alpha=0.3)
		
		# Monthly load factors
		load_factors = patterns['monthly_load_factors']
		
		ax2.plot(x, load_factors, marker='o', linewidth=2, markersize=8,
				color=self.colors['accent'])
		ax2.set_ylabel('Load Factor')
		ax2.set_xlabel('Month')
		ax2.set_title('Monthly Load Factor Trends')
		ax2.set_xticks(x)
		ax2.set_xticklabels(months)
		ax2.grid(True, alpha=0.3)
		ax2.set_ylim(0, 1)
		
		plt.tight_layout()
		plt.savefig(self.monthly_dir / "monthly_trends.png")
		plt.close()
	
	def _plot_seasonal_patterns(self, patterns: Dict) -> None:
		"""Plot seasonal analysis"""
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
		
		seasonal_averages = patterns['seasonal_averages']
		seasons = list(seasonal_averages.keys())
		values = list(seasonal_averages.values())
		
		# Seasonal bar chart
		colors_seasonal = [self.colors['success'], self.colors['warning'],
						  self.colors['accent'], self.colors['info']]
		bars = ax1.bar(seasons, values, color=colors_seasonal, alpha=0.7)
		ax1.set_ylabel('Average Load (MW)')
		ax1.set_title('Seasonal Average Load Comparison')
		ax1.grid(True, alpha=0.3)
		
		# Add value labels on bars
		for bar, value in zip(bars, values):
			height = bar.get_height()
			ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
					f'{value:.1f}', ha='center', va='bottom')
		
		# Monthly peaks
		monthly_peaks = patterns['monthly_peak_demands']
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
				  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		
		ax2.plot(range(12), monthly_peaks, marker='s', linewidth=2, markersize=6,
				color=self.colors['secondary'])
		ax2.set_ylabel('Peak Demand (MW)')
		ax2.set_xlabel('Month')
		ax2.set_title('Monthly Peak Demand Trends')
		ax2.set_xticks(range(12))
		ax2.set_xticklabels(months, rotation=45)
		ax2.grid(True, alpha=0.3)
		
		# Seasonal polar plot
		theta = np.linspace(0, 2*np.pi, 4, endpoint=False)
		ax3 = plt.subplot(2, 2, 3, projection='polar')
		ax3.bar(theta, values, width=2*np.pi/4, alpha=0.7, color=colors_seasonal)
		ax3.set_thetagrids([0, 90, 180, 270], seasons)
		ax3.set_title('Seasonal Load Distribution\n(Polar View)')
		
		# Monthly coefficient of variation
		monthly_cv = patterns['monthly_stds'] / patterns['monthly_means']
		ax4.bar(range(12), monthly_cv, color=self.colors['info'], alpha=0.7)
		ax4.set_ylabel('Coefficient of Variation')
		ax4.set_xlabel('Month')
		ax4.set_title('Monthly Load Variability (CV)')
		ax4.set_xticks(range(12))
		ax4.set_xticklabels(months, rotation=45)
		ax4.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(self.monthly_dir / "seasonal_patterns.png")
		plt.close()
	
	def _plot_monthly_statistics(self, patterns: Dict) -> None:
		"""Plot comprehensive monthly statistics"""
		fig, axes = plt.subplots(2, 2, figsize=(15, 12))
		axes = axes.flatten()
		
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
				  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		x = np.arange(12)
		
		# 1. Monthly means vs medians
		ax = axes[0]
		ax.plot(x, patterns['monthly_means'], marker='o', label='Mean', linewidth=2)
		ax.plot(x, patterns['monthly_medians'], marker='s', label='Median', linewidth=2)
		ax.set_ylabel('Load (MW)')
		ax.set_title('Monthly Mean vs Median Load')
		ax.set_xticks(x)
		ax.set_xticklabels(months, rotation=45)
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# 2. Monthly range (max - min)
		ax = axes[1]
		monthly_range = patterns['monthly_maxs'] - patterns['monthly_mins']
		ax.bar(x, monthly_range, color=self.colors['warning'], alpha=0.7)
		ax.set_ylabel('Load Range (MW)')
		ax.set_title('Monthly Load Range (Max - Min)')
		ax.set_xticks(x)
		ax.set_xticklabels(months, rotation=45)
		ax.grid(True, alpha=0.3)
		
		# 3. Load factor comparison
		ax = axes[2]
		load_factors = patterns['monthly_load_factors']
		colors = ['red' if lf < 0.5 else 'yellow' if lf < 0.7 else 'green' for lf in load_factors]
		bars = ax.bar(x, load_factors, color=colors, alpha=0.7)
		ax.set_ylabel('Load Factor')
		ax.set_title('Monthly Load Factor (Color: Red<0.5, Yellow<0.7, Green≥0.7)')
		ax.set_xticks(x)
		ax.set_xticklabels(months, rotation=45)
		ax.grid(True, alpha=0.3)
		ax.set_ylim(0, 1)
		
		# Add value labels
		for bar, lf in zip(bars, load_factors):
			height = bar.get_height()
			ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
					f'{lf:.3f}', ha='center', va='bottom', fontsize=8)
		
		# 4. Seasonal comparison radar chart
		ax = axes[3]
		ax.axis('off')  # Turn off the regular axis
		
		# Create a simple seasonal comparison table
		seasonal_data = patterns['seasonal_averages']
		table_data = [[season, f"{load:.1f} MW"] for season, load in seasonal_data.items()]
		table = ax.table(cellText=table_data, colLabels=['Season', 'Average Load'],
						cellLoc='center', loc='center')
		table.auto_set_font_size(False)
		table.set_fontsize(12)
		table.scale(1, 2)
		ax.set_title('Seasonal Load Summary', pad=20)
		
		plt.tight_layout()
		plt.savefig(self.monthly_dir / "monthly_statistics.png")
		plt.close()
	
	def _plot_monthly_box_plot(self, patterns: Dict) -> None:
		"""Plot monthly box plots"""
		fig, ax = plt.subplots(figsize=(14, 8))
		
		monthly_data = patterns['monthly_data']
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
				  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		
		# Create box plot
		bp = ax.boxplot(monthly_data, patch_artist=True, notch=True)
		
		# Color the boxes with seasonal colors
		seasonal_colors = []
		for i in range(12):
			if i in [2, 3, 4]:  # Spring
				seasonal_colors.append('#27AE60')
			elif i in [5, 6, 7]:  # Summer
				seasonal_colors.append('#F39C12')
			elif i in [8, 9, 10]:  # Fall
				seasonal_colors.append('#E74C3C')
			else:  # Winter
				seasonal_colors.append('#3498DB')
		
		for patch, color in zip(bp['boxes'], seasonal_colors):
			patch.set_facecolor(color)
			patch.set_alpha(0.7)
		
		ax.set_xlabel('Month')
		ax.set_ylabel('Daily Average Load (MW)')
		ax.set_title('Monthly Load Distribution - Daily Averages by Month')
		ax.set_xticklabels(months)
		ax.grid(True, alpha=0.3)
		
		# Add legend for seasons
		from matplotlib.patches import Patch
		legend_elements = [
			Patch(facecolor='#27AE60', alpha=0.7, label='Spring'),
			Patch(facecolor='#F39C12', alpha=0.7, label='Summer'),
			Patch(facecolor='#E74C3C', alpha=0.7, label='Fall'),
			Patch(facecolor='#3498DB', alpha=0.7, label='Winter')
		]
		ax.legend(handles=legend_elements, loc='upper right')
		
		plt.tight_layout()
		plt.savefig(self.monthly_dir / "monthly_box_plot.png")
		plt.close()
	
	def _plot_comprehensive_dashboard(self, daily_patterns: Dict, monthly_patterns: Dict, overall_stats: Dict) -> None:
		"""Create comprehensive analysis dashboard"""
		fig = plt.figure(figsize=(20, 12))
		
		# Create a complex grid layout
		gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
		
		# 1. Daily profile (top left, spans 2 columns)
		ax1 = fig.add_subplot(gs[0, :2])
		hours = np.arange(24)
		means = daily_patterns['hourly_means']
		ax1.plot(hours, means, linewidth=3, color=self.colors['primary'])
		ax1.fill_between(hours, means - daily_patterns['hourly_stds'], 
						means + daily_patterns['hourly_stds'], alpha=0.3)
		ax1.set_title('Daily Load Profile')
		ax1.set_xlabel('Hour')
		ax1.set_ylabel('Load (MW)')
		ax1.grid(True, alpha=0.3)
		
		# 2. Monthly trends (top right, spans 2 columns)
		ax2 = fig.add_subplot(gs[0, 2:])
		months = range(12)
		monthly_means = monthly_patterns['monthly_means']
		ax2.bar(months, monthly_means, color=self.colors['accent'], alpha=0.7)
		ax2.set_title('Monthly Average Load')
		ax2.set_xlabel('Month')
		ax2.set_ylabel('Load (MW)')
		month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
		ax2.set_xticks(months)
		ax2.set_xticklabels(month_names)
		ax2.grid(True, alpha=0.3)
		
		# 3. Load duration curve (middle left)
		ax3 = fig.add_subplot(gs[1, 0])
		# Create synthetic load duration curve from overall stats
		load_data = np.random.normal(overall_stats['mean'], overall_stats['std'], 8760)
		load_data = np.maximum(load_data, 0)  # Ensure non-negative
		sorted_loads = np.sort(load_data)[::-1]
		duration = np.arange(len(sorted_loads)) / len(sorted_loads) * 100
		ax3.plot(duration, sorted_loads, color=self.colors['secondary'], linewidth=2)
		ax3.set_title('Load Duration Curve')
		ax3.set_xlabel('Duration (%)')
		ax3.set_ylabel('Load (MW)')
		ax3.grid(True, alpha=0.3)
		
		# 4. Statistics table (middle center-left)
		ax4 = fig.add_subplot(gs[1, 1])
		ax4.axis('off')
		stats_data = [
			['Mean', f"{overall_stats['mean']:.1f} MW"],
			['Peak', f"{overall_stats['max']:.1f} MW"],
			['Min', f"{overall_stats['min']:.1f} MW"],
			['Load Factor', f"{overall_stats['load_factor']:.3f}"],
			['CV', f"{overall_stats['coefficient_of_variation']:.3f}"],
		]
		table = ax4.table(cellText=stats_data, colLabels=['Metric', 'Value'],
						 cellLoc='center', loc='center')
		table.auto_set_font_size(False)
		table.set_fontsize(10)
		table.scale(1, 1.5)
		ax4.set_title('Key Statistics', pad=20)
		
		# 5. Seasonal polar plot (middle center-right)
		ax5 = fig.add_subplot(gs[1, 2], projection='polar')
		seasonal_data = monthly_patterns['seasonal_averages']
		theta = np.linspace(0, 2*np.pi, 4, endpoint=False)
		values = list(seasonal_data.values())
		ax5.bar(theta, values, width=2*np.pi/4, alpha=0.7)
		ax5.set_thetagrids([0, 90, 180, 270], list(seasonal_data.keys()))
		ax5.set_title('Seasonal Pattern')
		
		# 6. Distribution histogram (middle right)
		ax6 = fig.add_subplot(gs[1, 3])
		ax6.hist(load_data, bins=30, alpha=0.7, color=self.colors['info'], density=True)
		ax6.set_title('Load Distribution')
		ax6.set_xlabel('Load (MW)')
		ax6.set_ylabel('Density')
		ax6.grid(True, alpha=0.3)
		
		# 7. Weekly pattern (bottom left, spans 2 columns)
		ax7 = fig.add_subplot(gs[2, :2])
		weekly_patterns = daily_patterns['weekly_patterns']
		days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
		for i, (day, pattern) in enumerate(zip(days, weekly_patterns)):
			ax7.plot(hours, pattern, label=day, linewidth=2)
		ax7.set_title('Weekly Load Patterns')
		ax7.set_xlabel('Hour')
		ax7.set_ylabel('Load (MW)')
		ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		ax7.grid(True, alpha=0.3)
		
		# 8. Key insights (bottom right, spans 2 columns)
		ax8 = fig.add_subplot(gs[2, 2:])
		ax8.axis('off')
		
		# Generate key insights
		peak_hour = np.argmax(daily_patterns['hourly_means'])
		valley_hour = np.argmin(daily_patterns['hourly_means'])
		peak_month = np.argmax(monthly_patterns['monthly_means'])
		valley_month = np.argmin(monthly_patterns['monthly_means'])
		
		month_names_full = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
						   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		
		insights_text = f"""
KEY INSIGHTS

Daily Pattern:
• Peak demand at {peak_hour:02d}:00
• Valley demand at {valley_hour:02d}:00
• Daily load factor: {daily_patterns['statistics']['load_factor']:.3f}

Monthly Pattern:
• Peak month: {month_names_full[peak_month]}
• Valley month: {month_names_full[valley_month]}
• Seasonal variation: {monthly_patterns['statistics']['seasonal_variation']:.1f} MW

System Characteristics:
• Annual load factor: {overall_stats['load_factor']:.3f}
• Load variability (CV): {overall_stats['coefficient_of_variation']:.3f}
• Peak-to-valley ratio: {overall_stats['peak_valley_ratio']:.2f}
		"""
		
		ax8.text(0.05, 0.95, insights_text, transform=ax8.transAxes, 
				fontsize=11, verticalalignment='top', fontfamily='monospace',
				bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
		
		# Overall title
		fig.suptitle('Comprehensive Load Analysis Dashboard', fontsize=16, fontweight='bold')
		
		plt.savefig(self.output_dir / "comprehensive_analysis.png")
		plt.close()