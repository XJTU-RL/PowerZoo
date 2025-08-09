#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Load Interpolation Tool - 统一负荷插值工具
Comprehensive tool for interpolating hourly load data to minute-level data

Combines the functionality of both interpolate_loads.py and load_interpolation.py
into a single, robust, and efficient solution.

Author: PowerZoo Team
Date: 2025-07-30
"""

import argparse
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Optional, Dict, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedLoadInterpolator:
	"""
	统一负荷插值器
	
	Features:
	- High-quality cubic spline interpolation
	- Enhanced boundary handling for seamless year transitions
	- Comprehensive validation and quality metrics
	- Batch processing with parallel execution
	- Flexible smoothing options
	- Detailed reporting and visualization
	"""
	
	def __init__(self, 
				 input_dir: Union[str, Path] = "data/Loads",
				 output_dir: Union[str, Path] = "data/Loads/minute_level"):
		"""
		Initialize the unified interpolator
		
		Args:
			input_dir: Input data directory path
			output_dir: Output data directory path
		"""
		self.input_dir = Path(input_dir)
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		
		# Time parameters
		self.hours_per_year = 8760  # 365 * 24
		self.minutes_per_year = 525600  # 365 * 24 * 60
		self.minutes_per_hour = 60
		
		# Supported file extensions
		self.supported_extensions = {'.csv', '.CSV', '.txt', '.dat', '.npy'}
		
		logger.info(f"Unified Load Interpolator initialized")
		logger.info(f"Input directory: {self.input_dir}")
		logger.info(f"Output directory: {self.output_dir}")
	
	def load_hourly_data(self, filepath: Path) -> np.ndarray:
		"""
		Load hourly load data from various formats
		
		Args:
			filepath: Path to the data file
			
		Returns:
			numpy array containing hourly load data
		"""
		try:
			ext = filepath.suffix.lower()
			
			if ext in ['.csv', '.CSV']:
				data = pd.read_csv(filepath, header=None)
				# Use first column as load data
				load_data = data.iloc[:, 0].values
			elif ext in ['.txt', '.dat']:
				load_data = np.loadtxt(filepath)
			elif ext == '.npy':
				load_data = np.load(filepath)
			else:
				raise ValueError(f"Unsupported file format: {ext}")
			
			# Validate data length
			if len(load_data) != self.hours_per_year:
				logger.warning(f"Data length {len(load_data)} != expected {self.hours_per_year}")
			
			return load_data.astype(np.float64)
			
		except Exception as e:
			logger.error(f"Error loading {filepath}: {e}")
			raise
	
	def create_time_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Create time arrays for interpolation
		
		Returns:
			Tuple of original hour time points and target minute time points
		"""
		# Original hour time points (0, 1, 2, ..., 8759)
		hour_times = np.arange(self.hours_per_year, dtype=np.float64)
		
		# Target minute time points (0, 1/60, 2/60, ..., 525599/60)
		minute_times = np.arange(self.minutes_per_year, dtype=np.float64) / self.minutes_per_hour
		
		return hour_times, minute_times
	
	def enhance_boundary_handling(self, hourly_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Enhanced boundary handling for seamless year transitions
		
		Args:
			hourly_data: Original hourly load data
			
		Returns:
			Extended time points and data points for better interpolation
		"""
		# Add 24 hours of data at both ends for better boundary conditions
		padding_hours = 24
		
		# End padding: use beginning of year data
		start_padding = hourly_data[:padding_hours]
		# Start padding: use end of year data
		end_padding = hourly_data[-padding_hours:]
		
		# Create extended data
		extended_data = np.concatenate([end_padding, hourly_data, start_padding])
		
		# Create extended time points
		extended_times = np.arange(-padding_hours, self.hours_per_year + padding_hours, dtype=np.float64)
		
		return extended_times, extended_data
	
	def apply_smoothing(self, data: np.ndarray, smooth_factor: float) -> np.ndarray:
		"""
		Apply smoothing to the interpolated data
		
		Args:
			data: Data to be smoothed
			smooth_factor: Smoothing strength (0.0-1.0)
			
		Returns:
			Smoothed data
		"""
		if smooth_factor <= 0:
			return data
		
		# Calculate window size based on smoothing factor
		window_size = max(1, int(smooth_factor * 20))  # Up to 20 minutes window
		
		if window_size > 1:
			# Use Gaussian-like weights for better smoothing
			weights = np.exp(-0.5 * (np.arange(window_size) - window_size//2)**2 / (window_size/3)**2)
			weights = weights / np.sum(weights)
			
			# Apply convolution with 'same' mode to preserve data length
			smoothed = np.convolve(data, weights, mode='same')
			return smoothed
		
		return data
	
	def interpolate_to_minutes(self, hourly_data: np.ndarray, 
							  smooth_factor: float = 0.0) -> np.ndarray:
		"""
		Interpolate hourly data to minute-level data
		
		Args:
			hourly_data: Hourly load data
			smooth_factor: Smoothing factor (0.0-1.0)
			
		Returns:
			Minute-level load data
		"""
		# Create time arrays
		hour_times, minute_times = self.create_time_arrays()
		
		# Enhanced boundary handling
		extended_times, extended_data = self.enhance_boundary_handling(hourly_data)
		
		try:
			# Use cubic spline interpolation with natural boundary conditions
			cs = CubicSpline(extended_times, extended_data, bc_type='natural')
			
			# Interpolate to minute level
			minute_data = cs(minute_times)
			
			# Apply smoothing if requested
			if smooth_factor > 0:
				minute_data = self.apply_smoothing(minute_data, smooth_factor)
			
			# Ensure non-negative values (load factors should be non-negative)
			minute_data = np.maximum(minute_data, 0.0)
			
			logger.info(f"Interpolation completed: {len(hourly_data)} -> {len(minute_data)} data points")
			
			return minute_data
			
		except Exception as e:
			logger.error(f"Interpolation error: {e}")
			raise
	
	def validate_interpolation(self, original: np.ndarray, interpolated: np.ndarray) -> Dict:
		"""
		Validate interpolation quality with comprehensive metrics
		
		Args:
			original: Original hourly data
			interpolated: Interpolated minute-level data
			
		Returns:
			Dictionary containing validation statistics
		"""
		# Extract hourly points from interpolated data for comparison
		hour_indices = np.arange(0, len(interpolated), self.minutes_per_hour)
		extracted_hourly = interpolated[hour_indices[:len(original)]]
		
		# Calculate error metrics
		mae = np.mean(np.abs(original - extracted_hourly))
		rmse = np.sqrt(np.mean((original - extracted_hourly) ** 2))
		max_error = np.max(np.abs(original - extracted_hourly))
		mape = np.mean(np.abs((original - extracted_hourly) / (original + 1e-8))) * 100
		
		# Calculate correlation coefficient
		correlation = np.corrcoef(original, extracted_hourly)[0, 1]
		
		# Calculate R-squared
		ss_res = np.sum((original - extracted_hourly) ** 2)
		ss_tot = np.sum((original - np.mean(original)) ** 2)
		r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
		
		validation_stats = {
			'mean_absolute_error': mae,
			'root_mean_square_error': rmse,
			'max_error': max_error,
			'mean_absolute_percentage_error': mape,
			'correlation': correlation,
			'r_squared': r_squared,
			'original_range': (np.min(original), np.max(original)),
			'interpolated_range': (np.min(interpolated), np.max(interpolated)),
			'original_mean': np.mean(original),
			'interpolated_mean': np.mean(interpolated),
			'original_std': np.std(original),
			'interpolated_std': np.std(interpolated)
		}
		
		return validation_stats
	
	def process_single_file(self, input_filename: str, 
						   output_filename: Optional[str] = None,
						   smooth_factor: float = 0.0) -> Dict:
		"""
		Process a single load file
		
		Args:
			input_filename: Input file name
			output_filename: Output file name (auto-generated if None)
			smooth_factor: Smoothing factor (0.0-1.0)
			
		Returns:
			Validation statistics dictionary
		"""
		input_path = self.input_dir / input_filename
		
		if not input_path.exists():
			raise FileNotFoundError(f"Input file not found: {input_path}")
		
		if output_filename is None:
			# Auto-generate output filename
			stem = input_path.stem
			output_filename = f"{stem}_minute_level.csv"
		
		output_path = self.output_dir / output_filename
		
		logger.info(f"Processing file: {input_path} -> {output_path}")
		
		# Load data
		hourly_data = self.load_hourly_data(input_path)
		
		# Interpolate
		minute_data = self.interpolate_to_minutes(hourly_data, smooth_factor)
		
		# Save results
		pd.DataFrame(minute_data).to_csv(output_path, header=False, index=False)
		
		# Validate
		validation_stats = self.validate_interpolation(hourly_data, minute_data)
		
		# Add file information
		validation_stats.update({
			'input_file': str(input_path),
			'output_file': str(output_path),
			'smooth_factor': smooth_factor,
			'processing_status': 'success'
		})
		
		logger.info(f"Validation - MAE: {validation_stats['mean_absolute_error']:.6f}, "
				   f"RMSE: {validation_stats['root_mean_square_error']:.6f}, "
				   f"Correlation: {validation_stats['correlation']:.6f}")
		
		return validation_stats
	
	def find_load_files(self) -> List[Path]:
		"""
		Find all supported load data files
		
		Returns:
			List of Path objects for load files
		"""
		load_files = []
		
		# Look for LoadShape*.* files with supported extensions
		for ext in self.supported_extensions:
			pattern = f"LoadShape*{ext}"
			files = list(self.input_dir.glob(pattern))
			load_files.extend(files)
		
		# Remove duplicates and sort
		load_files = sorted(list(set(load_files)))
		
		logger.info(f"Found {len(load_files)} load data files")
		return load_files
	
	def process_all_files(self, smooth_factor: float = 0.0, 
						 parallel: bool = True, max_workers: int = 4) -> Dict:
		"""
		Process all load files with optional parallel processing
		
		Args:
			smooth_factor: Smoothing factor (0.0-1.0)
			parallel: Whether to use parallel processing
			max_workers: Maximum number of parallel workers
			
		Returns:
			Dictionary containing all file processing results
		"""
		load_files = self.find_load_files()
		
		if not load_files:
			raise FileNotFoundError(f"No LoadShape*.* files found in {self.input_dir}")
		
		all_stats = {}
		
		if parallel and len(load_files) > 1:
			logger.info(f"Processing {len(load_files)} files in parallel...")
			
			# Parallel processing
			with ProcessPoolExecutor(max_workers=max_workers) as executor:
				# Submit all tasks
				future_to_file = {
					executor.submit(self._process_file_wrapper, file_path.name, smooth_factor): file_path
					for file_path in load_files
				}
				
				# Collect results
				for future in as_completed(future_to_file):
					file_path = future_to_file[future]
					try:
						stats = future.result()
						all_stats[file_path.name] = stats
					except Exception as e:
						logger.error(f"Error processing {file_path.name}: {e}")
						all_stats[file_path.name] = {
							'input_file': str(file_path),
							'processing_status': 'failed',
							'error': str(e)
						}
		else:
			logger.info(f"Processing {len(load_files)} files sequentially...")
			
			# Sequential processing
			for file_path in load_files:
				try:
					stats = self.process_single_file(file_path.name, smooth_factor=smooth_factor)
					all_stats[file_path.name] = stats
				except Exception as e:
					logger.error(f"Error processing {file_path.name}: {e}")
					all_stats[file_path.name] = {
						'input_file': str(file_path),
						'processing_status': 'failed',
						'error': str(e)
					}
		
		# Calculate overall statistics
		successful_stats = [s for s in all_stats.values() if s.get('processing_status') == 'success']
		if successful_stats:
			overall_stats = self._compute_overall_stats(successful_stats)
			all_stats['overall'] = overall_stats
		
		logger.info("All files processed!")
		return all_stats
	
	def _process_file_wrapper(self, filename: str, smooth_factor: float) -> Dict:
		"""Wrapper function for parallel processing"""
		return self.process_single_file(filename, smooth_factor=smooth_factor)
	
	def _compute_overall_stats(self, successful_stats: List[Dict]) -> Dict:
		"""
		Compute overall statistics from successful processing results
		
		Args:
			successful_stats: List of successful processing statistics
			
		Returns:
			Overall statistics dictionary
		"""
		metrics = ['mean_absolute_error', 'root_mean_square_error', 'correlation', 
				  'r_squared', 'mean_absolute_percentage_error']
		
		overall_stats = {
			'total_files': len(successful_stats),
			'smooth_factor': successful_stats[0].get('smooth_factor', 0.0) if successful_stats else 0.0
		}
		
		for metric in metrics:
			values = [stats[metric] for stats in successful_stats if metric in stats]
			if values:
				overall_stats.update({
					f'avg_{metric}': np.mean(values),
					f'min_{metric}': np.min(values),
					f'max_{metric}': np.max(values),
					f'std_{metric}': np.std(values)
				})
		
		return overall_stats
	
	def generate_comprehensive_report(self, stats: Dict, 
									 output_path: Optional[str] = None) -> None:
		"""
		Generate comprehensive interpolation report
		
		Args:
			stats: Processing statistics
			output_path: Report output path (auto-generated if None)
		"""
		if output_path is None:
			output_path = self.output_dir / "comprehensive_interpolation_report.txt"
		else:
			output_path = Path(output_path)
		
		output_path.parent.mkdir(parents=True, exist_ok=True)
		
		successful_files = [k for k, v in stats.items() 
						   if k != 'overall' and v.get('processing_status') == 'success']
		failed_files = [k for k, v in stats.items() 
					   if k != 'overall' and v.get('processing_status') == 'failed']
		
		with open(output_path, 'w', encoding='utf-8') as f:
			f.write("COMPREHENSIVE LOAD DATA INTERPOLATION REPORT\n")
			f.write("=" * 60 + "\n\n")
			f.write(f"Processing Time: {pd.Timestamp.now()}\n")
			f.write(f"Interpolation Method: Cubic Spline with Enhanced Boundary Handling\n")
			f.write(f"Data Transformation: Hourly (8760 points) -> Minute-level (525600 points)\n\n")
			
			# Processing Summary
			f.write("PROCESSING SUMMARY\n")
			f.write("-" * 30 + "\n")
			f.write(f"Total Files Processed: {len(successful_files) + len(failed_files)}\n")
			f.write(f"Successful: {len(successful_files)}\n")
			f.write(f"Failed: {len(failed_files)}\n")
			f.write(f"Success Rate: {len(successful_files)/(len(successful_files)+len(failed_files))*100:.1f}%\n\n")
			
			# Failed Files
			if failed_files:
				f.write("FAILED FILES\n")
				f.write("-" * 30 + "\n")
				for filename in failed_files:
					error = stats[filename].get('error', 'Unknown error')
					f.write(f"File: {filename}\n")
					f.write(f"Error: {error}\n\n")
			
			# Individual File Results
			if successful_files:
				f.write("INDIVIDUAL FILE VALIDATION RESULTS\n")
				f.write("-" * 40 + "\n")
				
				for filename in successful_files:
					file_stats = stats[filename]
					f.write(f"\nFile: {filename}\n")
					f.write(f"  Mean Absolute Error (MAE): {file_stats['mean_absolute_error']:.8f}\n")
					f.write(f"  Root Mean Square Error (RMSE): {file_stats['root_mean_square_error']:.8f}\n")
					f.write(f"  Maximum Error: {file_stats['max_error']:.8f}\n")
					f.write(f"  Mean Absolute Percentage Error (MAPE): {file_stats['mean_absolute_percentage_error']:.4f}%\n")
					f.write(f"  Correlation Coefficient: {file_stats['correlation']:.8f}\n")
					f.write(f"  R-squared: {file_stats['r_squared']:.8f}\n")
					f.write(f"  Original Data Range: {file_stats['original_range']}\n")
					f.write(f"  Interpolated Data Range: {file_stats['interpolated_range']}\n")
					f.write(f"  Smooth Factor: {file_stats['smooth_factor']}\n")
			
			# Overall Statistics
			if 'overall' in stats:
				overall = stats['overall']
				f.write(f"\nOVERALL STATISTICS\n")
				f.write("-" * 30 + "\n")
				f.write(f"Total Successfully Processed Files: {overall['total_files']}\n")
				f.write(f"Average MAE: {overall['avg_mean_absolute_error']:.8f}\n")
				f.write(f"Average RMSE: {overall['avg_root_mean_square_error']:.8f}\n")
				f.write(f"Average Correlation: {overall['avg_correlation']:.8f}\n")
				f.write(f"Average R-squared: {overall['avg_r_squared']:.8f}\n")
				f.write(f"Average MAPE: {overall['avg_mean_absolute_percentage_error']:.4f}%\n")
				f.write(f"MAE Range: {overall['min_mean_absolute_error']:.8f} - {overall['max_mean_absolute_error']:.8f}\n")
				f.write(f"Correlation Range: {overall['min_correlation']:.8f} - {overall['max_correlation']:.8f}\n")
			
			# Quality Assessment
			f.write("\nQUALITY ASSESSMENT\n")
			f.write("-" * 30 + "\n")
			if 'overall' in stats:
				avg_corr = stats['overall']['avg_correlation']
				avg_mae = stats['overall']['avg_mean_absolute_error']
				avg_r2 = stats['overall']['avg_r_squared']
				
				if avg_corr > 0.99 and avg_mae < 0.01 and avg_r2 > 0.98:
					f.write("✓ Interpolation Quality: EXCELLENT\n")
					f.write("  - Very high correlation and low error rates\n")
					f.write("  - Suitable for high-precision applications\n")
				elif avg_corr > 0.95 and avg_mae < 0.02 and avg_r2 > 0.90:
					f.write("✓ Interpolation Quality: GOOD\n")
					f.write("  - High correlation with acceptable error rates\n")
					f.write("  - Suitable for most practical applications\n")
				elif avg_corr > 0.90 and avg_mae < 0.05 and avg_r2 > 0.80:
					f.write("! Interpolation Quality: FAIR\n")
					f.write("  - Moderate correlation with noticeable errors\n")
					f.write("  - May require additional validation\n")
				else:
					f.write("✗ Interpolation Quality: NEEDS IMPROVEMENT\n")
					f.write("  - Low correlation or high error rates detected\n")
					f.write("  - Consider data preprocessing or different methods\n")
			
			f.write("\nTECHNICAL NOTES\n")
			f.write("-" * 30 + "\n")
			f.write("1. Cubic spline interpolation preserves original load trends\n")
			f.write("2. Enhanced boundary handling ensures smooth year transitions\n")
			f.write("3. All negative values corrected to 0 (load factors must be non-negative)\n")
			f.write("4. Optional smoothing applied to reduce high-frequency noise\n")
			f.write("5. Comprehensive validation metrics provided for quality assessment\n")
			f.write("6. Results validated against original hourly checkpoints\n")
		
		logger.info(f"Comprehensive report saved to: {output_path}")
	
	def plot_comparison(self, input_filename: str, sample_hours: int = 168) -> None:
		"""
		Plot interpolation comparison chart
		
		Args:
			input_filename: Input file name to visualize
			sample_hours: Number of hours to display (default: 168 = 1 week)
		"""
		try:
			import matplotlib.pyplot as plt
			
			# Load original data
			input_path = self.input_dir / input_filename
			original_data = self.load_hourly_data(input_path)
			
			# Interpolate
			minute_data = self.interpolate_to_minutes(original_data)
			
			# Select sample data for visualization
			sample_minutes = sample_hours * 60
			sample_original = original_data[:sample_hours]
			sample_minute = minute_data[:sample_minutes]
			
			# Create time axes
			hour_times = np.arange(sample_hours)
			minute_times = np.arange(sample_minutes) / 60
			
			# Create plot
			plt.figure(figsize=(15, 8))
			
			# Plot original hourly data
			plt.plot(hour_times, sample_original, 'ro-', markersize=4, linewidth=2,
					label='Original Hourly Data', alpha=0.8, zorder=3)
			
			# Plot interpolated minute data
			plt.plot(minute_times, sample_minute, 'b-', linewidth=1,
					label='Interpolated Minute Data', alpha=0.7, zorder=2)
			
			plt.xlabel('Time (Hours)')
			plt.ylabel('Load Factor')
			plt.title(f'Load Data Interpolation Comparison - {input_filename}\n'
					 f'Sample Period: First {sample_hours} Hours')
			plt.legend()
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			
			# Save plot
			plot_path = self.output_dir / f"interpolation_comparison_{input_path.stem}.png"
			plt.savefig(plot_path, dpi=300, bbox_inches='tight')
			logger.info(f"Comparison chart saved to: {plot_path}")
			
			# Show plot if in interactive environment
			try:
				plt.show()
			except:
				pass  # Ignore in non-interactive environments
				
		except ImportError:
			logger.error("Plotting requires matplotlib: pip install matplotlib")
		except Exception as e:
			logger.error(f"Plotting error: {e}")


def main():
	"""Main function for command line interface"""
	parser = argparse.ArgumentParser(
		description="Unified Load Data Interpolation Tool",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Process all files without smoothing
  python unified_load_interpolator.py
  
  # Process all files with light smoothing
  python unified_load_interpolator.py --smooth 0.1
  
  # Process single file
  python unified_load_interpolator.py --single LoadShape1.CSV
  
  # Generate comparison chart
  python unified_load_interpolator.py --plot
  
  # Use custom directories
  python unified_load_interpolator.py --input-dir custom/input --output-dir custom/output
		"""
	)
	
	parser.add_argument('--smooth', type=float, default=0.0, metavar='FACTOR',
					   help='Smoothing strength [0.0-1.0], 0 for no smoothing (default: 0.0)')
	parser.add_argument('--input-dir', type=str, default='data/Loads', metavar='DIR',
					   help='Input data directory (default: data/Loads)')
	parser.add_argument('--output-dir', type=str, default='data/Loads/minute_level', metavar='DIR',
					   help='Output data directory (default: data/Loads/minute_level)')
	parser.add_argument('--single', type=str, metavar='FILE',
					   help='Process only the specified single file')
	parser.add_argument('--plot', action='store_true',
					   help='Generate interpolation comparison chart')
	parser.add_argument('--no-parallel', action='store_true',
					   help='Disable parallel processing')
	parser.add_argument('--workers', type=int, default=4,
					   help='Number of parallel workers (default: 4)')
	
	args = parser.parse_args()
	
	# Validate parameters
	if args.smooth < 0.0 or args.smooth > 1.0:
		logger.error("Smoothing factor must be in range 0.0-1.0")
		return 1
	
	try:
		# Create interpolator
		interpolator = UnifiedLoadInterpolator(
			input_dir=args.input_dir,
			output_dir=args.output_dir
		)
		
		logger.info("Starting load data interpolation processing...")
		logger.info(f"Input directory: {interpolator.input_dir}")
		logger.info(f"Output directory: {interpolator.output_dir}")
		if args.smooth > 0:
			logger.info(f"Smoothing factor: {args.smooth}")
		
		# Process data
		if args.single:
			# Process single file
			logger.info(f"Processing single file: {args.single}")
			stats = {args.single: interpolator.process_single_file(args.single, smooth_factor=args.smooth)}
		else:
			# Process all files
			logger.info("Processing all LoadShape*.* files...")
			stats = interpolator.process_all_files(
				smooth_factor=args.smooth,
				parallel=not args.no_parallel,
				max_workers=args.workers
			)
		
		# Generate comprehensive report
		interpolator.generate_comprehensive_report(stats)
		
		# Plot comparison if requested
		if args.plot:
			if args.single:
				interpolator.plot_comparison(args.single)
			else:
				# Default to first available file
				load_files = interpolator.find_load_files()
				if load_files:
					interpolator.plot_comparison(load_files[0].name)
		
		logger.info("✓ Interpolation processing completed!")
		
		# Output summary statistics
		successful_files = [k for k, v in stats.items() 
						   if k != 'overall' and v.get('processing_status') == 'success']
		
		if 'overall' in stats and successful_files:
			overall = stats['overall']
			print(f"\nInterpolation Quality Summary:")
			print(f"  Files Processed: {len(successful_files)}")
			print(f"  Average MAE: {overall['avg_mean_absolute_error']:.6f}")
			print(f"  Average Correlation: {overall['avg_correlation']:.6f}")
			print(f"  Average R-squared: {overall['avg_r_squared']:.6f}")
			
			avg_corr = overall['avg_correlation']
			if avg_corr > 0.99:
				print("  Quality Assessment: ✓ EXCELLENT")
			elif avg_corr > 0.95:
				print("  Quality Assessment: ✓ GOOD")
			elif avg_corr > 0.90:
				print("  Quality Assessment: ! FAIR")
			else:
				print("  Quality Assessment: ✗ NEEDS IMPROVEMENT")
		
		print(f"\nOutput files location: {interpolator.output_dir}")
		print(f"Detailed report: {interpolator.output_dir}/comprehensive_interpolation_report.txt")
		
		return 0
		
	except FileNotFoundError as e:
		logger.error(f"File not found: {e}")
		return 1
	except Exception as e:
		logger.error(f"Processing error: {e}")
		import traceback
		traceback.print_exc()
		return 1


if __name__ == "__main__":
	sys.exit(main())