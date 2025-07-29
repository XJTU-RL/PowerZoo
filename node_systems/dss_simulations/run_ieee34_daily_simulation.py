#!/usr/bin/env python3
"""
IEEE 34-Node Daily Simulation Script
Based on ieee34Mod1_daily.dss with comprehensive analysis and visualization

This script performs a detailed daily simulation of the IEEE 34-node test feeder
with load curves and generates extensive analysis charts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set font for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

try:
    import opendssdirect as dss
except ImportError:
    print("Error: opendssdirect package not found. Please install it using:")
    print("pip install opendssdirect")
    sys.exit(1)

class IEEE34DailySimulation:
    """
    IEEE 34-Node Daily Simulation with Load Curves
    """
    
    def __init__(self, dss_file_path):
        """
        Initialize simulation parameters
        
        Args:
            dss_file_path (str): Path to the DSS file
        """
        self.dss_file_path = dss_file_path
        self.results = {}
        self.time_points = []
        self.simulation_hours = 24
        self.step_size_hours = 1
        
        # Create results directory
        self.results_dir = os.path.join(os.path.dirname(__file__), 'daily_simulation_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"IEEE 34-Node Daily Simulation initialized")
        print(f"DSS file: {dss_file_path}")
        print(f"Results directory: {self.results_dir}")
    
    def run_simulation(self):
        """
        Run the daily OpenDSS simulation
        """
        print("\n=== Starting Daily Simulation ===")
        
        try:
            # Clear previous circuit
            dss.run_command("Clear")
            print("Previous circuit cleared")
            
            # Compile DSS file
            dss.run_command(f'Compile "{self.dss_file_path}"')
            print("DSS file compiled successfully")
            
            # Verify circuit compilation
            if not dss.Circuit.Name():
                raise Exception("Circuit compilation failed")
            
            print(f"Circuit name: {dss.Circuit.Name()}")
            
            # Initialize result storage
            self._initialize_results()
            
            # Get system information
            self._get_system_info()
            
            # Run daily simulation
            self._run_daily_simulation()
            
            print("\nDaily simulation completed successfully")
            
        except Exception as e:
            print(f"Simulation error: {str(e)}")
            raise
    
    def _initialize_results(self):
        """
        Initialize data storage for results
        """
        print("Initializing result storage...")
        
        # Time series data
        self.results = {
            'time': [],
            'hour': [],
            
            # Voltage data
            'bus_voltages': {},
            'voltage_min': [],
            'voltage_max': [],
            'voltage_avg': [],
            
            # Power data
            'total_load_kw': [],
            'total_load_kvar': [],
            'total_losses_kw': [],
            'total_losses_kvar': [],
            'source_kw': [],
            'source_kvar': [],
            
            # Individual load data
            'load_powers': {},
            
            # Generator/Battery data
            'generator_powers': {},
            
            # Capacitor data
            'capacitor_kvar': {},
            
            # Regulator data
            'regulator_taps': {},
            
            # Line data
            'line_currents': {},
            'line_powers': {},
            
            # Transformer data
            'transformer_powers': {},
            'transformer_loadings': {},
        }
    
    def _get_system_info(self):
        """
        Get basic system information
        """
        print("Getting system information...")
        
        # Get all buses
        self.all_buses = dss.Circuit.AllBusNames()
        print(f"Total buses: {len(self.all_buses)}")
        
        # Get all loads
        self.all_loads = dss.Loads.AllNames()
        print(f"Total loads: {len(self.all_loads)}")
        
        # Get all generators
        self.all_generators = dss.Generators.AllNames()
        print(f"Total generators: {len(self.all_generators)}")
        
        # Get all capacitors
        self.all_capacitors = dss.Capacitors.AllNames()
        print(f"Total capacitors: {len(self.all_capacitors)}")
        
        # Get all lines
        self.all_lines = dss.Lines.AllNames()
        print(f"Total lines: {len(self.all_lines)}")
        
        # Get all transformers
        self.all_transformers = dss.Transformers.AllNames()
        print(f"Total transformers: {len(self.all_transformers)}")
    
    def _run_daily_simulation(self):
        """
        Run the daily time-domain simulation
        """
        print("\nRunning daily time-domain simulation...")
        
        # Set simulation mode to daily
        dss.run_command("Set mode=Daily")
        dss.run_command("Set number=1")
        dss.run_command("Set hour=0")
        dss.run_command("Set stepsize=3600")
        dss.run_command("Set sec=0")
        
        # Solve initial condition
        dss.Solution.Solve()
        
        # Simulation loop for 24 hours
        for hour in range(self.simulation_hours):
            print(f"\rSimulating hour {hour+1}/24...", end='', flush=True)
            
            # Solve current time step
            dss.Solution.Solve()
            
            if not dss.Solution.Converged():
                print(f"\nWarning: Solution did not converge at hour {hour}")
            
            # Collect data
            self._collect_simulation_data(hour)
            
            # Advance time
            if hour < self.simulation_hours - 1:
                dss.run_command("Set hour=" + str(hour + 1))
        
        print("\nTime-domain simulation completed")
    
    def _collect_simulation_data(self, hour):
        """
        Collect simulation data at current time step
        """
        # Time information
        self.results['time'].append(datetime(2024, 1, 1) + timedelta(hours=hour))
        self.results['hour'].append(hour)
        
        # Collect voltage data
        self._collect_voltage_data(hour)
        
        # Collect power data
        self._collect_power_data(hour)
        
        # Collect load data
        self._collect_load_data(hour)
        
        # Collect generator data
        self._collect_generator_data(hour)
        
        # Collect capacitor data
        self._collect_capacitor_data(hour)
        
        # Collect regulator data
        self._collect_regulator_data(hour)
        
        # Collect line data
        self._collect_line_data(hour)
        
        # Collect transformer data
        self._collect_transformer_data(hour)
    
    def _collect_voltage_data(self, hour):
        """
        Collect voltage data from all buses
        """
        voltages = []
        
        for bus_name in self.all_buses:
            dss.Circuit.SetActiveBus(bus_name)
            bus_voltages = dss.Bus.puVmagAngle()
            
            # Extract voltage magnitudes (every other element)
            v_mags = [bus_voltages[i] for i in range(0, len(bus_voltages), 2)]
            
            if bus_name not in self.results['bus_voltages']:
                self.results['bus_voltages'][bus_name] = []
            
            avg_voltage = np.mean(v_mags) if v_mags else 0
            self.results['bus_voltages'][bus_name].append(avg_voltage)
            voltages.extend(v_mags)
        
        # System-wide voltage statistics
        if voltages:
            self.results['voltage_min'].append(min(voltages))
            self.results['voltage_max'].append(max(voltages))
            self.results['voltage_avg'].append(np.mean(voltages))
        else:
            self.results['voltage_min'].append(0)
            self.results['voltage_max'].append(0)
            self.results['voltage_avg'].append(0)
    
    def _collect_power_data(self, hour):
        """
        Collect system power data
        """
        # Total system power
        total_power = dss.Circuit.TotalPower()
        self.results['source_kw'].append(-total_power[0])  # Negative because it's generation
        self.results['source_kvar'].append(-total_power[1])
        
        # System losses
        losses = dss.Circuit.Losses()
        self.results['total_losses_kw'].append(losses[0] / 1000)  # Convert to kW
        self.results['total_losses_kvar'].append(losses[1] / 1000)  # Convert to kVAR
        
        # Total load power
        total_load_kw = 0
        total_load_kvar = 0
        
        for load_name in self.all_loads:
            dss.Loads.Name(load_name)
            powers = dss.CktElement.Powers()
            if len(powers) >= 2:
                total_load_kw += powers[0]
                total_load_kvar += powers[1]
        
        self.results['total_load_kw'].append(total_load_kw)
        self.results['total_load_kvar'].append(total_load_kvar)
    
    def _collect_load_data(self, hour):
        """
        Collect individual load data
        """
        for load_name in self.all_loads:
            dss.Loads.Name(load_name)
            powers = dss.CktElement.Powers()
            
            if load_name not in self.results['load_powers']:
                self.results['load_powers'][load_name] = {'kw': [], 'kvar': []}
            
            if len(powers) >= 2:
                self.results['load_powers'][load_name]['kw'].append(powers[0])
                self.results['load_powers'][load_name]['kvar'].append(powers[1])
            else:
                self.results['load_powers'][load_name]['kw'].append(0)
                self.results['load_powers'][load_name]['kvar'].append(0)
    
    def _collect_generator_data(self, hour):
        """
        Collect generator/battery data
        """
        for gen_name in self.all_generators:
            dss.Generators.Name(gen_name)
            powers = dss.CktElement.Powers()
            
            if gen_name not in self.results['generator_powers']:
                self.results['generator_powers'][gen_name] = {'kw': [], 'kvar': []}
            
            if len(powers) >= 2:
                self.results['generator_powers'][gen_name]['kw'].append(-powers[0])  # Negative for generation
                self.results['generator_powers'][gen_name]['kvar'].append(-powers[1])
            else:
                self.results['generator_powers'][gen_name]['kw'].append(0)
                self.results['generator_powers'][gen_name]['kvar'].append(0)
    
    def _collect_capacitor_data(self, hour):
        """
        Collect capacitor data
        """
        for cap_name in self.all_capacitors:
            dss.Capacitors.Name(cap_name)
            
            if cap_name not in self.results['capacitor_kvar']:
                self.results['capacitor_kvar'][cap_name] = []
            
            # Get capacitor reactive power
            kvar = dss.Capacitors.kvar()
            self.results['capacitor_kvar'][cap_name].append(kvar)
    
    def _collect_regulator_data(self, hour):
        """
        Collect regulator tap data
        """
        # Get regulator transformers
        reg_transformers = [name for name in self.all_transformers if 'reg' in name.lower()]
        
        for reg_name in reg_transformers:
            dss.Transformers.Name(reg_name)
            
            if reg_name not in self.results['regulator_taps']:
                self.results['regulator_taps'][reg_name] = []
            
            # Get tap position
            tap = dss.Transformers.Tap()
            self.results['regulator_taps'][reg_name].append(tap)
    
    def _collect_line_data(self, hour):
        """
        Collect line current and power data
        """
        for line_name in self.all_lines:
            dss.Lines.Name(line_name)
            
            if line_name not in self.results['line_currents']:
                self.results['line_currents'][line_name] = []
                self.results['line_powers'][line_name] = {'kw': [], 'kvar': []}
            
            # Get line currents
            currents = dss.CktElement.CurrentsMagAng()
            if currents:
                # Average current magnitude
                current_mags = [currents[i] for i in range(0, len(currents), 2)]
                avg_current = np.mean(current_mags) if current_mags else 0
                self.results['line_currents'][line_name].append(avg_current)
            else:
                self.results['line_currents'][line_name].append(0)
            
            # Get line powers
            powers = dss.CktElement.Powers()
            if len(powers) >= 2:
                self.results['line_powers'][line_name]['kw'].append(powers[0])
                self.results['line_powers'][line_name]['kvar'].append(powers[1])
            else:
                self.results['line_powers'][line_name]['kw'].append(0)
                self.results['line_powers'][line_name]['kvar'].append(0)
    
    def _collect_transformer_data(self, hour):
        """
        Collect transformer data
        """
        for xfmr_name in self.all_transformers:
            if 'reg' not in xfmr_name.lower():  # Skip regulator transformers
                dss.Transformers.Name(xfmr_name)
                
                if xfmr_name not in self.results['transformer_powers']:
                    self.results['transformer_powers'][xfmr_name] = {'kw': [], 'kvar': []}
                    self.results['transformer_loadings'][xfmr_name] = []
                
                # Get transformer powers
                powers = dss.CktElement.Powers()
                if len(powers) >= 2:
                    self.results['transformer_powers'][xfmr_name]['kw'].append(powers[0])
                    self.results['transformer_powers'][xfmr_name]['kvar'].append(powers[1])
                else:
                    self.results['transformer_powers'][xfmr_name]['kw'].append(0)
                    self.results['transformer_powers'][xfmr_name]['kvar'].append(0)
                
                # Calculate loading percentage
                kva_rating = dss.Transformers.kVA()
                if kva_rating > 0 and len(powers) >= 2:
                    apparent_power = np.sqrt(powers[0]**2 + powers[1]**2)
                    loading_pct = (apparent_power / kva_rating) * 100
                    self.results['transformer_loadings'][xfmr_name].append(loading_pct)
                else:
                    self.results['transformer_loadings'][xfmr_name].append(0)
    
    def save_results_to_csv(self):
        """
        Save simulation results to CSV files
        """
        print("\nSaving results to CSV files...")
        
        # Main results DataFrame
        main_df = pd.DataFrame({
            'Time': self.results['time'],
            'Hour': self.results['hour'],
            'Voltage_Min_pu': self.results['voltage_min'],
            'Voltage_Max_pu': self.results['voltage_max'],
            'Voltage_Avg_pu': self.results['voltage_avg'],
            'Total_Load_kW': self.results['total_load_kw'],
            'Total_Load_kVAR': self.results['total_load_kvar'],
            'Source_kW': self.results['source_kw'],
            'Source_kVAR': self.results['source_kvar'],
            'Losses_kW': self.results['total_losses_kw'],
            'Losses_kVAR': self.results['total_losses_kvar']
        })
        
        main_df.to_csv(os.path.join(self.results_dir, 'daily_simulation_summary.csv'), index=False)
        
        # Bus voltages
        voltage_df = pd.DataFrame(self.results['bus_voltages'])
        voltage_df['Time'] = self.results['time']
        voltage_df.to_csv(os.path.join(self.results_dir, 'bus_voltages.csv'), index=False)
        
        # Load powers
        load_data = []
        for load_name, powers in self.results['load_powers'].items():
            for i, (kw, kvar) in enumerate(zip(powers['kw'], powers['kvar'])):
                load_data.append({
                    'Time': self.results['time'][i],
                    'Hour': i,
                    'Load': load_name,
                    'kW': kw,
                    'kVAR': kvar
                })
        
        load_df = pd.DataFrame(load_data)
        load_df.to_csv(os.path.join(self.results_dir, 'load_powers.csv'), index=False)
        
        print(f"Results saved to {self.results_dir}")
    
    def plot_comprehensive_analysis(self):
        """
        Generate comprehensive analysis plots
        """
        print("\nGenerating comprehensive analysis plots...")
        
        # Plot 1: System Overview
        self._plot_system_overview()
        
        # Plot 2: Voltage Analysis
        self._plot_voltage_analysis()
        
        # Plot 3: Power Flow Analysis
        self._plot_power_analysis()
        
        # Plot 4: Load Analysis
        self._plot_load_analysis()
        
        # Plot 5: Equipment Analysis
        self._plot_equipment_analysis()
        
        # Plot 6: Loss Analysis
        self._plot_loss_analysis()
        
        # Plot 7: Individual Bus Voltages
        self._plot_bus_voltages()
        
        # Plot 8: Line Loading Analysis
        self._plot_line_analysis()
        
        print("All analysis plots generated successfully")
    
    def _plot_system_overview(self):
        """
        Plot system overview
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IEEE 34-Node Daily Simulation - System Overview', fontsize=16, fontweight='bold')
        
        hours = self.results['hour']
        
        # Voltage profile
        axes[0, 0].plot(hours, self.results['voltage_min'], 'r-', label='Minimum', linewidth=2)
        axes[0, 0].plot(hours, self.results['voltage_max'], 'b-', label='Maximum', linewidth=2)
        axes[0, 0].plot(hours, self.results['voltage_avg'], 'g-', label='Average', linewidth=2)
        axes[0, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Min Limit (0.95)')
        axes[0, 0].axhline(y=1.05, color='r', linestyle='--', alpha=0.7, label='Max Limit (1.05)')
        axes[0, 0].set_title('System Voltage Profile')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Voltage (p.u.)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total power
        axes[0, 1].plot(hours, self.results['total_load_kw'], 'b-', label='Load kW', linewidth=2)
        axes[0, 1].plot(hours, self.results['source_kw'], 'r-', label='Source kW', linewidth=2)
        axes[0, 1].set_title('Active Power Flow')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Power (kW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reactive power
        axes[1, 0].plot(hours, self.results['total_load_kvar'], 'b-', label='Load kVAR', linewidth=2)
        axes[1, 0].plot(hours, self.results['source_kvar'], 'r-', label='Source kVAR', linewidth=2)
        axes[1, 0].set_title('Reactive Power Flow')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Power (kVAR)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # System losses
        axes[1, 1].plot(hours, self.results['total_losses_kw'], 'r-', label='Active Losses', linewidth=2)
        axes[1, 1].plot(hours, self.results['total_losses_kvar'], 'b-', label='Reactive Losses', linewidth=2)
        axes[1, 1].set_title('System Losses')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Losses (kW/kVAR)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'system_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("System overview plot saved")
    
    def _plot_voltage_analysis(self):
        """
        Plot detailed voltage analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IEEE 34-Node Daily Simulation - Voltage Analysis', fontsize=16, fontweight='bold')
        
        hours = self.results['hour']
        
        # Voltage statistics
        axes[0, 0].fill_between(hours, self.results['voltage_min'], self.results['voltage_max'], 
                               alpha=0.3, color='blue', label='Voltage Range')
        axes[0, 0].plot(hours, self.results['voltage_avg'], 'r-', label='Average', linewidth=2)
        axes[0, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(y=1.05, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('System Voltage Range')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Voltage (p.u.)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Voltage deviation
        voltage_dev = [abs(v - 1.0) for v in self.results['voltage_avg']]
        axes[0, 1].plot(hours, voltage_dev, 'g-', linewidth=2)
        axes[0, 1].set_title('Average Voltage Deviation from Nominal')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Voltage Deviation (p.u.)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Voltage unbalance (max - min)
        voltage_unbalance = [max_v - min_v for max_v, min_v in 
                           zip(self.results['voltage_max'], self.results['voltage_min'])]
        axes[1, 0].plot(hours, voltage_unbalance, 'orange', linewidth=2)
        axes[1, 0].set_title('System Voltage Unbalance')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Voltage Spread (p.u.)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Voltage histogram for peak hour
        peak_hour = np.argmax(self.results['total_load_kw'])
        peak_voltages = []
        for bus_name in self.all_buses:
            if bus_name in self.results['bus_voltages']:
                peak_voltages.append(self.results['bus_voltages'][bus_name][peak_hour])
        
        axes[1, 1].hist(peak_voltages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(x=0.95, color='r', linestyle='--', label='Min Limit')
        axes[1, 1].axvline(x=1.05, color='r', linestyle='--', label='Max Limit')
        axes[1, 1].set_title(f'Voltage Distribution at Peak Hour ({peak_hour})')
        axes[1, 1].set_xlabel('Voltage (p.u.)')
        axes[1, 1].set_ylabel('Number of Buses')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'voltage_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Voltage analysis plot saved")
    
    def _plot_power_analysis(self):
        """
        Plot power analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IEEE 34-Node Daily Simulation - Power Analysis', fontsize=16, fontweight='bold')
        
        hours = self.results['hour']
        
        # Power balance
        axes[0, 0].plot(hours, self.results['source_kw'], 'r-', label='Source kW', linewidth=2)
        axes[0, 0].plot(hours, self.results['total_load_kw'], 'b-', label='Load kW', linewidth=2)
        axes[0, 0].plot(hours, self.results['total_losses_kw'], 'g-', label='Losses kW', linewidth=2)
        axes[0, 0].set_title('Active Power Balance')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power factor
        power_factor = []
        for i in range(len(hours)):
            if self.results['source_kw'][i] != 0:
                s_apparent = np.sqrt(self.results['source_kw'][i]**2 + self.results['source_kvar'][i]**2)
                pf = abs(self.results['source_kw'][i]) / s_apparent if s_apparent > 0 else 0
                power_factor.append(pf)
            else:
                power_factor.append(0)
        
        axes[0, 1].plot(hours, power_factor, 'purple', linewidth=2)
        axes[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Target (0.9)')
        axes[0, 1].set_title('System Power Factor')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Power Factor')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss percentage
        loss_percentage = []
        for i in range(len(hours)):
            if self.results['source_kw'][i] > 0:
                loss_pct = (self.results['total_losses_kw'][i] / self.results['source_kw'][i]) * 100
                loss_percentage.append(loss_pct)
            else:
                loss_percentage.append(0)
        
        axes[1, 0].plot(hours, loss_percentage, 'red', linewidth=2)
        axes[1, 0].set_title('System Loss Percentage')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Losses (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Energy consumption
        energy_consumption = np.cumsum(self.results['total_load_kw'])
        axes[1, 1].plot(hours, energy_consumption, 'blue', linewidth=2)
        axes[1, 1].set_title('Cumulative Energy Consumption')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Energy (kWh)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'power_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Power analysis plot saved")
    
    def _plot_load_analysis(self):
        """
        Plot load analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IEEE 34-Node Daily Simulation - Load Analysis', fontsize=16, fontweight='bold')
        
        hours = self.results['hour']
        
        # Top 5 loads by peak power
        load_peaks = {}
        for load_name, powers in self.results['load_powers'].items():
            load_peaks[load_name] = max(powers['kw'])
        
        top_loads = sorted(load_peaks.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Plot top loads
        for load_name, _ in top_loads:
            axes[0, 0].plot(hours, self.results['load_powers'][load_name]['kw'], 
                           label=load_name, linewidth=2)
        
        axes[0, 0].set_title('Top 5 Loads - Active Power')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Load diversity
        load_diversity = []
        for i in range(len(hours)):
            individual_peaks = [max(powers['kw']) for powers in self.results['load_powers'].values()]
            sum_individual_peaks = sum(individual_peaks)
            system_peak = self.results['total_load_kw'][i]
            diversity = system_peak / sum_individual_peaks if sum_individual_peaks > 0 else 0
            load_diversity.append(diversity)
        
        axes[0, 1].plot(hours, load_diversity, 'green', linewidth=2)
        axes[0, 1].set_title('Load Diversity Factor')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Diversity Factor')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Load distribution pie chart (peak hour)
        peak_hour = np.argmax(self.results['total_load_kw'])
        load_values = []
        load_labels = []
        
        for load_name, powers in self.results['load_powers'].items():
            if powers['kw'][peak_hour] > 10:  # Only show loads > 10 kW
                load_values.append(powers['kw'][peak_hour])
                load_labels.append(load_name)
        
        # Group small loads
        other_loads = sum([powers['kw'][peak_hour] for load_name, powers in self.results['load_powers'].items() 
                          if powers['kw'][peak_hour] <= 10])
        if other_loads > 0:
            load_values.append(other_loads)
            load_labels.append('Others')
        
        axes[1, 0].pie(load_values, labels=load_labels, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title(f'Load Distribution at Peak Hour ({peak_hour})')
        
        # Load factor analysis
        load_factors = []
        for load_name, powers in self.results['load_powers'].items():
            if powers['kw']:
                avg_power = np.mean(powers['kw'])
                peak_power = max(powers['kw'])
                load_factor = avg_power / peak_power if peak_power > 0 else 0
                load_factors.append(load_factor)
        
        axes[1, 1].hist(load_factors, bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Load Factor Distribution')
        axes[1, 1].set_xlabel('Load Factor')
        axes[1, 1].set_ylabel('Number of Loads')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'load_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Load analysis plot saved")
    
    def _plot_equipment_analysis(self):
        """
        Plot equipment analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IEEE 34-Node Daily Simulation - Equipment Analysis', fontsize=16, fontweight='bold')
        
        hours = self.results['hour']
        
        # Capacitor reactive power
        if self.results['capacitor_kvar']:
            for cap_name, kvar_values in self.results['capacitor_kvar'].items():
                axes[0, 0].plot(hours, kvar_values, label=cap_name, linewidth=2)
            
            axes[0, 0].set_title('Capacitor Reactive Power')
            axes[0, 0].set_xlabel('Hour')
            axes[0, 0].set_ylabel('Reactive Power (kVAR)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Capacitor Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # Regulator taps
        if self.results['regulator_taps']:
            for reg_name, tap_values in self.results['regulator_taps'].items():
                axes[0, 1].plot(hours, tap_values, label=reg_name, linewidth=2, marker='o')
            
            axes[0, 1].set_title('Regulator Tap Positions')
            axes[0, 1].set_xlabel('Hour')
            axes[0, 1].set_ylabel('Tap Position')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Regulator Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Transformer loading
        if self.results['transformer_loadings']:
            for xfmr_name, loading_values in self.results['transformer_loadings'].items():
                axes[1, 0].plot(hours, loading_values, label=xfmr_name, linewidth=2)
            
            axes[1, 0].axhline(y=100, color='r', linestyle='--', alpha=0.7, label='100% Loading')
            axes[1, 0].set_title('Transformer Loading')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Loading (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Transformer Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Generator output
        if self.results['generator_powers']:
            for gen_name, powers in self.results['generator_powers'].items():
                axes[1, 1].plot(hours, powers['kw'], label=f'{gen_name} kW', linewidth=2)
            
            axes[1, 1].set_title('Generator Output')
            axes[1, 1].set_xlabel('Hour')
            axes[1, 1].set_ylabel('Power (kW)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Generator Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'equipment_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Equipment analysis plot saved")
    
    def _plot_loss_analysis(self):
        """
        Plot loss analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IEEE 34-Node Daily Simulation - Loss Analysis', fontsize=16, fontweight='bold')
        
        hours = self.results['hour']
        
        # Total losses
        axes[0, 0].plot(hours, self.results['total_losses_kw'], 'r-', label='Active Losses', linewidth=2)
        axes[0, 0].plot(hours, self.results['total_losses_kvar'], 'b-', label='Reactive Losses', linewidth=2)
        axes[0, 0].set_title('System Losses')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Losses (kW/kVAR)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss vs load relationship
        axes[0, 1].scatter(self.results['total_load_kw'], self.results['total_losses_kw'], 
                          alpha=0.7, color='red')
        axes[0, 1].set_title('Losses vs Load Relationship')
        axes[0, 1].set_xlabel('Total Load (kW)')
        axes[0, 1].set_ylabel('Total Losses (kW)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss percentage over time
        loss_percentage = []
        for i in range(len(hours)):
            if self.results['source_kw'][i] > 0:
                loss_pct = (self.results['total_losses_kw'][i] / self.results['source_kw'][i]) * 100
                loss_percentage.append(loss_pct)
            else:
                loss_percentage.append(0)
        
        axes[1, 0].plot(hours, loss_percentage, 'green', linewidth=2)
        axes[1, 0].set_title('Loss Percentage Over Time')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Losses (% of Generation)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative energy losses
        cumulative_losses = np.cumsum(self.results['total_losses_kw'])
        cumulative_energy = np.cumsum(self.results['source_kw'])
        
        axes[1, 1].plot(hours, cumulative_losses, 'r-', label='Cumulative Losses', linewidth=2)
        axes[1, 1].plot(hours, cumulative_energy, 'b-', label='Cumulative Generation', linewidth=2)
        axes[1, 1].set_title('Cumulative Energy')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Energy (kWh)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'loss_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Loss analysis plot saved")
    
    def _plot_bus_voltages(self):
        """
        Plot individual bus voltages
        """
        # Select important buses for plotting
        important_buses = ['800', '802', '806', '808', '812', '814', '816', '820', '822', 
                          '824', '828', '830', '832', '834', '836', '840', '842', '844', 
                          '846', '848', '850', '852', '854', '856', '858', '860', '862', 
                          '864', '888', '890']
        
        # Filter buses that exist in results
        available_buses = [bus for bus in important_buses if bus in self.results['bus_voltages']]
        
        if not available_buses:
            print("No bus voltage data available for plotting")
            return
        
        # Create subplots
        n_buses = len(available_buses)
        n_cols = 4
        n_rows = (n_buses + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        fig.suptitle('IEEE 34-Node Daily Simulation - Individual Bus Voltages', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        hours = self.results['hour']
        
        for i, bus_name in enumerate(available_buses):
            row = i // n_cols
            col = i % n_cols
            
            voltages = self.results['bus_voltages'][bus_name]
            axes[row, col].plot(hours, voltages, 'b-', linewidth=2)
            axes[row, col].axhline(y=0.95, color='r', linestyle='--', alpha=0.7)
            axes[row, col].axhline(y=1.05, color='r', linestyle='--', alpha=0.7)
            axes[row, col].set_title(f'Bus {bus_name}')
            axes[row, col].set_xlabel('Hour')
            axes[row, col].set_ylabel('Voltage (p.u.)')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_ylim(0.9, 1.1)
        
        # Hide unused subplots
        for i in range(n_buses, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'bus_voltages.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Bus voltages plot saved")
    
    def _plot_line_analysis(self):
        """
        Plot line analysis
        """
        if not self.results['line_currents']:
            print("No line data available for plotting")
            return
        
        # Select top lines by peak current
        line_peaks = {}
        for line_name, currents in self.results['line_currents'].items():
            line_peaks[line_name] = max(currents) if currents else 0
        
        top_lines = sorted(line_peaks.items(), key=lambda x: x[1], reverse=True)[:8]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IEEE 34-Node Daily Simulation - Line Analysis', fontsize=16, fontweight='bold')
        
        hours = self.results['hour']
        
        # Top line currents
        for line_name, _ in top_lines[:4]:
            axes[0, 0].plot(hours, self.results['line_currents'][line_name], 
                           label=line_name, linewidth=2)
        
        axes[0, 0].set_title('Top Line Currents')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Current (A)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Line power flows
        for line_name, _ in top_lines[:4]:
            axes[0, 1].plot(hours, self.results['line_powers'][line_name]['kw'], 
                           label=line_name, linewidth=2)
        
        axes[0, 1].set_title('Top Line Power Flows')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Power (kW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Current distribution at peak hour
        peak_hour = np.argmax(self.results['total_load_kw'])
        peak_currents = []
        for line_name in self.results['line_currents']:
            if self.results['line_currents'][line_name]:
                peak_currents.append(self.results['line_currents'][line_name][peak_hour])
        
        axes[1, 0].hist(peak_currents, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 0].set_title(f'Line Current Distribution at Peak Hour ({peak_hour})')
        axes[1, 0].set_xlabel('Current (A)')
        axes[1, 0].set_ylabel('Number of Lines')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Line utilization (assuming 200A rating for example)
        line_utilization = []
        for line_name in self.results['line_currents']:
            if self.results['line_currents'][line_name]:
                max_current = max(self.results['line_currents'][line_name])
                utilization = (max_current / 200) * 100  # Assuming 200A rating
                line_utilization.append(utilization)
        
        axes[1, 1].hist(line_utilization, bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(x=100, color='r', linestyle='--', alpha=0.7, label='100% Utilization')
        axes[1, 1].set_title('Line Utilization Distribution')
        axes[1, 1].set_xlabel('Utilization (%)')
        axes[1, 1].set_ylabel('Number of Lines')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'line_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Line analysis plot saved")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\nGenerating summary report...")
        
        report = []
        report.append("IEEE 34-Node Daily Simulation Summary Report")
        report.append("=" * 50)
        report.append(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"DSS File: {self.dss_file_path}")
        report.append(f"Simulation Duration: {self.simulation_hours} hours")
        report.append("")
        
        # System statistics
        report.append("System Statistics:")
        report.append("-" * 20)
        report.append(f"Total Buses: {len(self.all_buses)}")
        report.append(f"Total Loads: {len(self.all_loads)}")
        report.append(f"Total Lines: {len(self.all_lines)}")
        report.append(f"Total Transformers: {len(self.all_transformers)}")
        report.append(f"Total Capacitors: {len(self.all_capacitors)}")
        report.append(f"Total Generators: {len(self.all_generators)}")
        report.append("")
        
        # Voltage analysis
        report.append("Voltage Analysis:")
        report.append("-" * 15)
        report.append(f"Minimum Voltage: {min(self.results['voltage_min']):.4f} p.u.")
        report.append(f"Maximum Voltage: {max(self.results['voltage_max']):.4f} p.u.")
        report.append(f"Average Voltage: {np.mean(self.results['voltage_avg']):.4f} p.u.")
        
        # Check voltage violations
        voltage_violations = sum(1 for v in self.results['voltage_min'] if v < 0.95) + \
                           sum(1 for v in self.results['voltage_max'] if v > 1.05)
        report.append(f"Voltage Violations: {voltage_violations} time points")
        report.append("")
        
        # Power analysis
        report.append("Power Analysis:")
        report.append("-" * 15)
        report.append(f"Peak Load: {max(self.results['total_load_kw']):.2f} kW")
        report.append(f"Minimum Load: {min(self.results['total_load_kw']):.2f} kW")
        report.append(f"Average Load: {np.mean(self.results['total_load_kw']):.2f} kW")
        report.append(f"Total Energy Consumed: {sum(self.results['total_load_kw']):.2f} kWh")
        report.append(f"Peak Losses: {max(self.results['total_losses_kw']):.2f} kW")
        report.append(f"Total Energy Losses: {sum(self.results['total_losses_kw']):.2f} kWh")
        
        # Loss percentage
        avg_loss_pct = np.mean([(loss/gen)*100 for loss, gen in 
                               zip(self.results['total_losses_kw'], self.results['source_kw']) 
                               if gen > 0])
        report.append(f"Average Loss Percentage: {avg_loss_pct:.2f}%")
        report.append("")
        
        # Load factor
        system_load_factor = np.mean(self.results['total_load_kw']) / max(self.results['total_load_kw'])
        report.append(f"System Load Factor: {system_load_factor:.3f}")
        report.append("")
        
        # Top loads
        report.append("Top 5 Loads by Peak Power:")
        report.append("-" * 30)
        load_peaks = {}
        for load_name, powers in self.results['load_powers'].items():
            load_peaks[load_name] = max(powers['kw'])
        
        top_loads = sorted(load_peaks.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (load_name, peak_power) in enumerate(top_loads, 1):
            report.append(f"{i}. {load_name}: {peak_power:.2f} kW")
        
        report.append("")
        report.append("Analysis complete. Check the generated plots for detailed visualizations.")
        
        # Save report
        report_text = "\n".join(report)
        with open(os.path.join(self.results_dir, 'simulation_report.txt'), 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nSummary report saved to {self.results_dir}")

def main():
    """
    Main function to run the IEEE 34-node daily simulation
    """
    # DSS file path
    dss_file = "/home/zhengxiaodong/exps/PowerZoo/envs/powerzoo_llm/node_systems_with_pv/34Bus/ieee34Mod1_daily.dss"
    
    if not os.path.exists(dss_file):
        print(f"Error: DSS file not found: {dss_file}")
        return
    
    try:
        # Create simulation instance
        simulation = IEEE34DailySimulation(dss_file)
        
        # Run simulation
        simulation.run_simulation()
        
        # Save results
        simulation.save_results_to_csv()
        
        # Generate plots
        simulation.plot_comprehensive_analysis()
        
        # Generate summary report
        simulation.generate_summary_report()
        
        print("\n=== Daily Simulation Completed Successfully ===")
        print(f"Results saved to: {simulation.results_dir}")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()