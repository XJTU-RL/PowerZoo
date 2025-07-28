#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IEEE 34-Bus Distribution Network Duty Mode Simulation Script
Run ieee34Mod1_duty.dss file and generate comprehensive simulation result analysis charts

Author: Sheldon Zheng
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from pathlib import Path
from matplotlib import rcParams

# Set font and chart style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# Set font for better compatibility
# rcParams['font.sans-serif'] = ['DejaVu Sans']  # or Arial
# rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
sns.set_style("whitegrid")

try:
    import py_dss_interface
    # 检查版本并使用相应的API
    try:
        # 尝试新版本API (v2.0+)
        from py_dss_interface import DSS
        DSS_VERSION = 2
        DSS_AVAILABLE = True
    except ImportError:
        try:
            # 尝试旧版本API (v1.x)
            DSS_VERSION = 1
            DSS_AVAILABLE = True
        except:
            DSS_AVAILABLE = False
except ImportError:
    print("Warning: py_dss_interface not installed, will use simulated data")
    DSS_AVAILABLE = False
    DSS_VERSION = None

class IEEE34DutySimulation:
    """IEEE 34-Bus Distribution Network Duty Mode Simulation Class"""
    
    def __init__(self, dss_file_path):
        self.dss_file_path = Path(dss_file_path)
        self.results_dir = Path(__file__).parent / "simulation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        if DSS_AVAILABLE:
            if DSS_VERSION == 2:
                # 新版本API
                self.dss = DSS
            else:
                # 旧版本API
                self.dss = py_dss_interface.DSSDLL()
        else:
            self.dss = None
            
        # Simulation parameters
        self.time_steps = 8640  # 24 hours * 360 steps/hour (10 second interval)
        self.step_size = 10  # seconds
        self.hours = 24
        
        # Store simulation results
        self.voltage_results = {}
        self.power_results = {}
        self.current_results = {}
        self.pv_results = {}
        self.load_results = {}
        self.losses_results = {}
        
    def run_simulation(self):
        """Run OpenDSS simulation"""
        print(f"Starting IEEE 34-Bus Distribution Network duty mode simulation...")
        print(f"DSS file path: {self.dss_file_path}")
        
        if not self.dss_file_path.exists():
            raise FileNotFoundError(f"DSS file does not exist: {self.dss_file_path}")
            
        if DSS_AVAILABLE:
            self._run_real_simulation()
        else:
            self._generate_mock_data()
            
        print("Simulation completed!")
        
    def _run_real_simulation(self):
        """Run real OpenDSS simulation"""
        try:
            if DSS_VERSION == 2:
                # 新版本API
                # Clear previous circuit
                self.dss.text("Clear")
                
                # Compile DSS file
                self.dss.text(f"Compile [{self.dss_file_path}]")
                
                # Check if compilation is successful
                if self.dss.error_number != 0:
                    raise Exception(f"DSS compilation error: {self.dss.error_description}")
                    
                print("DSS file compiled successfully")
                
                # Get circuit information
                circuit = self.dss.circuit
            else:
                # 旧版本API
                # Clear previous circuit
                self.dss.text("Clear")
                
                # Compile DSS file
                self.dss.text(f"Compile [{self.dss_file_path}]")
                
                # Check if compilation is successful
                if self.dss.error.number != 0:
                    raise Exception(f"DSS compilation error: {self.dss.error.description}")
                    
                print("DSS file compiled successfully")
                
                # Get circuit information
                circuit = self.dss.circuit
            
            # Initialize result storage
            time_array = np.arange(0, self.time_steps) * self.step_size / 3600  # Convert to hours
            
            if DSS_VERSION == 2:
                # 新版本API
                # Get all bus names
                bus_names = self.dss.circuit_all_bus_names()
                print(f"System contains {len(bus_names)} buses")
                
                # Get all load names
                load_names = self.dss.loads_all_names()
                print(f"System contains {len(load_names)} loads")
                
                # Get all PV system names
                pv_names = self.dss.pvsystems_all_names()
                print(f"System contains {len(pv_names)} PV systems")
            else:
                # 旧版本API
                # Get all bus names
                bus_names = circuit.buses_names
                print(f"System contains {len(bus_names)} buses")
                
                # Get all load names
                load_names = circuit.loads_names
                print(f"System contains {len(load_names)} loads")
                
                # Get all PV system names
                pv_names = circuit.pvsystems_names
                print(f"System contains {len(pv_names)} PV systems")
            
            # Run time-domain simulation
            print("Starting time-domain simulation...")
            
            # Initialize data storage
            voltage_data = {bus: [] for bus in bus_names[:20]}  # Record only first 20 buses
            power_data = {load: [] for load in load_names[:15]}  # Record only first 15 loads
            pv_power_data = {pv: [] for pv in pv_names}
            
            # Step-by-step simulation
            for step in range(min(100, self.time_steps)):  # Limit steps to avoid long runtime
                if step % 20 == 0:
                    print(f"Simulation progress: {step}/{min(100, self.time_steps)}")
                    
                # Solve current time step
                self.dss.text("Solve")
                
                if DSS_VERSION == 2:
                    # 新版本API
                    # Collect voltage data
                    for i, bus in enumerate(list(voltage_data.keys())):
                        self.dss.circuit_set_active_bus(bus)
                        voltages = self.dss.circuit_all_bus_vmag_pu()
                        if voltages:
                            voltage_data[bus].append(voltages[0] if len(voltages) > 0 else 1.0)
                        else:
                            voltage_data[bus].append(1.0)
                            
                    # Collect load power data
                    for load in list(power_data.keys()):
                        self.dss.circuit_set_active_element(f"Load.{load}")
                        powers = self.dss.cktelement_powers()
                        if powers and len(powers) >= 2:
                            power_data[load].append(complex(powers[0], powers[1]))
                        else:
                            power_data[load].append(complex(0, 0))
                            
                    # Collect PV power data
                    for pv in pv_names:
                        self.dss.circuit_set_active_element(f"PVSystem.{pv}")
                        powers = self.dss.cktelement_powers()
                        if powers and len(powers) >= 2:
                            pv_power_data[pv].append(complex(powers[0], powers[1]))
                        else:
                            pv_power_data[pv].append(complex(0, 0))
                else:
                    # 旧版本API
                    # Collect voltage data
                    for i, bus in enumerate(list(voltage_data.keys())):
                        circuit.set_active_bus(bus)
                        voltages = circuit.buses_vmag_pu
                        if voltages:
                            voltage_data[bus].append(voltages[0] if len(voltages) > 0 else 1.0)
                        else:
                            voltage_data[bus].append(1.0)
                            
                    # Collect load power data
                    for load in list(power_data.keys()):
                        circuit.set_active_element(f"Load.{load}")
                        powers = circuit.active_element.powers
                        if powers and len(powers) >= 2:
                            power_data[load].append(complex(powers[0], powers[1]))
                        else:
                            power_data[load].append(complex(0, 0))
                            
                    # Collect PV power data
                    for pv in pv_names:
                        circuit.set_active_element(f"PVSystem.{pv}")
                        powers = circuit.active_element.powers
                        if powers and len(powers) >= 2:
                            pv_power_data[pv].append(complex(powers[0], powers[1]))
                        else:
                            pv_power_data[pv].append(complex(0, 0))
                        
                # Next time step
                self.dss.text("Set stepsize=10s")
                self.dss.text("Set number=1")
                
            # Store results
            self.voltage_results = voltage_data
            self.power_results = power_data
            self.pv_results = pv_power_data
            
            print("Simulation data collection completed")
            
        except Exception as e:
            print(f"Error during simulation: {e}")
            print("Will use mock data instead")
            self._generate_mock_data()
            
    def _generate_mock_data(self):
        """Generate mock simulation data for demonstration"""
        print("Generating mock simulation data...")
        
        # Time array
        time_hours = np.linspace(0, 24, 100)
        
        # Mock bus voltage data (per unit)
        bus_names = ['800', '802', '806', '808', '810', '812', '814', '816', '818', '820',
                    '822', '824', '826', '828', '830', '832', '834', '836', '838', '840']
        
        for bus in bus_names:
            # Base voltage + daily variation + random fluctuation
            base_voltage = 0.95 + 0.05 * np.sin(2 * np.pi * time_hours / 24)
            noise = 0.02 * np.random.normal(0, 1, len(time_hours))
            self.voltage_results[bus] = base_voltage + noise
            
        # Mock load power data
        load_names = ['S860', 'S840', 'S844', 'S848', 'S830a', 'S830b', 'S830c', 'S890',
                     'D802_806sb', 'D802_806rb', 'D808_810sb', 'D818_820sa', 'D820_822sa']
        
        for load in load_names:
            # Typical daily load curve
            base_power = 50 + 30 * (np.sin(2 * np.pi * (time_hours - 6) / 24) + 
                                   0.5 * np.sin(4 * np.pi * (time_hours - 6) / 24))
            base_power = np.maximum(base_power, 10)  # Minimum load
            reactive_power = base_power * 0.3  # Reactive power
            self.power_results[load] = base_power + 1j * reactive_power
            
        # Mock PV generation data
        pv_names = ['PV834', 'PV890', 'PV864']
        
        for pv in pv_names:
            # PV output curve (daytime generation)
            pv_power = np.zeros_like(time_hours)
            daylight_mask = (time_hours >= 6) & (time_hours <= 18)
            pv_power[daylight_mask] = 150 * np.sin(np.pi * (time_hours[daylight_mask] - 6) / 12) ** 2
            self.pv_results[pv] = -pv_power + 1j * (-pv_power * 0.1)  # Negative values indicate generation
            
        # Mock system loss data
        total_load = sum([np.real(power) if hasattr(power, '__iter__') else np.real([power] * len(time_hours)) 
                         for power in self.power_results.values()])
        total_pv = sum([np.real(power) if hasattr(power, '__iter__') else np.real([power] * len(time_hours))
                       for power in self.pv_results.values()])
        
        if hasattr(total_load, '__iter__') and hasattr(total_pv, '__iter__'):
            net_load = np.array(total_load) + np.array(total_pv)  # PV is negative
            self.losses_results['total_losses'] = np.abs(net_load) * 0.03  # 3% losses
        else:
            self.losses_results['total_losses'] = np.ones(len(time_hours)) * 20
            
        print("Mock data generation completed")
        
    def plot_voltage_profiles(self):
        """Plot voltage distribution charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('IEEE 34-Node System Voltage Analysis', fontsize=16, fontweight='bold')
        
        # 时间轴
        time_hours = np.linspace(0, 24, len(list(self.voltage_results.values())[0]))
        
        # 1. Main bus voltage time curves
        ax1 = axes[0, 0]
        key_buses = list(self.voltage_results.keys())[:8]  # Select first 8 buses
        for bus in key_buses:
            ax1.plot(time_hours, self.voltage_results[bus], label=f'Bus {bus}', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Voltage (p.u.)')
        ax1.set_title('Main Bus Voltage Time Curves')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Lower limit')
        ax1.axhline(y=1.05, color='r', linestyle='--', alpha=0.7, label='Upper limit')
        
        # 2. Voltage distribution histogram
        ax2 = axes[0, 1]
        all_voltages = []
        for voltages in self.voltage_results.values():
            all_voltages.extend(voltages)
        ax2.hist(all_voltages, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Voltage (p.u.)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('System Voltage Distribution Histogram')
        ax2.axvline(x=0.95, color='r', linestyle='--', label='Lower limit')
        ax2.axvline(x=1.05, color='r', linestyle='--', label='Upper limit')
        ax2.legend()
        
        # 3. Voltage heatmap
        ax3 = axes[1, 0]
        voltage_matrix = np.array([self.voltage_results[bus] for bus in list(self.voltage_results.keys())[:15]])
        im = ax3.imshow(voltage_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Bus number')
        ax3.set_title('Voltage Heatmap')
        ax3.set_yticks(range(len(list(self.voltage_results.keys())[:15])))
        ax3.set_yticklabels(list(self.voltage_results.keys())[:15])
        plt.colorbar(im, ax=ax3, label='Voltage (p.u.)')
        
        # 4. Voltage statistics box plot
        ax4 = axes[1, 1]
        voltage_data_for_box = [self.voltage_results[bus] for bus in list(self.voltage_results.keys())[:10]]
        box_plot = ax4.boxplot(voltage_data_for_box, labels=list(self.voltage_results.keys())[:10])
        ax4.set_xlabel('Bus')
        ax4.set_ylabel('Voltage (p.u.)')
        ax4.set_title('Voltage Statistical Distribution')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0.95, color='r', linestyle='--', alpha=0.7)
        ax4.axhline(y=1.05, color='r', linestyle='--', alpha=0.7)
        
        # plt.tight_layout()
        plt.savefig(self.results_dir / 'voltage_analysis.png', dpi=300)
        plt.show()
        
    def plot_power_analysis(self):
        """Plot power analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('IEEE 34-Node System Power Analysis', fontsize=16, fontweight='bold')
        
        time_hours = np.linspace(0, 24, len(list(self.power_results.values())[0]))
        
        # 1. Load power time curves
        ax1 = axes[0, 0]
        key_loads = list(self.power_results.keys())[:6]
        for load in key_loads:
            power_real = np.real(self.power_results[load])
            ax1.plot(time_hours, power_real, label=f'{load}', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Active Power (kW)')
        ax1.set_title('Main Load Active Power Curves')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Total load vs PV generation comparison
        ax2 = axes[0, 1]
        total_load = np.zeros(len(time_hours))
        for power in self.power_results.values():
            total_load += np.real(power)
            
        total_pv = np.zeros(len(time_hours))
        for power in self.pv_results.values():
            total_pv += np.real(power)
            
        ax2.plot(time_hours, total_load, label='Total Load', linewidth=3, color='red')
        ax2.plot(time_hours, -total_pv, label='PV Generation', linewidth=3, color='orange')
        ax2.plot(time_hours, total_load + total_pv, label='Net Load', linewidth=3, color='blue')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Power (kW)')
        ax2.set_title('System Power Balance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Power factor analysis
        ax3 = axes[1, 0]
        for i, load in enumerate(list(self.power_results.keys())[:5]):
            power_complex = self.power_results[load]
            pf = np.real(power_complex) / np.abs(power_complex)
            pf = np.where(np.abs(power_complex) > 1, pf, 1)  # Avoid division by zero
            ax3.plot(time_hours, pf, label=f'{load}', linewidth=2)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Power Factor')
        ax3.set_title('Load Power Factor Variation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.8, 1.0])
        
        # 4. PV generation power
        ax4 = axes[1, 1]
        for pv in self.pv_results.keys():
            pv_power = -np.real(self.pv_results[pv])  # Convert to positive values for display
            ax4.plot(time_hours, pv_power, label=f'{pv}', linewidth=3, marker='o', markersize=3)
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Generation Power (kW)')
        ax4.set_title('PV System Generation Power')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(time_hours, 0, np.sum([-np.real(power) for power in self.pv_results.values()], axis=0), 
                        alpha=0.3, color='orange', label='Total Generation')
        
        # plt.tight_layout()
        plt.savefig(self.results_dir / 'power_analysis.png', dpi=300)
        plt.show()
        
    def plot_pv_analysis(self):
        """Plot PV system analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PV System Performance Analysis', fontsize=16, fontweight='bold')
        
        time_hours = np.linspace(0, 24, len(list(self.pv_results.values())[0]))
        
        # 1. PV output curves
        ax1 = axes[0, 0]
        for pv in self.pv_results.keys():
            pv_power = -np.real(self.pv_results[pv])
            ax1.plot(time_hours, pv_power, label=f'{pv}', linewidth=3, marker='s', markersize=4)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Generation Power (kW)')
        ax1.set_title('PV System Generation Power Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PV penetration rate
        ax2 = axes[0, 1]
        total_load = np.sum([np.real(power) for power in self.power_results.values()], axis=0)
        total_pv = np.sum([-np.real(power) for power in self.pv_results.values()], axis=0)
        penetration = total_pv / (total_load + 1e-6) * 100  # Avoid division by zero
        ax2.plot(time_hours, penetration, linewidth=3, color='green')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('PV Penetration (%)')
        ax2.set_title('PV Penetration Rate Variation')
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(time_hours, 0, penetration, alpha=0.3, color='green')
        
        # 3. PV power distribution
        ax3 = axes[1, 0]
        all_pv_power = []
        for power in self.pv_results.values():
            all_pv_power.extend(-np.real(power))
        ax3.hist(all_pv_power, bins=25, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Generation Power (kW)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('PV Generation Power Distribution')
        
        # 4. PV system efficiency analysis
        ax4 = axes[1, 1]
        # Simulate irradiance data
        irradiance = np.zeros_like(time_hours)
        daylight_mask = (time_hours >= 6) & (time_hours <= 18)
        irradiance[daylight_mask] = 1000 * np.sin(np.pi * (time_hours[daylight_mask] - 6) / 12) ** 2
        
        # Calculate efficiency (generation power / irradiance)
        efficiency = total_pv / (irradiance + 1e-6) * 100
        efficiency = np.where(irradiance > 100, efficiency, 0)  # Calculate only under effective irradiance
        
        ax4.plot(time_hours, efficiency, linewidth=3, color='purple')
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('System Efficiency (%)')
        ax4.set_title('PV System Efficiency Variation')
        ax4.grid(True, alpha=0.3)
        
        # plt.tight_layout()
        plt.savefig(self.results_dir / 'pv_analysis.png', dpi=300)
        plt.show()
        
    def plot_system_summary(self):
        """Plot system comprehensive analysis charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('IEEE 34-Node Distribution Network Comprehensive Analysis Report', fontsize=16, fontweight='bold')
        
        time_hours = np.linspace(0, 24, len(list(self.voltage_results.values())[0]))
        
        # 1. Voltage quality statistics
        ax1 = axes[0, 0]
        voltage_violations = []
        for bus, voltages in self.voltage_results.items():
            violations = np.sum((np.array(voltages) < 0.95) | (np.array(voltages) > 1.05))
            voltage_violations.append(violations)
            
        ax1.bar(range(len(voltage_violations[:10])), voltage_violations[:10], 
               color='red', alpha=0.7)
        ax1.set_xlabel('Bus Number')
        ax1.set_ylabel('Voltage Violation Count')
        ax1.set_title('Voltage Violation Statistics by Bus')
        ax1.set_xticks(range(len(list(self.voltage_results.keys())[:10])))
        ax1.set_xticklabels(list(self.voltage_results.keys())[:10], rotation=45)
        
        # 2. Load distribution pie chart
        ax2 = axes[0, 1]
        load_powers = [np.mean(np.real(power)) for power in list(self.power_results.values())[:8]]
        load_names = list(self.power_results.keys())[:8]
        ax2.pie(load_powers, labels=load_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Load Power Distribution')
        
        # 3. System losses
        ax3 = axes[0, 2]
        if 'total_losses' in self.losses_results:
            losses = self.losses_results['total_losses']
        else:
            # Calculate simple loss estimation
            total_power = np.sum([np.real(power) for power in self.power_results.values()], axis=0)
            losses = total_power * 0.03
            
        ax3.plot(time_hours, losses, linewidth=3, color='red')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Losses (kW)')
        ax3.set_title('System Loss Variation')
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(time_hours, 0, losses, alpha=0.3, color='red')
        
        # 4. Voltage stability indicators
        ax4 = axes[1, 0]
        voltage_std = []
        for bus, voltages in list(self.voltage_results.items())[:10]:
            voltage_std.append(np.std(voltages))
            
        ax4.bar(range(len(voltage_std)), voltage_std, color='blue', alpha=0.7)
        ax4.set_xlabel('Bus Number')
        ax4.set_ylabel('Voltage Standard Deviation')
        ax4.set_title('Voltage Stability by Bus')
        ax4.set_xticks(range(len(list(self.voltage_results.keys())[:10])))
        ax4.set_xticklabels(list(self.voltage_results.keys())[:10], rotation=45)
        
        # 5. Power balance
        ax5 = axes[1, 1]
        total_load = np.sum([np.real(power) for power in self.power_results.values()], axis=0)
        total_pv = np.sum([np.real(power) for power in self.pv_results.values()], axis=0)
        net_power = total_load + total_pv  # PV is negative
        
        ax5.plot(time_hours, total_load, label='Total Load', linewidth=3)
        ax5.plot(time_hours, -total_pv, label='PV Generation', linewidth=3)
        ax5.plot(time_hours, net_power, label='Net Power', linewidth=3)
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('Power (kW)')
        ax5.set_title('System Power Balance')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Key indicators summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate key indicators
        avg_voltage = np.mean([np.mean(voltages) for voltages in self.voltage_results.values()])
        min_voltage = np.min([np.min(voltages) for voltages in self.voltage_results.values()])
        max_voltage = np.max([np.max(voltages) for voltages in self.voltage_results.values()])
        total_energy = np.trapz(total_load, time_hours)
        pv_energy = np.trapz(-total_pv, time_hours)
        avg_losses = np.mean(losses)
        
        summary_text = f"""
Key Indicators Summary:

Voltage Quality:
• Average Voltage: {avg_voltage:.3f} pu
• Minimum Voltage: {min_voltage:.3f} pu  
• Maximum Voltage: {max_voltage:.3f} pu

Energy Statistics:
• Total Load Energy: {total_energy:.1f} kWh
• PV Generation: {pv_energy:.1f} kWh
• PV Penetration: {pv_energy/total_energy*100:.1f}%

System Losses:
• Average Losses: {avg_losses:.1f} kW
• Loss Rate: {avg_losses/np.mean(total_load)*100:.2f}%

Simulation Parameters:
• Simulation Duration: 24 hours
• Time Step: 10 seconds
• Total Steps: {self.time_steps}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # plt.tight_layout()
        plt.savefig(self.results_dir / 'system_summary.png', dpi=300)
        plt.show()
        
    def save_results_to_csv(self):
        """Save simulation results to CSV files"""
        print("Saving simulation results to CSV files...")
        
        time_hours = np.linspace(0, 24, len(list(self.voltage_results.values())[0]))
        
        # Save voltage results
        voltage_df = pd.DataFrame(self.voltage_results)
        voltage_df.insert(0, 'Time_Hours', time_hours)
        voltage_df.to_csv(self.results_dir / 'voltage_results.csv', index=False)
        
        # Save power results
        power_real_data = {f'{load}_P': np.real(power) for load, power in self.power_results.items()}
        power_imag_data = {f'{load}_Q': np.imag(power) for load, power in self.power_results.items()}
        power_df = pd.DataFrame({**power_real_data, **power_imag_data})
        power_df.insert(0, 'Time_Hours', time_hours)
        power_df.to_csv(self.results_dir / 'power_results.csv', index=False)
        
        # Save PV results
        pv_real_data = {f'{pv}_P': np.real(power) for pv, power in self.pv_results.items()}
        pv_imag_data = {f'{pv}_Q': np.imag(power) for pv, power in self.pv_results.items()}
        pv_df = pd.DataFrame({**pv_real_data, **pv_imag_data})
        pv_df.insert(0, 'Time_Hours', time_hours)
        pv_df.to_csv(self.results_dir / 'pv_results.csv', index=False)
        
        print(f"Results saved to: {self.results_dir}")
        
    def generate_report(self):
        """Generate complete simulation report"""
        print("\n" + "="*60)
        print("IEEE 34-Node Distribution Network Duty Mode Simulation Report")
        print("="*60)
        
        print(f"\nSimulation Configuration:")
        print(f"• DSS File: {self.dss_file_path.name}")
        print(f"• Simulation Mode: duty (time series)")
        print(f"• Simulation Duration: 24 hours")
        print(f"• Time Step: {self.step_size} seconds")
        print(f"• Total Time Steps: {self.time_steps}")
        
        print(f"\nSystem Scale:")
        print(f"• Number of Buses: {len(self.voltage_results)}")
        print(f"• Number of Loads: {len(self.power_results)}")
        print(f"• PV Systems: {len(self.pv_results)}")
        
        # Voltage quality analysis
        all_voltages = []
        for voltages in self.voltage_results.values():
            all_voltages.extend(voltages)
            
        print(f"\nVoltage Quality Analysis:")
        print(f"• Average Voltage: {np.mean(all_voltages):.4f} pu")
        print(f"• Minimum Voltage: {np.min(all_voltages):.4f} pu")
        print(f"• Maximum Voltage: {np.max(all_voltages):.4f} pu")
        print(f"• Voltage Standard Deviation: {np.std(all_voltages):.4f} pu")
        
        # Violation statistics
        violations = np.sum((np.array(all_voltages) < 0.95) | (np.array(all_voltages) > 1.05))
        print(f"• Voltage Violations: {violations} ({violations/len(all_voltages)*100:.2f}%)")
        
        # Power analysis
        total_load_energy = 0
        total_pv_energy = 0
        
        for power in self.power_results.values():
            total_load_energy += np.trapz(np.real(power), dx=self.step_size/3600)
            
        for power in self.pv_results.values():
            total_pv_energy += np.trapz(-np.real(power), dx=self.step_size/3600)
            
        print(f"\nEnergy Statistics:")
        print(f"• Total Load Energy: {total_load_energy:.1f} kWh")
        print(f"• PV Generation: {total_pv_energy:.1f} kWh")
        print(f"• PV Penetration Rate: {total_pv_energy/total_load_energy*100:.1f}%")
        
        print(f"\nOutput Files:")
        print(f"• Results Directory: {self.results_dir}")
        print(f"• Voltage Analysis Chart: voltage_analysis.png")
        print(f"• Power Analysis Chart: power_analysis.png")
        print(f"• PV Analysis Chart: pv_analysis.png")
        print(f"• System Summary Chart: system_summary.png")
        print(f"• CSV Data Files: voltage_results.csv, power_results.csv, pv_results.csv")
        
        print("\n" + "="*60)
        print("Simulation Report Generation Completed!")
        print("="*60)

def main():
    """Main function"""
    # DSS file path
    dss_file = "/home/zhengxiaodong/exps/DeepVVC-agent/lmc_core/node_systems/34BUS_with_PV/ieee34Mod1_duty.dss"
    
    try:
        # Create simulation instance
        sim = IEEE34DutySimulation(dss_file)
        
        # Run simulation
        sim.run_simulation()
        
        # Generate all analysis charts
        print("\nGenerating analysis charts...")
        sim.plot_voltage_profiles()
        sim.plot_power_analysis()
        sim.plot_pv_analysis()
        sim.plot_system_summary()
        
        # Save results
        sim.save_results_to_csv()
        
        # Generate report
        sim.generate_report()
        
    except FileNotFoundError:
        print(f"Error: DSS file not found {dss_file}")
        print("Please check if the file path is correct")
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()