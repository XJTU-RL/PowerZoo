#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试PV注入效果的脚本
验证方案A：PV元件与曲线解耦，每个worker独立管理曲线文件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import opendssdirect as dss
import pandas as pd
from envs.smartgrid.data_process.loadprofile import LoadProfile
from envs.smartgrid.data_process.loadprofile_config import ConfigGenerator


def test_pv_injection(system_name="34Bus_PV_Aggressive", worker_id=0, episode_idx=1):
    """测试PV注入效果
    
    Args:
        system_name: 系统名称
        worker_id: Worker ID
        episode_idx: Episode索引
    """
    print(f"\n{'='*60}")
    print(f"测试PV注入 - 系统: {system_name}, Worker: {worker_id}")
    print(f"{'='*60}\n")
    
    # 1. 设置路径
    base_path = "/home/zhengxiaodong/exps/PowerZoo"
    dss_folder_path = os.path.join(base_path, "node_systems", system_name)
    dss_file = os.path.join(dss_folder_path, "ieee34Mod1_duty.dss")
    
    # PV和温度数据路径
    pv_data_path = os.path.join(base_path, "data/PV/expanded_irradiance")
    temp_data_path = os.path.join(base_path, "data/PV/generated_temperature")
    
    # 2. 生成PV曲线文件
    print("步骤1: 生成PV曲线文件...")
    config_gen = ConfigGenerator(dss_folder_path, steps=96, worker_idx=worker_id)
    
    # 生成指定episode的PV曲线
    success = config_gen.select_pv_temperature_profile(
        episode_idx=episode_idx,
        irradiation_path=pv_data_path,
        temperature_path=temp_data_path
    )
    
    if success:
        print(f"✓ 成功生成 pv_data_{worker_id}.dss")
    else:
        print(f"✗ 生成PV曲线失败，使用默认配置")
        # 生成默认配置
        config_gen.ensure_default_pv_config()
    
    # 3. 初始化OpenDSS并编译主DSS
    print("\n步骤2: 编译主DSS文件...")
    os.chdir(dss_folder_path)  # 切换到DSS文件目录
    
    dss.Basic.ClearAll()
    dss.Text.Command(f"compile {os.path.basename(dss_file)}")
    print(f"✓ 编译完成: {os.path.basename(dss_file)}")
    
    # 4. 注入worker特定的PV曲线文件
    print(f"\n步骤3: 注入Worker {worker_id} 的PV曲线...")
    pv_curve_file = f"pv_data_{worker_id}.dss"
    if os.path.exists(pv_curve_file):
        dss.Text.Command(f"redirect {pv_curve_file}")
        print(f"✓ 成功注入: {pv_curve_file}")
        
        # 5. 动态绑定曲线到PVSystem
        print("\n步骤4: 动态绑定曲线到PV系统...")
        # 使用正确的API获取PV系统列表
        pv_names = []
        dss.Circuit.SetActiveClass("PVSystem")
        elem = dss.ActiveClass.First()
        while elem > 0:
            full_name = dss.CktElement.Name()
            pv_name = full_name.split('.')[1] if '.' in full_name else full_name
            pv_names.append(pv_name)
            # 绑定曲线
            dss.Text.Command(f"PVSystem.{pv_name}.daily=MyIrrad")
            dss.Text.Command(f"PVSystem.{pv_name}.Tdaily=MyTemp")
            dss.Text.Command(f"PVSystem.{pv_name}.EffCurve=MyEff")
            dss.Text.Command(f"PVSystem.{pv_name}.P-TCurve=MyPvsT")
            print(f"  ✓ 绑定曲线到 {pv_name}")
            elem = dss.ActiveClass.Next()
    else:
        print(f"✗ 文件不存在: {pv_curve_file}")
        return
    
    # 6. 测试不同时间步的PV出力
    print("\n步骤5: 测试PV出力...")
    print("-" * 50)
    
    # 设置仿真模式
    dss.Text.Command("Set mode=duty")
    dss.Text.Command("Set stepsize=60")
    dss.Text.Command("Set number=1")
    
    # 测试多个时间步
    test_steps = [0, 6, 12, 24, 36, 48]  # 0点、6点、12点...
    
    for step in test_steps:
        # 设置时间
        hour = step // 60
        minute = step % 60
        dss.Text.Command(f"Set hour={hour}")
        dss.Text.Command(f"Set sec={minute*60}")
        
        # 求解
        dss.Solution.Solve()
        
        # 检查PV系统
        print(f"\n时间步 {step} (Hour={hour}:{minute:02d}):")
        
        # 显示时间信息
        print(f"  求解时间: Hour={hour}:{minute:02d}")
        
        # 遍历所有PV系统
        pv_total = 0
        pv_count = 0
        
        dss.Circuit.SetActiveClass("PVSystem")
        elem = dss.ActiveClass.First()
        while elem > 0:
            pv_name = dss.CktElement.Name()
            # 获取PV功率（注意：OpenDSS返回的是负值表示注入）
            powers = dss.CktElement.Powers()  # 返回[P1, Q1, P2, Q2, ...]
            if powers:
                p_kw = -powers[0]  # 取第一相的有功功率，转为正值
                pv_total += p_kw
                pv_count += 1
                print(f"  {pv_name}: P = {p_kw:.2f} kW")
            elem = dss.ActiveClass.Next()
        
        if pv_count > 0:
            print(f"  总PV出力: {pv_total:.2f} kW ({pv_count}个PV系统)")
        else:
            print("  未找到PV系统")
    
    # 7. 统计电压情况
    print("\n" + "="*50)
    print("步骤6: 电压统计...")
    
    # 获取所有节点电压
    all_voltages = []
    bus_names = dss.Circuit.AllBusNames
    
    for bus in bus_names:
        dss.Circuit.SetActiveBus(bus)
        voltages = dss.Bus.puVmagAngle[::2]  # 获取幅值（跳过角度）
        all_voltages.extend([v for v in voltages if v > 0])  # 排除无效值
    
    if all_voltages:
        min_v = min(all_voltages)
        max_v = max(all_voltages)
        avg_v = sum(all_voltages) / len(all_voltages)
        
        # 统计越限
        violations = [v for v in all_voltages if v < 0.95 or v > 1.05]
        violation_rate = len(violations) / len(all_voltages) * 100
        
        print(f"  最小电压: {min_v:.4f} p.u.")
        print(f"  最大电压: {max_v:.4f} p.u.")
        print(f"  平均电压: {avg_v:.4f} p.u.")
        print(f"  电压越限率: {violation_rate:.1f}% ({len(violations)}/{len(all_voltages)})")
    
    # 7. 验收判断
    print("\n" + "="*50)
    print("验收结果:")
    
    if pv_total > 1.0:  # 如果总PV出力大于1kW
        print("✓ PV注入成功！白天时段PV出力正常")
    else:
        print("✗ PV注入可能有问题，出力过低")
    
    if violation_rate < 30:
        print("✓ 电压违规率在可接受范围内")
    else:
        print("⚠ 电压违规率较高，需要进一步调整")
    
    print("\n测试完成！")
    
    # 清理临时文件
    if config_gen:
        config_gen.cleanup_generated_dss_files(keep_base_files=False)
    
    return pv_total, violation_rate


if __name__ == "__main__":
    # 测试不同系统
    systems = [
        "34Bus_PV_Aggressive",
        "34Bus_PV_Optimized",
        "34Bus_PV_Conservative"
    ]
    
    for system in systems:
        try:
            pv_power, viol_rate = test_pv_injection(system, worker_id=0, episode_idx=1)
            print(f"\n系统 {system} 测试结果: PV={pv_power:.1f}kW, 违规率={viol_rate:.1f}%")
        except Exception as e:
            print(f"\n系统 {system} 测试失败: {e}")