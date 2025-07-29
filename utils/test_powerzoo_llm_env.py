#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PowerZoo LLM环境测试脚本
验证环境是否能正确初始化和运行

@Author: Xiaodong Zheng
"""

import os
import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def test_environment_import():
	"""测试环境模块导入"""
	print("🔍 测试环境模块导入...")
	
	try:
		from envs.power_envs.powerzoo_llm.env_register import make_base_env
		print("✅ env_register导入成功")
		
		from envs.power_envs.powerzoo_llm.env import Env
		print("✅ env模块导入成功")
		
		from envs.power_envs.powerzoo_llm.powerzoo_env import PowerZooEnv
		print("✅ powerzoo_env导入成功")
		
		return True
	except Exception as e:
		print(f"❌ 模块导入失败: {e}")
		traceback.print_exc()
		return False

def test_basic_env_creation():
	"""测试基础环境创建"""
	print("\n🔍 测试基础环境创建...")
	
	try:
		from envs.power_envs.powerzoo_llm.env_register import make_base_env
		
		# 测试不带worker_idx的环境创建
		print("  测试无worker_idx环境...")
		env = make_base_env('34Bus_pv')
		print(f"✅ 基础环境创建成功: {type(env)}")
		
		# 简单测试环境属性
		if hasattr(env, 'action_space'):
			print(f"  动作空间: {env.action_space}")
		if hasattr(env, 'observation_space'):
			print(f"  观察空间类型: {type(env.observation_space)}")
		
		return True
	except Exception as e:
		print(f"❌ 基础环境创建失败: {e}")
		traceback.print_exc()
		return False

def test_worker_env_creation():
	"""测试worker环境创建"""
	print("\n🔍 测试worker环境创建...")
	
	try:
		from envs.power_envs.powerzoo_llm.env_register import make_base_env
		
		# 测试带worker_idx的环境创建
		for worker_idx in [0, 1, 5]:
			print(f"  测试worker {worker_idx}...")
			env = make_base_env('34Bus_pv', worker_idx=worker_idx)
			print(f"✅ Worker {worker_idx} 环境创建成功")
		
		return True
	except Exception as e:
		print(f"❌ Worker环境创建失败: {e}")
		traceback.print_exc()
		return False

def test_powerzoo_env_wrapper():
	"""测试PowerZooEnv包装器"""
	print("\n🔍 测试PowerZooEnv包装器...")
	
	try:
		from envs.power_envs.powerzoo_llm.env_register import make_base_env
		from envs.power_envs.powerzoo_llm.powerzoo_env import PowerZooEnv
		
		# 创建基础环境
		base_env = make_base_env('34Bus_pv', worker_idx=0)
		
		# 创建包装器环境
		env_args = {
			'env_name': '34Bus_pv',
			'observe_actions': True,
			'act_limit': False
		}
		
		wrapped_env = PowerZooEnv(base_env, env_args, rank=0)
		print(f"✅ PowerZooEnv包装器创建成功")
		
		# 测试环境属性
		print(f"  智能体数量: {wrapped_env.n_agents}")
		print(f"  动作空间: {wrapped_env.action_space}")
		print(f"  观察空间: {wrapped_env.observation_space}")
		
		return True
	except Exception as e:
		print(f"❌ PowerZooEnv包装器失败: {e}")
		traceback.print_exc()
		return False

def test_environment_reset():
	"""测试环境重置"""
	print("\n🔍 测试环境重置...")
	
	try:
		from envs.power_envs.powerzoo_llm.env_register import make_base_env
		from envs.power_envs.powerzoo_llm.powerzoo_env import PowerZooEnv
		
		# 创建环境
		base_env = make_base_env('34Bus_pv', worker_idx=0)
		env_args = {
			'env_name': '34Bus_pv',
			'observe_actions': True,
			'act_limit': False
		}
		wrapped_env = PowerZooEnv(base_env, env_args, rank=0)
		
		# 测试reset
		print("  执行环境reset...")
		obs = wrapped_env.reset()
		print(f"✅ 环境reset成功")
		print(f"  观察维度: {[o.shape if hasattr(o, 'shape') else len(o) for o in obs]}")
		
		return True
	except Exception as e:
		print(f"❌ 环境reset失败: {e}")
		traceback.print_exc()
		return False

def test_environment_step():
	"""测试环境step"""
	print("\n🔍 测试环境step...")
	
	try:
		from envs.power_envs.powerzoo_llm.env_register import make_base_env
		from envs.power_envs.powerzoo_llm.powerzoo_env import PowerZooEnv
		import numpy as np
		
		# 创建环境
		base_env = make_base_env('34Bus_pv', worker_idx=0)
		env_args = {
			'env_name': '34Bus_pv',
			'observe_actions': True,
			'act_limit': False
		}
		wrapped_env = PowerZooEnv(base_env, env_args, rank=0)
		
		# Reset环境
		obs = wrapped_env.reset()
		
		# 创建随机动作
		print("  生成随机动作...")
		actions = []
		for i in range(wrapped_env.n_agents):
			if hasattr(wrapped_env.action_space[i], 'n'):
				# 离散动作空间
				action = np.random.randint(wrapped_env.action_space[i].n)
			else:
				# 连续动作空间
				action = wrapped_env.action_space[i].sample()
			actions.append(action)
		
		print(f"  动作: {actions}")
		
		# 执行step
		print("  执行环境step...")
		obs, rewards, dones, infos = wrapped_env.step(actions)
		
		print(f"✅ 环境step成功")
		print(f"  奖励: {rewards}")
		print(f"  完成标志: {dones}")
		print(f"  信息条目: {len(infos)}")
		
		return True
	except Exception as e:
		print(f"❌ 环境step失败: {e}")
		traceback.print_exc()
		return False

def test_data_loading():
	"""测试数据文件加载"""
	print("\n🔍 测试数据文件加载...")
	
	try:
		base_dir = Path("/home/zhengxiaodong/exps/PowerZoo/envs/power_envs/powerzoo_llm/node_systems_with_pv/34Bus")
		
		# 检查pv_data.dss文件
		pv_data_file = base_dir / "pv_data.dss"
		if pv_data_file.exists():
			print(f"✅ pv_data.dss文件存在")
			
			with open(pv_data_file, 'r') as f:
				content = f.read()
				if '20250307_irradiance.csv' in content:
					print(f"✅ PV数据文件引用正确")
				else:
					print(f"⚠️  PV数据文件引用可能有问题")
		else:
			print(f"❌ pv_data.dss文件不存在")
		
		# 检查数据文件
		irrad_file = base_dir / "irradiation" / "000" / "20250307_irradiance.csv"
		temp_file = base_dir / "temperature" / "000" / "20250307_temperature.csv"
		
		if irrad_file.exists() and temp_file.exists():
			print(f"✅ 数据文件存在")
			
			# 检查文件内容
			with open(irrad_file, 'r') as f:
				first_line = f.readline().strip()
				try:
					float(first_line)
					print(f"✅ 辐照度文件格式正确（无表头）")
				except ValueError:
					print(f"⚠️  辐照度文件可能有表头: {first_line}")
			
			with open(temp_file, 'r') as f:
				first_line = f.readline().strip()
				try:
					float(first_line)
					print(f"✅ 温度文件格式正确（无表头）")
				except ValueError:
					print(f"⚠️  温度文件可能有表头: {first_line}")
		else:
			print(f"❌ 数据文件不存在")
		
		return True
	except Exception as e:
		print(f"❌ 数据文件检查失败: {e}")
		traceback.print_exc()
		return False

def main():
	"""主测试函数"""
	print("🧪 PowerZoo LLM环境测试")
	print("=" * 50)
	
	tests = [
		("模块导入", test_environment_import),
		("数据文件", test_data_loading),
		("基础环境创建", test_basic_env_creation),
		("Worker环境创建", test_worker_env_creation),
		("PowerZooEnv包装器", test_powerzoo_env_wrapper),
		("环境重置", test_environment_reset),
		("环境执行", test_environment_step),
	]
	
	results = []
	for test_name, test_func in tests:
		try:
			result = test_func()
			results.append((test_name, result))
		except Exception as e:
			print(f"❌ 测试 {test_name} 异常: {e}")
			results.append((test_name, False))
	
	# 汇总结果
	print("\n" + "=" * 50)
	print("🎯 测试结果汇总:")
	passed = 0
	for test_name, result in results:
		status = "✅ 通过" if result else "❌ 失败"
		print(f"  {test_name}: {status}")
		if result:
			passed += 1
	
	print(f"\n总体结果: {passed}/{len(results)} 测试通过")
	
	if passed == len(results):
		print("🎉 所有测试通过！环境配置正确。")
		return 0
	else:
		print("⚠️  存在测试失败，请检查配置。")
		return 1

if __name__ == "__main__":
	sys.exit(main())