#!/bin/bash

# 测试DDPG算法的脚本
# 使用连续动作空间和较少的训练步数进行快速测试

echo "开始测试DDPG算法..."
echo "使用连续动作空间，训练步数: 1000"
echo "环境: powerzoo_single, 电路: 13Bus"
echo "实验结果将保存到: /home/zhengxiaodong/exps/PowerZoo/results"
echo ""

# 运行DDPG训练
python examples/single_agent/scripts/train_single_agent.py \
    --algo ddpg \
    --env powerzoo_single \
    --circuit_name 13Bus \
    --total_timesteps 1000 \
    --action_space_type continuous \
    --exp_name "test_ddpg_continuous" \
    --verbose 2 \
    --log_level INFO

echo ""
echo "DDPG测试完成！"
echo "请检查 /home/zhengxiaodong/exps/PowerZoo/results 目录下的实验结果"
