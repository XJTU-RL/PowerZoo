#!/bin/bash

# @File    : run_dsr_aggregation.sh
# @Time    : 2025/01/27
# @Author  : Xiaodong Zheng
# @Description: 灵活的DSR聚合训练运行脚本
# 文件路径: examples/run_dsr_aggregation.sh

# 脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# 设置默认参数
SYSTEM_NAME="123Bus"
ALGORITHM="mappo"
EXPERIMENT_NAME="dsr_aggregation"
USE_AGGREGATION=false
N_LOAD_AGENTS=""
AGGREGATION_METHOD="zone"
SEED=1
EPISODE_LENGTH=15
NUM_ENV_STEPS=10000000
N_ROLLOUT_THREADS=3
HIDDEN_SIZE=128
LEARNING_RATE=5e-4
USE_EVAL=false
USE_WANDB=false
SAVE_INTERVAL=25
EVAL_INTERVAL=25

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo -e "${BLUE}DSR聚合训练脚本使用说明${NC}"
    echo "="*60
    echo "用法: $0 [选项]"
    echo ""
    echo "预设场景:"
    echo "  --small         小系统训练 (13Bus, 无聚合)"
    echo "  --medium        中等系统训练 (123Bus, 可选聚合)"
    echo "  --large         大系统训练 (8500-Node, 强制聚合)"
    echo "  --quick         快速测试 (少量步数)"
    echo "  --benchmark     基准测试 (完整训练)"
    echo ""
    echo "系统参数:"
    echo "  --system        电力系统 [13Bus|34Bus|123Bus|8500-Node] (默认: $SYSTEM_NAME)"
    echo "  --algorithm     算法 [mappo|maddpg|qmix|...] (默认: $ALGORITHM)"
    echo "  --experiment    实验名称 (默认: $EXPERIMENT_NAME)"
    echo ""
    echo "聚合参数:"
    echo "  --aggregation   启用负荷聚合"
    echo "  --no-aggregation 禁用负荷聚合"
    echo "  --agents        负荷智能体数量 (默认: 自动)"
    echo "  --method        聚合方法 [zone|priority|random] (默认: $AGGREGATION_METHOD)"
    echo ""
    echo "训练参数:"
    echo "  --seed          随机种子 (默认: $SEED)"
    echo "  --steps         训练步数 (默认: $NUM_ENV_STEPS)"
    echo "  --episode-len   回合长度 (默认: $EPISODE_LENGTH)"
    echo "  --threads       并行线程数 (默认: $N_ROLLOUT_THREADS)"
    echo "  --hidden        隐藏层大小 (默认: $HIDDEN_SIZE)"
    echo "  --lr            学习率 (默认: $LEARNING_RATE)"
    echo ""
    echo "评估和日志:"
    echo "  --eval          启用评估"
    echo "  --wandb         启用wandb日志"
    echo "  --save-interval 保存间隔 (默认: $SAVE_INTERVAL)"
    echo "  --eval-interval 评估间隔 (默认: $EVAL_INTERVAL)"
    echo ""
    echo "其他:"
    echo "  --help, -h      显示此帮助信息"
    echo "  --dry-run       仅显示命令，不执行"
    echo "  --verbose       详细输出"
    echo ""
    echo "示例:"
    echo "  $0 --small --eval                    # 小系统训练+评估"
    echo "  $0 --large --agents 30 --wandb       # 大系统30个智能体+wandb"
    echo "  $0 --system 123Bus --aggregation --method priority  # 自定义配置"
    echo "  $0 --quick --algorithm maddpg        # 快速测试MADDPG"
}

# 预设场景函数
set_small_preset() {
    SYSTEM_NAME="13Bus"
    USE_AGGREGATION=false
    N_ROLLOUT_THREADS=2
    NUM_ENV_STEPS=1000000
    echo -e "${GREEN}使用小系统预设: 13Bus, 无聚合${NC}"
}

set_medium_preset() {
    SYSTEM_NAME="123Bus"
    USE_AGGREGATION=true
    N_LOAD_AGENTS=10
    N_ROLLOUT_THREADS=3
    NUM_ENV_STEPS=5000000
    echo -e "${GREEN}使用中等系统预设: 123Bus, 10个负荷智能体${NC}"
}

set_large_preset() {
    SYSTEM_NAME="8500-Node"
    USE_AGGREGATION=true
    N_LOAD_AGENTS=50
    N_ROLLOUT_THREADS=4
    NUM_ENV_STEPS=20000000
    HIDDEN_SIZE=256
    echo -e "${GREEN}使用大系统预设: 8500-Node, 50个负荷智能体${NC}"
}

set_quick_preset() {
    NUM_ENV_STEPS=100000
    EPISODE_LENGTH=10
    SAVE_INTERVAL=10
    EVAL_INTERVAL=10
    echo -e "${YELLOW}使用快速测试预设: 10万步${NC}"
}

set_benchmark_preset() {
    NUM_ENV_STEPS=50000000
    USE_EVAL=true
    SAVE_INTERVAL=50
    EVAL_INTERVAL=50
    echo -e "${BLUE}使用基准测试预设: 5000万步+评估${NC}"
}

# 解析命令行参数
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --small)
            set_small_preset
            shift
            ;;
        --medium)
            set_medium_preset
            shift
            ;;
        --large)
            set_large_preset
            shift
            ;;
        --quick)
            set_quick_preset
            shift
            ;;
        --benchmark)
            set_benchmark_preset
            shift
            ;;
        --system)
            SYSTEM_NAME="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --aggregation)
            USE_AGGREGATION=true
            shift
            ;;
        --no-aggregation)
            USE_AGGREGATION=false
            shift
            ;;
        --agents)
            N_LOAD_AGENTS="$2"
            shift 2
            ;;
        --method)
            AGGREGATION_METHOD="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --steps)
            NUM_ENV_STEPS="$2"
            shift 2
            ;;
        --episode-len)
            EPISODE_LENGTH="$2"
            shift 2
            ;;
        --threads)
            N_ROLLOUT_THREADS="$2"
            shift 2
            ;;
        --hidden)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --eval)
            USE_EVAL=true
            shift
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 验证参数
if [[ ! "$SYSTEM_NAME" =~ ^(13Bus|34Bus|123Bus|8500-Node)$ ]]; then
    echo -e "${RED}错误: 无效的系统名称 '$SYSTEM_NAME'${NC}"
    echo "支持的系统: 13Bus, 34Bus, 123Bus, 8500-Node"
    exit 1
fi

if [[ ! "$AGGREGATION_METHOD" =~ ^(zone|priority|random)$ ]]; then
    echo -e "${RED}错误: 无效的聚合方法 '$AGGREGATION_METHOD'${NC}"
    echo "支持的方法: zone, priority, random"
    exit 1
fi

# 构建命令
CMD="python examples/multi_agent/scripts/train_dsr_aggregation.py"
CMD+=" --system_name $SYSTEM_NAME"
CMD+=" --algorithm_name $ALGORITHM"
CMD+=" --experiment_name $EXPERIMENT_NAME"
CMD+=" --seed $SEED"
CMD+=" --episode_length $EPISODE_LENGTH"
CMD+=" --num_env_steps $NUM_ENV_STEPS"
CMD+=" --n_rollout_threads $N_ROLLOUT_THREADS"
CMD+=" --hidden_size $HIDDEN_SIZE"
CMD+=" --lr $LEARNING_RATE"
CMD+=" --save_interval $SAVE_INTERVAL"
CMD+=" --eval_interval $EVAL_INTERVAL"
CMD+=" --load_aggregation_method $AGGREGATION_METHOD"

if [[ "$USE_AGGREGATION" == "true" ]]; then
    CMD+=" --use_load_aggregation"
fi

if [[ -n "$N_LOAD_AGENTS" ]]; then
    CMD+=" --n_load_agents $N_LOAD_AGENTS"
fi

if [[ "$USE_EVAL" == "true" ]]; then
    CMD+=" --use_eval"
fi

if [[ "$USE_WANDB" == "true" ]]; then
    CMD+=" --use_wandb"
fi

# 显示配置信息
echo -e "${BLUE}DSR聚合训练配置${NC}"
echo "="*60
echo -e "系统: ${GREEN}$SYSTEM_NAME${NC}"
echo -e "算法: ${GREEN}$ALGORITHM${NC}"
echo -e "实验名称: ${GREEN}$EXPERIMENT_NAME${NC}"
echo -e "使用聚合: ${GREEN}$USE_AGGREGATION${NC}"
if [[ "$USE_AGGREGATION" == "true" ]]; then
    echo -e "聚合方法: ${GREEN}$AGGREGATION_METHOD${NC}"
    if [[ -n "$N_LOAD_AGENTS" ]]; then
        echo -e "负荷智能体数: ${GREEN}$N_LOAD_AGENTS${NC}"
    else
        echo -e "负荷智能体数: ${YELLOW}自动${NC}"
    fi
fi
echo -e "随机种子: ${GREEN}$SEED${NC}"
echo -e "训练步数: ${GREEN}$NUM_ENV_STEPS${NC}"
echo -e "回合长度: ${GREEN}$EPISODE_LENGTH${NC}"
echo -e "并行线程: ${GREEN}$N_ROLLOUT_THREADS${NC}"
echo -e "隐藏层大小: ${GREEN}$HIDDEN_SIZE${NC}"
echo -e "学习率: ${GREEN}$LEARNING_RATE${NC}"
echo -e "使用评估: ${GREEN}$USE_EVAL${NC}"
echo -e "使用wandb: ${GREEN}$USE_WANDB${NC}"
echo "="*60

if [[ "$VERBOSE" == "true" ]]; then
    echo -e "${YELLOW}完整命令:${NC}"
    echo "$CMD"
    echo ""
fi

# 执行命令
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN] 将执行以下命令:${NC}"
    echo "$CMD"
else
    echo -e "${GREEN}开始训练...${NC}"
    echo ""
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 执行训练
    eval $CMD
    
    # 检查执行结果
    if [[ $? -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}训练完成!${NC}"
    else
        echo ""
        echo -e "${RED}训练失败!${NC}"
        exit 1
    fi
fi