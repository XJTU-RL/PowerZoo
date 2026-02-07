#!/bin/bash
# @File    : run_district_dispatch.sh
# @Time    : 2025/01/28
# @Author  : Xiaodong Zheng
# @Description: 区域调度环境训练启动脚本
# 文件路径: examples/multi_agent/launchers/run_district_dispatch.sh

# ====================
# 默认参数
# ====================
ALGORITHM="happo"
EXPERIMENT_NAME="district_dispatch"
SEED=42
NUM_ENV_STEPS=5000000
EPISODE_LENGTH=96
N_ROLLOUT_THREADS=1
HIDDEN_SIZE=128
LEARNING_RATE=3e-4
USE_EVAL=true
USE_WANDB=false
SAVE_INTERVAL=25
EVAL_INTERVAL=25
GPU_ID=0

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 帮助信息
show_help() {
    echo -e "${BLUE}District Dispatch 训练脚本${NC}"
    echo "============================================================"
    echo "用法: $0 [选项]"
    echo ""
    echo "预设场景:"
    echo "  --quick         快速测试 (5万步, 用于验证环境可用性)"
    echo "  --standard      标准训练 (500万步, 默认配置)"
    echo "  --benchmark     基准测试 (2000万步, 完整评估)"
    echo ""
    echo "算法选择:"
    echo "  --algorithm     算法名称 [happo|mappo|hasac|maddpg|...] (默认: $ALGORITHM)"
    echo ""
    echo "训练参数:"
    echo "  --seed          随机种子 (默认: $SEED)"
    echo "  --steps         训练步数 (默认: $NUM_ENV_STEPS)"
    echo "  --episode-len   回合长度 (默认: $EPISODE_LENGTH)"
    echo "  --hidden        隐藏层大小 (默认: $HIDDEN_SIZE)"
    echo "  --lr            学习率 (默认: $LEARNING_RATE)"
    echo "  --gpu           GPU ID (默认: $GPU_ID)"
    echo ""
    echo "评估和日志:"
    echo "  --eval          启用评估 (默认: 开启)"
    echo "  --no-eval       禁用评估"
    echo "  --wandb         启用 wandb 日志"
    echo "  --save-interval 保存间隔 (默认: $SAVE_INTERVAL)"
    echo "  --eval-interval 评估间隔 (默认: $EVAL_INTERVAL)"
    echo ""
    echo "其他:"
    echo "  --experiment    实验名称 (默认: $EXPERIMENT_NAME)"
    echo "  --dry-run       仅显示命令，不执行"
    echo "  --help, -h      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --quick                          # 快速验证"
    echo "  $0 --algorithm mappo --seed 1       # MAPPO 训练"
    echo "  $0 --benchmark --wandb              # 基准测试 + wandb"
    echo "  $0 --algorithm hasac --steps 1000000  # HASAC 短训练"
}

# 预设场景
set_quick_preset() {
    NUM_ENV_STEPS=50000
    SAVE_INTERVAL=5
    EVAL_INTERVAL=5
    echo -e "${YELLOW}[Quick] 5万步快速验证${NC}"
}

set_standard_preset() {
    NUM_ENV_STEPS=5000000
    SAVE_INTERVAL=25
    EVAL_INTERVAL=25
    echo -e "${GREEN}[Standard] 500万步标准训练${NC}"
}

set_benchmark_preset() {
    NUM_ENV_STEPS=20000000
    USE_EVAL=true
    SAVE_INTERVAL=50
    EVAL_INTERVAL=50
    echo -e "${BLUE}[Benchmark] 2000万步基准测试${NC}"
}

# 解析命令行参数
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --quick)
            set_quick_preset
            shift
            ;;
        --standard)
            set_standard_preset
            shift
            ;;
        --benchmark)
            set_benchmark_preset
            shift
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
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
        --hidden)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --eval)
            USE_EVAL=true
            shift
            ;;
        --no-eval)
            USE_EVAL=false
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
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/${ALGORITHM}_district_dispatch_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# 构建训练命令
CMD="python examples/multi_agent/scripts/train.py"
CMD+=" --algo $ALGORITHM"
CMD+=" --env district_dispatch"
CMD+=" --exp_name ${EXPERIMENT_NAME}_${TIMESTAMP}"
CMD+=" --seed $SEED"
CMD+=" --episode_length $EPISODE_LENGTH"
CMD+=" --num_env_steps $NUM_ENV_STEPS"
CMD+=" --n_rollout_threads $N_ROLLOUT_THREADS"
CMD+=" --hidden_size $HIDDEN_SIZE"
CMD+=" --lr $LEARNING_RATE"
CMD+=" --save_interval $SAVE_INTERVAL"
CMD+=" --eval_interval $EVAL_INTERVAL"
# HAPPO默认 data_chunk_length=60 不能整除 episode_length=96，需显式覆盖
CMD+=" --data_chunk_length 48"
# n_data_chunks = episode_length/data_chunk_length * n_rollout_threads = 2
# mini_batch 不能超过 n_data_chunks
CMD+=" --actor_num_mini_batch 1"
CMD+=" --critic_num_mini_batch 1"

if [[ "$USE_EVAL" == "true" ]]; then
    CMD+=" --use_eval"
fi

if [[ "$USE_WANDB" == "true" ]]; then
    CMD+=" --use_wandb"
fi

# 显示配置
echo ""
echo -e "${BLUE}District Dispatch Training Configuration${NC}"
echo "============================================================"
echo -e "Algorithm:     ${GREEN}$ALGORITHM${NC}"
echo -e "Environment:   ${GREEN}district_dispatch (34Bus 3-Zone)${NC}"
echo -e "Experiment:    ${GREEN}${EXPERIMENT_NAME}_${TIMESTAMP}${NC}"
echo -e "Seed:          ${GREEN}$SEED${NC}"
echo -e "Steps:         ${GREEN}$NUM_ENV_STEPS${NC}"
echo -e "Episode Len:   ${GREEN}$EPISODE_LENGTH${NC}"
echo -e "Hidden Size:   ${GREEN}$HIDDEN_SIZE${NC}"
echo -e "Learning Rate: ${GREEN}$LEARNING_RATE${NC}"
echo -e "GPU:           ${GREEN}$GPU_ID${NC}"
echo -e "Eval:          ${GREEN}$USE_EVAL${NC}"
echo -e "W&B:           ${GREEN}$USE_WANDB${NC}"
echo -e "Result Dir:    ${GREEN}$RESULT_DIR${NC}"
echo "============================================================"
echo ""

# 执行
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN] Command:${NC}"
    echo "$CMD"
else
    echo -e "${GREEN}Starting training...${NC}"
    echo ""
    eval $CMD 2>&1 | tee "$RESULT_DIR/training.log"

    if [[ $? -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}Training completed!${NC}"
        echo -e "Results saved to: ${BLUE}$RESULT_DIR${NC}"
    else
        echo ""
        echo -e "${RED}Training failed! Check $RESULT_DIR/training.log${NC}"
        exit 1
    fi
fi
