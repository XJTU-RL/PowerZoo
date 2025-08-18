#!/bin/bash

# 单智能体强化学习训练启动脚本
# 提供多种预设配置和快速启动选项

set -e  # 遇到错误时退出

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 默认参数
ALGO="ppo"
ENV="powerzoo_single"
CIRCUIT="13Bus"
TOTAL_TIMESTEPS=100000
SEED=42
DEVICE="auto"
EXP_NAME=""
VERBOSE=1

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印帮助信息
show_help() {
    echo -e "${BLUE}单智能体强化学习训练启动脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -a, --algo ALGO           算法 (ppo, dqn, sac, a2c, ddpg, td3, her) [默认: ppo]"
    echo "  -e, --env ENV             环境 [默认: powerzoo_single]"
    echo "  -c, --circuit CIRCUIT     电路名称 (13Bus, 34Bus, 123Bus, 8500Node) [默认: 13Bus]"
    echo "  -t, --timesteps STEPS     总训练步数 [默认: 100000]"
    echo "  -s, --seed SEED           随机种子 [默认: 42]"
    echo "  -d, --device DEVICE       计算设备 (auto, cpu, cuda) [默认: auto]"
    echo "  -n, --name NAME           实验名称 [默认: 自动生成]"
    echo "  -v, --verbose LEVEL       详细程度 (0, 1, 2) [默认: 1]"
    echo "  --eval-only               仅评估模式"
    echo "  --model-path PATH         预训练模型路径"
    echo "  --config-path PATH        配置文件路径"
    echo "  --log-dir DIR             日志目录 [默认: ./logs]"
    echo "  --model-dir DIR           模型保存目录 [默认: ./models]"
    echo "  --normalize               标准化环境"
    echo "  --no-tensorboard          禁用TensorBoard"
    echo "  --quick                   快速训练 (10000步)"
    echo "  --long                    长时间训练 (500000步)"
    echo "  --debug                   调试模式 (1000步)"
    echo "  -h, --help                显示此帮助信息"
    echo ""
    echo "预设配置:"
    echo "  --preset-ppo-13bus        PPO + 13Bus 标准训练"
    echo "  --preset-dqn-34bus        DQN + 34Bus 训练"
    echo "  --preset-sac-123bus       SAC + 123Bus 训练"
    echo "  --preset-debug            调试配置 (快速训练)"
    echo ""
    echo "示例:"
    echo "  $0 --algo ppo --circuit 13Bus --timesteps 50000"
    echo "  $0 --preset-ppo-13bus --seed 123"
    echo "  $0 --quick --algo dqn"
    echo "  $0 --eval-only --model-path ./models/best_model.zip"
}

# 打印状态信息
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# 打印警告信息
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 打印错误信息
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python_env() {
    if ! command -v python &> /dev/null; then
        print_error "Python未找到，请确保Python已安装"
        exit 1
    fi
    
    # 检查必要的包
    python -c "import stable_baselines3" 2>/dev/null || {
        print_error "stable_baselines3未安装，请运行: pip install stable_baselines3"
        exit 1
    }
    
    print_status "Python环境检查通过"
}

# 检查项目结构
check_project_structure() {
    if [[ ! -f "examples/single_agent/scripts/train_single_agent.py" ]]; then
        print_error "未找到examples/single_agent/scripts/train_single_agent.py，请确保在正确的项目目录中"
        exit 1
    fi
    
    if [[ ! -d "configs" ]]; then
        print_error "未找到configs目录，请确保项目结构完整"
        exit 1
    fi
    
    print_status "项目结构检查通过"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--algo)
                ALGO="$2"
                shift 2
                ;;
            -e|--env)
                ENV="$2"
                shift 2
                ;;
            -c|--circuit)
                CIRCUIT="$2"
                shift 2
                ;;
            -t|--timesteps)
                TOTAL_TIMESTEPS="$2"
                shift 2
                ;;
            -s|--seed)
                SEED="$2"
                shift 2
                ;;
            -d|--device)
                DEVICE="$2"
                shift 2
                ;;
            -n|--name)
                EXP_NAME="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="$2"
                shift 2
                ;;
            --eval-only)
                EVAL_ONLY="--eval_only"
                shift
                ;;
            --model-path)
                MODEL_PATH="--model_path $2"
                shift 2
                ;;
            --config-path)
                CONFIG_PATH="--config_path $2"
                shift 2
                ;;
            --log-dir)
                LOG_DIR="--log_dir $2"
                shift 2
                ;;
            --model-dir)
                MODEL_DIR="--model_save_dir $2"
                shift 2
                ;;
            --normalize)
                NORMALIZE="--normalize_env"
                shift
                ;;
            --no-tensorboard)
                NO_TENSORBOARD="--no_tensorboard"
                shift
                ;;
            --quick)
                TOTAL_TIMESTEPS=10000
                print_status "启用快速训练模式 (10000步)"
                shift
                ;;
            --long)
                TOTAL_TIMESTEPS=500000
                print_status "启用长时间训练模式 (500000步)"
                shift
                ;;
            --debug)
                TOTAL_TIMESTEPS=1000
                VERBOSE=2
                print_status "启用调试模式 (1000步, 详细输出)"
                shift
                ;;
            --preset-ppo-13bus)
                ALGO="ppo"
                CIRCUIT="13Bus"
                TOTAL_TIMESTEPS=100000
                print_status "使用PPO + 13Bus预设配置"
                shift
                ;;
            --preset-dqn-34bus)
                ALGO="dqn"
                CIRCUIT="34Bus"
                TOTAL_TIMESTEPS=150000
                print_status "使用DQN + 34Bus预设配置"
                shift
                ;;
            --preset-sac-123bus)
                ALGO="sac"
                CIRCUIT="123Bus"
                TOTAL_TIMESTEPS=200000
                print_status "使用SAC + 123Bus预设配置"
                shift
                ;;
            --preset-debug)
                ALGO="ppo"
                CIRCUIT="13Bus"
                TOTAL_TIMESTEPS=1000
                VERBOSE=2
                print_status "使用调试预设配置"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 构建训练命令
build_command() {
    CMD="python examples/single_agent/scripts/train_single_agent.py"
    CMD="$CMD --algo $ALGO"
    CMD="$CMD --env $ENV"
    CMD="$CMD --circuit_name $CIRCUIT"
    CMD="$CMD --total_timesteps $TOTAL_TIMESTEPS"
    CMD="$CMD --seed $SEED"
    CMD="$CMD --device $DEVICE"
    CMD="$CMD --verbose $VERBOSE"
    
    if [[ -n "$EXP_NAME" ]]; then
        CMD="$CMD --exp_name $EXP_NAME"
    fi
    
    if [[ -n "$EVAL_ONLY" ]]; then
        CMD="$CMD $EVAL_ONLY"
    fi
    
    if [[ -n "$MODEL_PATH" ]]; then
        CMD="$CMD $MODEL_PATH"
    fi
    
    if [[ -n "$CONFIG_PATH" ]]; then
        CMD="$CMD $CONFIG_PATH"
    fi
    
    if [[ -n "$LOG_DIR" ]]; then
        CMD="$CMD $LOG_DIR"
    fi
    
    if [[ -n "$MODEL_DIR" ]]; then
        CMD="$CMD $MODEL_DIR"
    fi
    
    if [[ -n "$NORMALIZE" ]]; then
        CMD="$CMD $NORMALIZE"
    fi
    
    if [[ -n "$NO_TENSORBOARD" ]]; then
        CMD="$CMD $NO_TENSORBOARD"
    fi
    
    echo "$CMD"
}

# 打印训练配置
print_config() {
    echo -e "${BLUE}=== 训练配置 ===${NC}"
    echo "算法: $ALGO"
    echo "环境: $ENV"
    echo "电路: $CIRCUIT"
    echo "训练步数: $TOTAL_TIMESTEPS"
    echo "随机种子: $SEED"
    echo "计算设备: $DEVICE"
    echo "详细程度: $VERBOSE"
    if [[ -n "$EXP_NAME" ]]; then
        echo "实验名称: $EXP_NAME"
    fi
    echo -e "${BLUE}==================${NC}"
    echo ""
}

# 主函数
main() {
    echo -e "${BLUE}PowerZoo 单智能体强化学习训练${NC}"
    echo ""
    
    # 解析参数
    parse_args "$@"
    
    # 检查环境
    check_python_env
    check_project_structure
    
    # 切换到项目根目录
    cd "$(dirname "$SCRIPT_DIR")"
    
    # 打印配置
    print_config
    
    # 构建命令
    TRAIN_CMD=$(build_command)
    
    # 打印命令
    print_status "执行命令: $TRAIN_CMD"
    echo ""
    
    # 执行训练
    eval "$TRAIN_CMD"
    
    # 检查执行结果
    if [[ $? -eq 0 ]]; then
        print_status "训练完成！"
    else
        print_error "训练失败！"
        exit 1
    fi
}

# 运行主函数
main "$@"