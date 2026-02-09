#!/bin/bash
#
# PowerZoo Training Frontend - 启动脚本
#
# 使用方法:
#   ./start.sh          # 同时启动前端和后端
#   ./start.sh backend  # 只启动后端
#   ./start.sh frontend # 只启动前端
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  PowerZoo Training Frontend${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# 检查Python环境
check_python() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}Python: $(python --version)${NC}"
}

# 检查Node环境
check_node() {
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: Node.js not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}Node.js: $(node --version)${NC}"
}

# 安装后端依赖
install_backend() {
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip install -r backend/requirements.txt -q
}

# 安装前端依赖
install_frontend() {
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        npm install
    fi
}

# 启动后端
start_backend() {
    echo -e "${GREEN}Starting backend on http://localhost:8000${NC}"
    cd backend && python main.py &
    BACKEND_PID=$!
    cd ..
}

# 启动前端
start_frontend() {
    echo -e "${GREEN}Starting frontend on http://localhost:3000${NC}"
    npm run dev &
    FRONTEND_PID=$!
}

# 清理函数
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# 主逻辑
case "${1:-all}" in
    backend)
        check_python
        install_backend
        start_backend
        echo ""
        echo -e "${GREEN}Backend is running. Press Ctrl+C to stop.${NC}"
        wait $BACKEND_PID
        ;;
    frontend)
        check_node
        install_frontend
        start_frontend
        echo ""
        echo -e "${GREEN}Frontend is running. Press Ctrl+C to stop.${NC}"
        wait $FRONTEND_PID
        ;;
    all|*)
        check_python
        check_node
        install_backend
        install_frontend
        echo ""
        start_backend
        sleep 2
        start_frontend
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}  Services are running:${NC}"
        echo -e "${GREEN}  - Frontend: http://localhost:3000${NC}"
        echo -e "${GREEN}  - Backend:  http://localhost:8000${NC}"
        echo -e "${GREEN}======================================${NC}"
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop all services.${NC}"
        wait
        ;;
esac
