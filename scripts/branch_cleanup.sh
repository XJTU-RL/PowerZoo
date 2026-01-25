#!/bin/bash
# PowerZoo Branch Cleanup Script
# 用于执行分支清理和管理操作
# 使用前请仔细阅读 BRANCH_MANAGEMENT.md

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数：打印带颜色的消息
print_info() {
	echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
	echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
	echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
	echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：检查命令是否存在
check_command() {
	if ! command -v "$1" &> /dev/null; then
		print_error "命令 $1 未找到，请先安装"
		exit 1
	fi
}

# 检查必要的命令
check_command git
check_command gh

# 确保在git仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
	print_error "当前目录不是git仓库"
	exit 1
fi

print_info "开始分支清理流程..."
echo ""

# 1. 删除已合并的分支
print_info "步骤 1: 检查并删除已合并的分支"
MERGED_BRANCH="claude/training-config-frontend-wXO8f"

if git show-ref --verify --quiet refs/remotes/origin/$MERGED_BRANCH; then
	print_warning "发现已合并分支: $MERGED_BRANCH"
	read -p "是否删除此分支? (y/N): " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		git push origin --delete $MERGED_BRANCH
		print_success "已删除分支: $MERGED_BRANCH"
	else
		print_info "跳过删除 $MERGED_BRANCH"
	fi
else
	print_success "分支 $MERGED_BRANCH 已被删除或不存在"
fi

echo ""

# 2. 列出需要合并的Dependabot PR
print_info "步骤 2: 列出待处理的Dependabot PR"
echo ""
print_info "低风险PR（建议直接合并）:"
echo "  - PR #27: actions/checkout v4→v6"
echo "  - PR #26: actions/setup-python v5→v6"
echo "  - PR #25: codecov/codecov-action v4→v5"
echo "  - PR #30: numpy版本范围扩展"
echo "  - PR #29: torch版本范围扩展"
echo "  - PR #28: torchvision版本范围扩展"
echo ""
print_warning "需要测试的PR:"
echo "  - PR #31: gymnasium 0.29.1→1.1.1 (需要先测试)"
echo ""

read -p "是否批量合并低风险PR (27,26,25,30,29,28)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
	print_info "开始合并低风险PR..."
	
	# CI相关PR
	for pr in 27 26 25; do
		print_info "正在合并 PR #$pr..."
		if gh pr merge $pr --merge; then
			print_success "PR #$pr 已合并"
		else
			print_error "PR #$pr 合并失败"
		fi
	done
	
	# Python依赖PR
	for pr in 30 29 28; do
		print_info "正在合并 PR #$pr..."
		if gh pr merge $pr --merge; then
			print_success "PR #$pr 已合并"
		else
			print_error "PR #$pr 合并失败"
		fi
	done
	
	print_success "低风险PR合并完成"
else
	print_info "跳过批量合并"
fi

echo ""

# 3. 提醒测试gymnasium PR
print_warning "步骤 3: Gymnasium PR (#31) 需要手动测试"
echo ""
print_info "测试步骤："
echo "  1. git fetch origin"
echo "  2. git checkout dependabot/pip/gymnasium-1.1.1"
echo "  3. pip install --upgrade pip"
echo "  4. pip install -e ."
echo "  5. pytest tests/"
echo "  6. 如果测试通过: gh pr merge 31 --merge"
echo ""

read -p "是否现在测试 Gymnasium PR? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
	print_info "切换到 gymnasium 分支..."
	git fetch origin
	git checkout dependabot/pip/gymnasium-1.1.1
	
	print_info "升级pip并安装依赖..."
	pip install --upgrade pip
	pip install -e .
	
	print_info "运行测试..."
	if pytest tests/ -v; then
		print_success "测试通过！"
		read -p "是否合并 PR #31? (y/N): " -n 1 -r
		echo
		if [[ $REPLY =~ ^[Yy]$ ]]; then
			gh pr merge 31 --merge
			print_success "PR #31 已合并"
		fi
	else
		print_error "测试失败，请检查后再合并"
	fi
	
	# 返回主分支
	git checkout main
else
	print_info "跳过测试，请稍后手动测试"
fi

echo ""

# 4. 清理本地引用
print_info "步骤 4: 清理本地过期的远程分支引用"
git fetch --all --prune
print_success "清理完成"

echo ""

# 5. 显示当前分支状态
print_info "步骤 5: 当前分支状态"
echo ""
git branch -r --sort=-committerdate --format='%(refname:short) %(committerdate:short) %(authorname)' | head -15

echo ""
print_success "分支清理流程完成！"
print_info "请查看 BRANCH_MANAGEMENT.md 了解详细信息"
