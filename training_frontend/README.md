# PowerZoo Training Frontend

用于配置和管理PowerZoo多智能体强化学习训练任务的Web界面。

## 功能特性

### 算法支持
- **15个多智能体算法**: HAPPO, HATRPO, HAA2C, MAPPO, SHOM, SN-MAPPO, DAN-HAPPO, HADDPG, HATD3, HASAC, HAD3QN, MADDPG, MATD3, QMIX, 2TS-VVC
- **8个单智能体算法**: PPO, A2C, DDPG, TD3, SAC, DQN, HER, PPO_PV

### 环境支持
- PowerZoo VVC (电压无功控制)
- SmartGrid (智能微网)
- Stackelberg 13/34/123-Bus (博弈环境)
- DSR (配电网恢复)
- OpenAI Gym (标准环境)

### 配置管理
- 完整的训练参数配置
- 模型结构配置 (隐藏层、激活函数、RNN)
- 算法特定参数 (On-policy/Off-policy)
- 配置验证和错误提示
- 预设配置快速加载
- 配置导出/导入

### 任务管理
- 启动/停止训练任务
- 实时日志查看
- 任务状态监控
- 历史任务管理

## 快速开始

### 依赖环境
- Python 3.10+
- Node.js 18+
- npm 9+

### 安装和启动

```bash
# 进入前端目录
cd training_frontend

# 一键启动（推荐）
./start.sh

# 或分别启动
./start.sh backend   # 只启动后端 (http://localhost:8000)
./start.sh frontend  # 只启动前端 (http://localhost:3000)
```

### 手动启动

```bash
# 安装后端依赖
pip install -r backend/requirements.txt

# 安装前端依赖
npm install

# 启动后端
cd backend && python main.py

# 启动前端（新终端）
npm run dev
```

## 目录结构

```
training_frontend/
├── backend/              # FastAPI后端
│   ├── main.py          # API入口
│   └── requirements.txt # Python依赖
├── src/                  # React前端
│   ├── components/      # UI组件
│   │   └── config/      # 配置面板组件
│   ├── pages/           # 页面组件
│   ├── stores/          # Zustand状态管理
│   ├── services/        # API服务
│   ├── types/           # TypeScript类型
│   └── styles/          # 全局样式
├── public/              # 静态资源
├── package.json         # 前端依赖
├── vite.config.ts       # Vite配置
├── start.sh             # 启动脚本
└── README.md            # 说明文档
```

## API端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/algorithms` | 获取所有算法 |
| GET | `/api/environments` | 获取所有环境 |
| GET | `/api/presets` | 获取预设配置 |
| GET | `/api/config/default/{algo}/{env}` | 获取默认配置 |
| POST | `/api/config/validate` | 验证配置 |
| POST | `/api/training/start` | 启动训练 |
| GET | `/api/training/tasks` | 获取任务列表 |
| GET | `/api/training/tasks/{id}/logs` | 获取任务日志 |
| POST | `/api/training/tasks/{id}/stop` | 停止任务 |
| DELETE | `/api/training/tasks/{id}` | 删除任务 |

## 预设配置

| 预设 | 算法 | 环境 | 说明 |
|------|------|------|------|
| quick_test | HAPPO | smartgrid | 快速测试 (100K steps) |
| standard_training | HAPPO | smartgrid | 标准训练 (2M steps) |
| high_performance | MAPPO | smartgrid | 高性能 (5M steps) |
| stackelberg_game | SN-MAPPO | stackelberg_34bus | Stackelberg博弈 |
| dsr_restoration | HAPPO | dsr | 配电网恢复 |
| off_policy_training | HASAC | powerzoo | Off-Policy训练 |

## 技术栈

- **前端**: React 18 + TypeScript + Ant Design 5 + Zustand
- **后端**: FastAPI + Pydantic v2
- **构建**: Vite 5

## 许可证

MIT License
