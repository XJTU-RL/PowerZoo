"""公共内核入口。"""
from .runtime_env import Env
from .core_env import ActionSpace  # 兼容旧导入；正式运行使用内核的 agent_spaces
