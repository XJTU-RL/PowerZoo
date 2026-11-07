# PowerZoo 训练界面

当前环境配置页直接编辑 YAML，并通过公共 `ConfigLoader` 进行预览、校验和启动。
不再维护另一套按叶子名拼装参数的表单。环境页右侧展示完整生效参数，输入完整路径可查看来源。
切换环境预设会重新载入其差异配置；算法默认值由解析器读取，不复制到 UI schema。

安装可选界面依赖后，从当前 worktree 根目录启动：

```powershell
conda activate powerzoo-win
python -m pip install -e ".[viz]"
python -m training_ui
```

依赖可选安装不属于 Windows 核心运行基线。当前验证覆盖配置 builder 与 CLI 的最终参数一致性；
尚未在本机进行 Gradio 浏览器交互验收。安装说明以根目录 requirements-viz.txt 为准。

界面保存的是可重放的完整 YAML 和来源文件，训练子进程仍使用 examples/multi_agent/scripts/train.py。
来源字段不会混入环境参数。监控和模型浏览页面保留原实现，需要分别验证其研究任务日志格式。

具体参数、优先级、真实能力边界见 [配置指南](../docs/configuration.md) 与
[参数流代码审计](../docs/configuration-audit.md)。
