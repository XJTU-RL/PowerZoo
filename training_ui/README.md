# PowerZoo 训练工作台

从仓库根目录安装可选界面依赖并启动：

```bash
python -m pip install -e ".[viz]"
python -m training_ui
```

浏览器打开 `http://127.0.0.1:7860`。默认仅监听本机，可用 `--port` 更换端口。

## 检查训练环境

在「环境检查」页点击「运行环境检查」。检查使用训练子进程所用的 Python 和项目目录，报告解释器、关键依赖的导入结果、PyTorch 设备及 OpenDSS 原生求解结果。检查不会启动训练，也不会安装依赖。

如果报告缺少 `gymnasium`、`gym` 等依赖，先确认报告里的 Python 路径。可以切换到已安装训练依赖的环境，也可以在计划用于训练的环境中，从仓库根目录执行 `python -m pip install -e ".[viz]"`。切换解释器后，用该环境重新运行 `python -m training_ui`。

## 配置、启动与查看产物

1. 选择环境、算法和实验名称。左侧 YAML 只填写需要覆盖的配置，右侧预览完整生效配置。
2. 点击「校验并预览」，也可按完整路径查询参数来源。点击「保存配置并训练」后，工作台将配置快照保存到 `training_ui/.cache/`，并调用公共入口 `scripts/train.py`。
3. 在「训练监控」查看当前服务会话中的任务状态、配置路径与日志。在「模型管理」扫描默认 `results/` 目录，选择实验查看配置和检查点。

首次验证可选择 VVC、`happo` 和环境预设 `vvc`，将左侧 YAML 改为：

```yaml
system_ref: 13Bus
training_ref: smoke
train:
  num_env_steps: 24
  eval_interval: 1
```

该配置在 CPU 上运行一个 24 步采样回合，并保存一次更新后的模型。训练使用仓库的 IEEE13 电路和可复现的合成运行场景，适合检查链路。

参数含义、优先级和适用范围见[配置指南](../docs/configuration.md)及[参数流代码审计](../docs/configuration-audit.md)。
