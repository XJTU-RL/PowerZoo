# Hugging Face Spaces

所有 Space 的源文件统一存放在此目录，每个子目录对应一个独立应用。

- `powerzoo/`：英文主站，原 `huggingface_space/` 的相同内容已合并至此。
- `powerzoo-zh/`：中文主站。
- `powerzoo-{vvc,smartgrid,dsr,stackelberg,district-dispatch}/`：各环境英文应用。
- 对应 `-zh/` 子目录：各环境中文应用。

`.github/workflows/sync-hf-space.yml` 将 `powerzoo/` 同步到 `XJTU-RL/PowerZoo`。
两个 `tools/plot_algorithm_*.py` 图表生成脚本向 `powerzoo/data/` 导出主站数据。
各应用的启动入口及依赖见其自身的 `README.md` 和 `requirements.txt`。
