# Hugging Face Spaces

所有 Space 的源文件统一存放在此目录，每个子目录对应一个独立应用。

- `powerzoo/`：英文主站，原 `huggingface_space/` 的相同内容已合并至此。
- `powerzoo-zh/`：中文主站。
- `powerzoo-{vvc,smartgrid,dsr,stackelberg,district-dispatch}/`：各环境英文应用。
- 对应 `-zh/` 子目录：各环境中文应用。

`.github/workflows/sync-hf-space.yml` 先校验共享资源，再同步已配置发布目标的主站。
两个 `tools/plot_algorithm_*.py` 图表生成脚本向英文主站导出数据，并刷新中文共享副本。
各应用的启动入口及依赖见其自身的 `README.md` 和 `requirements.txt`。

## 共享资源维护

英文主站 `powerzoo/` 是主站 `data/*.json`、`assets/architecture.svg` 和
`requirements.txt` 的唯一维护来源；环境演示的共享依赖以
`powerzoo-vvc/requirements.txt` 为准。修改来源文件后运行：

```bash
python tools/sync_hf_shared_assets.py --write
python tools/sync_hf_shared_assets.py --check
```

两个算法图表生成脚本也会刷新这些共享副本。每个 Space 仍保留自己的文件，
可独立构建，不需要从部署目录外导入依赖或资源。CI 会检查中英资源漂移和多余的中文 JSON。

英文主站发布目标保持 `XJTU-RL/PowerZoo`。中文主站的实际远程仓库尚未在源码中声明，
确认后需将 GitHub 仓库变量 `HF_ZH_SPACE` 设置为完整的 `owner/repo`，并确保
`HF_TOKEN` 有写权限；未配置时工作流明确跳过中文发布。PR 只运行一致性检查，不发布。
