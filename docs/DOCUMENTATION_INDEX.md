# PowerZoo Documentation Index

> **Last Updated**: 2025-11-26
> **Purpose**: Central index for all documentation in the PowerZoo project

---

## Project Overview

| Document | Location | Description |
|----------|----------|-------------|
| Project README | [README.md](../README.md) | Main project introduction and quick start |
| PowerZoo Environment Guide | [PowerZoo环境说明.md](../PowerZoo环境说明.md) | Chinese documentation for PowerZoo |
| Claude Instructions | [CLAUDE.md](../CLAUDE.md) | AI assistant configuration |

---

## Environment Documentation

### PowerZoo (Classic)
| Document | Location | Description |
|----------|----------|-------------|
| Main Environment | `envs/powerzoo/` | Classic PowerZoo environment code |

### PowerZoo LLM
| Document | Location | Description |
|----------|----------|-------------|
| Code Review Report | [smartgrid_code_review_report.md](./smartgrid_code_review_report.md) | Deep code review findings |
| Fix Summary | [smartgrid_fix_summary.md](./smartgrid_fix_summary.md) | Summary of applied fixes |
| Single Agent README | [envs/smartgrid/single_agent/README.md](../envs/smartgrid/single_agent/README.md) | Single agent configuration |

### DSR Environment
| Document | Location | Description |
|----------|----------|-------------|
| DSR README | [envs/dsr/README.md](../envs/dsr/README.md) | DSR environment overview |
| DSR Specifications | [envs/dsr/specifications.md](../envs/dsr/specifications.md) | Technical specifications |

### Stackelberg Environment
| Document | Location | Description |
|----------|----------|-------------|
| Stackelberg README | [envs/stackelberg/README.md](../envs/stackelberg/README.md) | Stackelberg game environment |
| Detailed README | [envs/stackelberg/README_STACKELBERG.md](../envs/stackelberg/README_STACKELBERG.md) | In-depth documentation |
| Structure Guide | [envs/stackelberg/STRUCTURE.md](../envs/stackelberg/STRUCTURE.md) | Code structure explanation |
| Optimization Summary | [envs/stackelberg/OPTIMIZATION_SUMMARY.md](../envs/stackelberg/OPTIMIZATION_SUMMARY.md) | Optimization notes |

---

## Algorithm Documentation

### Two-Timescale VVC
| Document | Location | Description |
|----------|----------|-------------|
| Algorithm README | [algorithms/twots_vvc/README.md](../algorithms/twots_vvc/README.md) | Two-timescale VVC algorithm |

### HAPPO Diagnostics
| Document | Location | Description |
|----------|----------|-------------|
| Diagnostics Summary | [HAPPO_DIAGNOSTICS_SUMMARY.md](../HAPPO_DIAGNOSTICS_SUMMARY.md) | HAPPO training diagnostics |

---

## Configuration Documentation

| Document | Location | Description |
|----------|----------|-------------|
| Config System README | [configs/envs_cfgs/CONFIG_SYSTEM_README.md](../configs/envs_cfgs/CONFIG_SYSTEM_README.md) | Configuration system overview |
| PV Plans README | [configs/envs_cfgs/smartgrid_pv_plans/README.md](../configs/envs_cfgs/smartgrid_pv_plans/README.md) | PV planning configurations |

---

## Data Documentation

### Load Data
| Document | Location | Description |
|----------|----------|-------------|
| Loads README | [data/Loads/README.md](../data/Loads/README.md) | Load data overview |
| Load Splitter Guide | [data/Loads/README_load_splitter.md](../data/Loads/README_load_splitter.md) | Load splitting utility |
| Usage Examples | [data/Loads/USAGE_EXAMPLES.md](../data/Loads/USAGE_EXAMPLES.md) | Example usage patterns |
| Minute-Level Data | [data/Loads/minute_level/README.md](../data/Loads/minute_level/README.md) | Minute-level load data |

### PV Data
| Document | Location | Description |
|----------|----------|-------------|
| PV README | [data/PV/README.md](../data/PV/README.md) | PV data overview |
| OpenDSS Integration | [data/PV/PV_OpenDSS_Integration_Guide.md](../data/PV/PV_OpenDSS_Integration_Guide.md) | OpenDSS integration guide |
| Visualization Guide | [data/PV/README_visualization.md](../data/PV/README_visualization.md) | Data visualization |
| Field Definition | [data/PV/filed_definition.md](../data/PV/filed_definition.md) | Data field definitions |
| Temperature Extension | [data/PV/generated_temperature/温度数据扩展指南.md](../data/PV/generated_temperature/温度数据扩展指南.md) | Temperature data guide |

### Batch Visualizations
| Document | Location | Description |
|----------|----------|-------------|
| Batch Summary | [data/PV/batch_visualizations/batch_summary.md](../data/PV/batch_visualizations/batch_summary.md) | Visualization batch summary |

---

## Node Systems Documentation

| Document | Location | Description |
|----------|----------|-------------|
| Node Systems Guide | [node_systems/OpenDSS节点系统使用说明.md](../node_systems/OpenDSS节点系统使用说明.md) | OpenDSS node system usage |
| 9500-Node Models | [node_systems/9500-Node/Original DSS Models/README.md](../node_systems/9500-Node/Original%20DSS%20Models/README.md) | 9500-node system docs |
| DSS Simulations | [node_systems/dss_simulations/README.md](../node_systems/dss_simulations/README.md) | DSS simulation examples |

---

## Examples Documentation

| Document | Location | Description |
|----------|----------|-------------|
| Examples README | [examples/README.md](../examples/README.md) | Example scripts overview |
| Multi-Agent Examples | [examples/multi_agent/README.md](../examples/multi_agent/README.md) | Multi-agent training |
| Single-Agent Examples | [examples/single_agent/README.md](../examples/single_agent/README.md) | Single-agent training |

---

## Test Documentation

| Document | Location | Description |
|----------|----------|-------------|
| Test README | [tests/README.md](../tests/README.md) | Testing overview |

---

## Research Papers

| Document | Location | Description |
|----------|----------|-------------|
| Async MARL Framework | [papers/2024-Asynchronous-multi-agent...](../papers/2024-Asynchronous-multi-agent-reinforcement-learning-based-framework-for-bi-level-noncooperative-game-theoretic-dem.md) | Research paper notes |
| MARL for DSR | [papers/A-multi-agent-reinforcement...](../papers/A-multi-agent-reinforcement-learning-method-for-distribution-system-restoration-considering-dynamic-network-reconfi.md) | DSR paper notes |

---

## Technical Reports

| Document | Location | Description |
|----------|----------|-------------|
| Long Methods Analysis | [docs/long_methods_refactoring_plan.md](./long_methods_refactoring_plan.md) | Methods requiring refactoring |

---

## Documentation Status

### Active Documentation (Current)
- All README files in data/, envs/, examples/, configs/
- Code review reports in docs/
- Algorithm documentation

### Needs Update
- Some Chinese documentation may need English translation
- API documentation needs generation (recommend Sphinx)

### Deprecated (Consider Removal)
- None identified - all documentation appears relevant

---

## Documentation Guidelines

### For New Documentation
1. Place general documentation in `docs/`
2. Keep module-specific READMEs with the code
3. Use English for technical documentation
4. Include code examples where applicable

### Naming Conventions
- Use `README.md` for module overviews
- Use descriptive names for specific guides (e.g., `PV_OpenDSS_Integration_Guide.md`)
- Use `UPPERCASE` for important documents (e.g., `CLAUDE.md`, `README.md`)

### Structure Template
```markdown
# Title

> **Purpose**: Brief description
> **Last Updated**: YYYY-MM-DD

## Overview
Brief introduction

## Usage
How to use this module/feature

## API Reference (if applicable)
Key classes/functions

## Examples
Code examples

## Troubleshooting
Common issues and solutions
```
