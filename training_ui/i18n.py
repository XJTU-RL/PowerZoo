"""Internationalization support for PowerZoo Training UI.

Centralized translation module. All UI-facing strings are defined here
in both English and Chinese. The ``t()`` function is the sole public API
for retrieving translated text.

Usage::

	from training_ui.i18n import t
	gr.Button(t("btn_start_training"))
"""

LANG: str = "zh"  # Default language: "zh" or "en"

_TEXTS: dict[str, dict[str, str]] = {
	# =====================================================================
	# App-level
	# =====================================================================
	"app_title": {
		"en": "PowerZoo Training Manager",
		"zh": "PowerZoo 训练管理系统",
	},
	"app_heading": {
		"en": "# PowerZoo Training Management System\n"
			"Configure, launch, and monitor multi-agent reinforcement learning training for power system environments.",
		"zh": "# PowerZoo 训练管理系统\n"
			"配置、启动和监控面向电力系统环境的多智能体强化学习训练。",
	},

	# =====================================================================
	# Tab labels (top-level)
	# =====================================================================
	"tab_vvc": {"en": "VVC", "zh": "VVC"},
	"tab_smartgrid": {"en": "SmartGrid", "zh": "智能电网"},
	"tab_stackelberg": {"en": "Stackelberg", "zh": "Stackelberg博弈"},
	"tab_dsr": {"en": "DSR", "zh": "DSR故障恢复"},
	"tab_monitor": {"en": "Monitor", "zh": "训练监控"},
	"tab_models": {"en": "Models", "zh": "模型管理"},

	# =====================================================================
	# Sub-tab labels (shared across env tabs)
	# =====================================================================
	"subtab_env_config": {"en": "Environment Config", "zh": "环境配置"},
	"subtab_algo_config": {"en": "Algorithm Config", "zh": "算法配置"},
	"subtab_training_settings": {"en": "Training Settings", "zh": "训练设置"},
	"subtab_launch": {"en": "Launch", "zh": "启动训练"},

	# =====================================================================
	# Launch panel (base_tab.py)
	# =====================================================================
	"label_exp_name": {"en": "Experiment Name", "zh": "实验名称"},
	"placeholder_exp_name": {"en": "Enter experiment name...", "zh": "输入实验名称..."},
	"label_model_dir": {"en": "Model Directory (resume training)", "zh": "模型目录（恢复训练）"},
	"placeholder_model_dir": {"en": "Leave empty for new training", "zh": "留空则开始新训练"},
	"label_log_dir": {"en": "Log Directory", "zh": "日志目录"},
	"label_config_preview": {"en": "Config Preview (YAML)", "zh": "配置预览 (YAML)"},
	"btn_preview_config": {"en": "Preview Config", "zh": "预览配置"},
	"btn_validate": {"en": "Validate", "zh": "验证"},
	"btn_start_training": {"en": "Start Training", "zh": "开始训练"},

	# Launch callback messages
	"err_no_algo": {"en": "No algorithm selected.", "zh": "未选择算法。"},
	"err_no_exp_name": {"en": "Experiment name is empty.", "zh": "实验名称为空。"},
	"err_episode_positive": {"en": "Episode length must be positive.", "zh": "回合长度必须为正数。"},
	"msg_validation_failed": {"en": "**Validation Failed**", "zh": "**验证失败**"},
	"msg_validation_passed": {"en": "**Validation Passed** -- config looks good.", "zh": "**验证通过** -- 配置正常。"},
	"msg_validation_error": {"en": "**Validation Error**", "zh": "**验证错误**"},
	"msg_error": {"en": "**Error**", "zh": "**错误**"},
	"msg_training_started": {"en": "**Training Started**", "zh": "**训练已启动**"},
	"msg_task_id": {"en": "Task ID", "zh": "任务 ID"},
	"msg_algorithm": {"en": "Algorithm", "zh": "算法"},
	"msg_environment": {"en": "Environment", "zh": "环境"},
	"msg_config": {"en": "Config", "zh": "配置"},
	"msg_launch_error": {"en": "**Launch Error**", "zh": "**启动错误**"},
	"err_preview": {"en": "# Error generating preview:", "zh": "# 预览生成错误:"},

	# =====================================================================
	# Algorithm selector (algo_selector.py)
	# =====================================================================
	"label_algo_family": {"en": "Algorithm Family", "zh": "算法族"},
	"label_algorithm": {"en": "Algorithm", "zh": "算法"},
	"algo_select_prompt": {
		"en": "*Select an algorithm to see its description.*",
		"zh": "*选择一个算法以查看其描述。*",
	},
	"algo_no_family": {
		"en": "*No algorithms in this family.*",
		"zh": "*该算法族中没有算法。*",
	},
	"algo_info_family": {"en": "Family", "zh": "算法族"},
	"algo_info_config": {"en": "Config", "zh": "配置文件"},
	"algo_info_compatible": {"en": "Compatible Envs", "zh": "兼容环境"},
	"algo_info_all": {"en": "All", "zh": "全部"},

	# =====================================================================
	# System selector (system_selector.py)
	# =====================================================================
	"label_ieee_system": {"en": "IEEE System", "zh": "IEEE 测试系统"},
	"system_no_metadata": {
		"en": "No metadata available for",
		"zh": "无元数据可用:",
	},
	"system_select_prompt": {
		"en": "*Select a system to see its details.*",
		"zh": "*选择一个系统以查看其详情。*",
	},
	"system_property": {"en": "Property", "zh": "属性"},
	"system_value": {"en": "Value", "zh": "值"},
	"system_nodes": {"en": "Nodes", "zh": "节点数"},
	"system_pv_units": {"en": "PV Units", "zh": "光伏单元"},
	"system_pv_penetration": {"en": "PV Penetration", "zh": "光伏渗透率"},
	"system_districts": {"en": "Districts", "zh": "区域数"},
	"system_episode_length": {"en": "Episode Length", "zh": "回合长度"},

	# =====================================================================
	# Log viewer (log_viewer.py)
	# =====================================================================
	"label_training_log": {"en": "Training Log", "zh": "训练日志"},
	"placeholder_log": {
		"en": "Training output will appear here once a run starts...",
		"zh": "训练开始后，输出将显示在此处...",
	},

	# =====================================================================
	# VVC tab (vvc_tab.py)
	# =====================================================================
	"vvc_accordion_settings": {"en": "VVC Environment Settings", "zh": "VVC 环境设置"},
	"label_episode_length": {"en": "Episode Length", "zh": "回合长度"},
	"info_episode_length": {"en": "Maximum steps per episode", "zh": "每回合最大步数"},
	"label_env_seed": {"en": "Env Seed", "zh": "环境随机种子"},
	"label_mode": {"en": "Mode", "zh": "模式"},
	"info_mode": {"en": "Environment execution mode", "zh": "环境执行模式"},
	"label_use_s_matrix": {"en": "Use S Matrix (SHOM)", "zh": "使用 S 矩阵 (SHOM)"},
	"label_big2small": {"en": "Big to Small (SHOM)", "zh": "大到小 (SHOM)"},
	"accordion_runtime_flags": {"en": "Runtime Flags", "zh": "运行时标志"},
	"label_render": {"en": "Render", "zh": "渲染"},
	"label_plot": {"en": "Plot", "zh": "绘图"},
	"label_testing_mode": {"en": "Testing Mode", "zh": "测试模式"},
	"label_record_node": {"en": "Record Node Info", "zh": "记录节点信息"},
	"label_dss_auto_control": {"en": "DSS Auto Control", "zh": "DSS 自动控制"},
	"info_dss_auto_control": {
		"en": "If enabled, OpenDSS controls override RL actions",
		"zh": "启用后，OpenDSS 控制将覆盖 RL 动作",
	},
	"accordion_logging": {"en": "Logging", "zh": "日志配置"},
	"label_system_logging": {"en": "System Logging", "zh": "系统日志"},
	"label_realtime_log": {"en": "Realtime Log", "zh": "实时日志"},
	"label_log_directory": {"en": "Log Directory", "zh": "日志目录"},
	"label_log_buffer_size": {"en": "Log Buffer Size", "zh": "日志缓冲区大小"},
	"label_log_save_interval": {"en": "Log Save Interval", "zh": "日志保存间隔"},

	# =====================================================================
	# SmartGrid tab (smartgrid_tab.py)
	# =====================================================================
	"smartgrid_accordion_settings": {"en": "SmartGrid Environment Settings", "zh": "智能电网环境设置"},
	"info_smartgrid_episode": {
		"en": "Maximum steps per episode (360 = 15min resolution over 24h)",
		"zh": "每回合最大步数（360 = 24小时内15分钟分辨率）",
	},
	"accordion_reward_weights": {"en": "Reward Weights", "zh": "奖励权重"},
	"reward_weights_desc": {
		"en": "Override system default reward weights. Values from `environment_specific.reward_weights` in YAML.",
		"zh": "覆盖系统默认奖励权重。值来自 YAML 中的 `environment_specific.reward_weights`。",
	},
	"label_power_loss": {"en": "Power Loss", "zh": "功率损耗"},
	"label_capacitor": {"en": "Capacitor", "zh": "电容器"},
	"label_regulator": {"en": "Regulator", "zh": "调压器"},
	"label_battery_soc": {"en": "Battery SOC", "zh": "电池 SOC"},
	"label_battery_discharge": {"en": "Battery Discharge", "zh": "电池放电"},
	"label_pv_control": {"en": "PV Control", "zh": "光伏控制"},
	"accordion_voltage_constraints": {"en": "Voltage Constraints", "zh": "电压约束"},
	"label_voltage_min": {"en": "Voltage Min (p.u.)", "zh": "最低电压 (p.u.)"},
	"label_voltage_max": {"en": "Voltage Max (p.u.)", "zh": "最高电压 (p.u.)"},
	"label_voltage_penalty_scale": {"en": "Voltage Penalty Scale", "zh": "电压惩罚系数"},
	"label_constraint_aware": {"en": "Constraint Aware", "zh": "约束感知"},
	"accordion_curriculum": {"en": "Curriculum Learning", "zh": "课程学习"},
	"label_enable_curriculum": {"en": "Enable Curriculum Learning", "zh": "启用课程学习"},
	"label_voltage_violation_penalty": {"en": "Voltage Violation Penalty", "zh": "电压越限惩罚"},
	"label_power_loss_weight": {"en": "Power Loss Weight", "zh": "功率损耗权重"},
	"label_control_cost_weight": {"en": "Control Cost Weight", "zh": "控制代价权重"},
	"label_llm_enhanced": {"en": "LLM Enhanced", "zh": "LLM 增强"},
	"info_llm_enhanced": {
		"en": "Enable LLM-enhanced observations and action explanations",
		"zh": "启用 LLM 增强的观测和动作解释",
	},

	# =====================================================================
	# Stackelberg tab (stackelberg_tab.py)
	# =====================================================================
	"label_stackelberg_variant": {"en": "Stackelberg Variant", "zh": "Stackelberg 变体"},
	"info_stackelberg_variant": {
		"en": "Selects the IEEE bus system and default agent count",
		"zh": "选择 IEEE 母线系统和默认智能体数量",
	},
	"accordion_agent_config": {"en": "Agent Configuration", "zh": "智能体配置"},
	"label_n_consumer_agents": {"en": "Number of Consumer Agents", "zh": "消费者智能体数量"},
	"label_max_episode_steps": {"en": "Max Episode Steps", "zh": "最大回合步数"},
	"label_seed": {"en": "Seed", "zh": "随机种子"},
	"label_consumer_bus_mapping": {"en": "Consumer Bus Mapping (JSON)", "zh": "消费者母线映射 (JSON)"},
	"accordion_uc_action": {"en": "UC Action Space", "zh": "UC 动作空间"},
	"label_price_signal_low": {"en": "Price Signal Low", "zh": "价格信号下限"},
	"label_price_signal_high": {"en": "Price Signal High", "zh": "价格信号上限"},
	"label_dr_incentive_low": {"en": "DR Incentive Low", "zh": "DR 激励下限"},
	"label_dr_incentive_high": {"en": "DR Incentive High", "zh": "DR 激励上限"},
	"label_capacity_alloc_low": {"en": "Capacity Allocation Low", "zh": "容量分配下限"},
	"label_capacity_alloc_high": {"en": "Capacity Allocation High", "zh": "容量分配上限"},
	"label_ess_charge_low": {"en": "ESS Charge Low", "zh": "储能充电下限"},
	"label_ess_charge_high": {"en": "ESS Charge High", "zh": "储能充电上限"},
	"label_der_curtail_low": {"en": "DER Curtailment Low", "zh": "DER 弃电下限"},
	"label_der_curtail_high": {"en": "DER Curtailment High", "zh": "DER 弃电上限"},
	"accordion_consumer_action": {"en": "Consumer Action Space", "zh": "消费者动作空间"},
	"label_load_adj_low": {"en": "Load Adjustment Low", "zh": "负荷调整下限"},
	"info_load_adj_low": {"en": "Maximum load reduction (30%)", "zh": "最大负荷削减 (30%)"},
	"label_load_adj_high": {"en": "Load Adjustment High", "zh": "负荷调整上限"},
	"info_load_adj_high": {"en": "Maximum load increase (10%)", "zh": "最大负荷增加 (10%)"},
	"label_der_output_low": {"en": "DER Output Low", "zh": "DER 输出下限"},
	"label_der_output_high": {"en": "DER Output High", "zh": "DER 输出上限"},
	"accordion_reward_weights_stk": {"en": "Reward Weights", "zh": "奖励权重"},
	"heading_uc_rewards": {"en": "#### UC Rewards", "zh": "#### UC 奖励"},
	"label_electricity_revenue": {"en": "Electricity Revenue", "zh": "售电收入"},
	"label_market_cost": {"en": "Market Cost", "zh": "市场成本"},
	"label_der_profit": {"en": "DER Profit", "zh": "DER 利润"},
	"label_dr_cost": {"en": "DR Cost", "zh": "DR 成本"},
	"label_system_loss": {"en": "System Loss", "zh": "系统损耗"},
	"label_voltage_violation": {"en": "Voltage Violation", "zh": "电压越限"},
	"label_carbon_reduction": {"en": "Carbon Reduction", "zh": "碳减排"},
	"heading_consumer_rewards": {"en": "#### Consumer Rewards", "zh": "#### 消费者奖励"},
	"label_electricity_cost": {"en": "Electricity Cost", "zh": "用电成本"},
	"label_comfort_loss": {"en": "Comfort Loss", "zh": "舒适度损失"},
	"label_dr_revenue": {"en": "DR Revenue", "zh": "DR 收入"},
	"label_voltage_quality": {"en": "Voltage Quality", "zh": "电压质量"},
	"accordion_physical_constraints": {"en": "Physical Constraints", "zh": "物理约束"},
	"label_line_capacity_factor": {"en": "Line Capacity Factor", "zh": "线路容量因子"},
	"label_max_load_change_rate": {"en": "Max Load Change Rate", "zh": "最大负荷变化率"},
	"label_ess_ramp_rate": {"en": "ESS Ramp Rate", "zh": "储能爬坡速率"},
	"accordion_market": {"en": "Market Configuration", "zh": "市场配置"},
	"label_base_price": {"en": "Base Price ($/kWh)", "zh": "基础电价 ($/kWh)"},
	"label_peak_multiplier": {"en": "Peak Multiplier", "zh": "峰时倍率"},
	"label_valley_multiplier": {"en": "Valley Multiplier", "zh": "谷时倍率"},
	"label_market_volatility": {"en": "Market Volatility", "zh": "市场波动率"},
	"accordion_tou": {"en": "TOU Configuration", "zh": "分时电价配置"},
	"label_peak_hours": {"en": "Peak Hours", "zh": "峰时时段"},
	"label_valley_hours": {"en": "Valley Hours", "zh": "谷时时段"},
	"accordion_ess": {"en": "ESS Configuration", "zh": "储能配置"},
	"label_ess_total_capacity": {"en": "Total Capacity (MWh)", "zh": "总容量 (MWh)"},
	"label_ess_initial_soc": {"en": "Initial SOC", "zh": "初始 SOC"},
	"label_ess_charge_efficiency": {"en": "Charge Efficiency", "zh": "充电效率"},
	"label_ess_discharge_efficiency": {"en": "Discharge Efficiency", "zh": "放电效率"},
	"label_ess_self_discharge": {"en": "Self Discharge Rate", "zh": "自放电率"},
	"label_ess_max_power": {"en": "Max Power (MW)", "zh": "最大功率 (MW)"},
	"label_ess_min_soc": {"en": "Min SOC", "zh": "最小 SOC"},
	"label_ess_max_soc": {"en": "Max SOC", "zh": "最大 SOC"},
	"accordion_der": {"en": "DER Configuration", "zh": "DER 配置"},
	"label_der_total_capacity": {"en": "Total Capacity (MW)", "zh": "总容量 (MW)"},
	"label_der_availability": {"en": "Availability Profile", "zh": "可用性配置"},
	"label_der_forecast_error": {"en": "Forecast Error Std", "zh": "预测误差标准差"},
	"label_der_curtailment_cost": {"en": "Curtailment Cost ($/kWh)", "zh": "弃电成本 ($/kWh)"},
	"accordion_dr": {"en": "DR Configuration", "zh": "需求响应配置"},
	"label_dr_max_ratio": {"en": "Max DR Ratio", "zh": "最大 DR 比率"},
	"label_dr_min_response": {"en": "Min Response Time (h)", "zh": "最小响应时间 (h)"},
	"label_dr_fatigue": {"en": "Fatigue Factor", "zh": "疲劳因子"},
	"label_dr_participation": {"en": "Participation Rate", "zh": "参与率"},
	"accordion_carbon_n1": {"en": "Carbon & N-1 Security", "zh": "碳排放与 N-1 安全"},
	"heading_carbon_tracking": {"en": "#### Carbon Tracking", "zh": "#### 碳排放追踪"},
	"label_track_emissions": {"en": "Track Emissions", "zh": "追踪碳排放"},
	"label_grid_carbon_intensity": {"en": "Grid Carbon Intensity (kg CO2/kWh)", "zh": "电网碳强度 (kg CO2/kWh)"},
	"label_carbon_price": {"en": "Carbon Price ($/kg CO2)", "zh": "碳价 ($/kg CO2)"},
	"heading_n1_security": {"en": "#### N-1 Security", "zh": "#### N-1 安全"},
	"label_n1_enable": {"en": "Enable N-1 Security", "zh": "启用 N-1 安全"},
	"label_contingency_prob": {"en": "Contingency Probability", "zh": "事故概率"},
	"label_recovery_time": {"en": "Recovery Time (h)", "zh": "恢复时间 (h)"},
	"label_debug": {"en": "Debug", "zh": "调试"},
	"label_verbose": {"en": "Verbose", "zh": "详细输出"},

	# =====================================================================
	# DSR tab (dsr_tab.py)
	# =====================================================================
	"label_dsr_variant": {"en": "DSR Variant", "zh": "DSR 变体"},
	"info_dsr_variant": {
		"en": "dsr=123Bus, dsr_13bus=13Bus, dsr_8500node=8500Node",
		"zh": "dsr=123Bus, dsr_13bus=13Bus, dsr_8500node=8500Node",
	},
	"accordion_episode_settings": {"en": "Episode Settings", "zh": "回合设置"},
	"accordion_device_config": {"en": "Device Configuration", "zh": "设备配置"},
	"label_n_dg": {"en": "Diesel Generators (n_dg)", "zh": "柴油发电机 (n_dg)"},
	"label_n_pv": {"en": "PV Units (n_pv)", "zh": "光伏单元 (n_pv)"},
	"label_n_switch": {"en": "Switches (n_switch)", "zh": "开关 (n_switch)"},
	"label_n_load_levels": {"en": "Load Priority Levels", "zh": "负荷优先级等级"},
	"accordion_load_agg": {"en": "Load Aggregation", "zh": "负荷聚合"},
	"label_use_load_agg": {"en": "Use Load Aggregation", "zh": "使用负荷聚合"},
	"info_use_load_agg": {
		"en": "Aggregate individual loads into agent groups",
		"zh": "将单个负荷聚合为智能体组",
	},
	"label_agg_method": {"en": "Aggregation Method", "zh": "聚合方法"},
	"accordion_fault_config": {"en": "Fault Configuration", "zh": "故障配置"},
	"label_fault_scenarios": {"en": "Fault Scenarios", "zh": "故障场景数"},
	"label_min_faults": {"en": "Min Faults per Scenario", "zh": "每场景最少故障数"},
	"label_max_faults": {"en": "Max Faults per Scenario", "zh": "每场景最多故障数"},
	"accordion_physical_constraints_dsr": {"en": "Physical Constraints", "zh": "物理约束"},
	"label_max_load_per_step": {"en": "Max Load per Step (kW)", "zh": "每步最大负荷 (kW)"},
	"accordion_reward_weights_dsr": {"en": "Reward Weights", "zh": "奖励权重"},
	"label_restore_reward": {"en": "Restore Reward", "zh": "恢复奖励"},
	"label_voltage_reward": {"en": "Voltage Reward", "zh": "电压奖励"},
	"label_overload_penalty": {"en": "Overload Penalty", "zh": "过载惩罚"},
	"label_done_penalty": {"en": "Done Penalty", "zh": "终止惩罚"},
	"info_done_penalty": {
		"en": "Penalty when episode ends without full restoration",
		"zh": "回合结束但未完全恢复时的惩罚",
	},
	"accordion_priority_weights": {"en": "Priority Weights", "zh": "优先级权重"},
	"priority_weights_desc": {
		"en": "Load priority weights by level (higher = more important)",
		"zh": "各等级负荷优先级权重（越高越重要）",
	},
	"label_priority_1": {"en": "Priority 1 (Critical)", "zh": "优先级 1（关键）"},
	"label_priority_2": {"en": "Priority 2 (Important)", "zh": "优先级 2（重要）"},
	"label_priority_3": {"en": "Priority 3 (Normal)", "zh": "优先级 3（普通）"},
	"accordion_advanced": {"en": "Advanced Features", "zh": "高级功能"},
	"label_use_action_mask": {"en": "Use Action Mask", "zh": "使用动作掩码"},
	"info_use_action_mask": {
		"en": "Mask invalid actions in the action space",
		"zh": "在动作空间中屏蔽无效动作",
	},
	"label_use_dynamic_network": {"en": "Use Dynamic Network", "zh": "使用动态网络"},
	"info_use_dynamic_network": {
		"en": "Enable dynamic network topology changes",
		"zh": "启用动态网络拓扑变化",
	},
	"heading_action_space": {"en": "#### Action Space Configuration", "zh": "#### 动作空间配置"},
	"label_pv_power_levels": {"en": "PV Power Levels", "zh": "光伏功率等级"},
	"label_load_action_levels": {"en": "Load Action Levels", "zh": "负荷动作等级"},
	"label_pv_max_power": {"en": "PV Max Power (kW)", "zh": "光伏最大功率 (kW)"},
	"heading_overload_detection": {"en": "#### Overload Detection", "zh": "#### 过载检测"},
	"label_overload_threshold": {"en": "Overload Threshold", "zh": "过载阈值"},
	"label_line_disconnect_prob": {"en": "Line Disconnect Probability", "zh": "线路断开概率"},
	"info_render_large": {
		"en": "Large systems: rendering not recommended",
		"zh": "大型系统：不建议渲染",
	},
	"info_record_node_8500": {
		"en": "8500node: disabled by default to save memory",
		"zh": "8500node: 默认禁用以节省内存",
	},

	# =====================================================================
	# Monitor tab (monitor_tab.py)
	# =====================================================================
	"monitor_heading": {"en": "## Training Monitor", "zh": "## 训练监控"},
	"label_active_task": {"en": "Active Task", "zh": "当前任务"},
	"btn_refresh": {"en": "Refresh", "zh": "刷新"},
	"label_task_info": {"en": "Task Info", "zh": "任务信息"},
	"btn_stop_training": {"en": "Stop Training", "zh": "停止训练"},
	"monitor_log_heading": {"en": "### Training Log", "zh": "### 训练日志"},
	"label_log_output": {"en": "Log Output", "zh": "日志输出"},
	"monitor_history_heading": {"en": "### Task History", "zh": "### 任务历史"},
	"label_all_tasks": {"en": "All Tasks", "zh": "全部任务"},
	"col_task_id": {"en": "Task ID", "zh": "任务 ID"},
	"col_algorithm": {"en": "Algorithm", "zh": "算法"},
	"col_environment": {"en": "Environment", "zh": "环境"},
	"col_status": {"en": "Status", "zh": "状态"},
	"col_start_time": {"en": "Start Time", "zh": "开始时间"},
	"col_duration": {"en": "Duration", "zh": "持续时间"},
	"msg_no_task": {"en": "No task selected", "zh": "未选择任务"},
	"msg_task_stopped": {"en": "stopped", "zh": "已停止"},
	"msg_task_stop_failed": {"en": "Failed to stop", "zh": "停止失败"},

	# =====================================================================
	# Models tab (models_tab.py)
	# =====================================================================
	"models_heading": {"en": "## Trained Models", "zh": "## 已训练模型"},
	"label_env_filter": {"en": "Environment Filter", "zh": "环境筛选"},
	"label_algo_filter": {"en": "Algorithm Filter", "zh": "算法筛选"},
	"btn_scan_results": {"en": "Scan Results", "zh": "扫描结果"},
	"label_discovered_models": {"en": "Discovered Models", "zh": "发现的模型"},
	"col_env": {"en": "Env", "zh": "环境"},
	"col_system": {"en": "System", "zh": "系统"},
	"col_experiment": {"en": "Experiment", "zh": "实验"},
	"col_run_id": {"en": "Run ID", "zh": "运行 ID"},
	"col_checkpoints": {"en": "Checkpoints", "zh": "检查点"},
	"label_model_details": {"en": "Model Details", "zh": "模型详情"},
	"label_training_config": {"en": "Training Config", "zh": "训练配置"},
	"label_model_path": {"en": "Model Path", "zh": "模型路径"},
	"btn_copy_path": {"en": "Copy Path", "zh": "复制路径"},
	"label_checkpoint": {"en": "Checkpoint", "zh": "检查点"},

	# =====================================================================
	# Parameter panels (param_panels.py)
	# =====================================================================
	"accordion_seed_device": {"en": "Seed & Device", "zh": "随机种子与设备"},
	"label_specify_seed": {"en": "Specify Seed", "zh": "指定随机种子"},
	"label_use_cuda": {"en": "Use CUDA", "zh": "使用 CUDA"},
	"label_cuda_deterministic": {"en": "CUDA Deterministic", "zh": "CUDA 确定性"},
	"label_torch_threads": {"en": "Torch Threads", "zh": "Torch 线程数"},

	"accordion_training_settings": {"en": "Training Settings", "zh": "训练设置"},
	"label_rollout_threads": {"en": "Rollout Threads", "zh": "采样线程数"},
	"label_total_env_steps": {"en": "Total Env Steps", "zh": "总环境步数"},
	"label_log_interval": {"en": "Log Interval (episodes)", "zh": "日志间隔（回合）"},
	"label_eval_interval": {"en": "Eval Interval (episodes)", "zh": "评估间隔（回合）"},
	"label_save_interval": {"en": "Save Interval (episodes)", "zh": "保存间隔（回合）"},
	"label_value_norm": {"en": "Value Normalization", "zh": "值归一化"},
	"label_linear_lr_decay": {"en": "Linear LR Decay", "zh": "线性学习率衰减"},

	"accordion_network": {"en": "Network Architecture", "zh": "网络架构"},
	"label_hidden_sizes": {"en": "Hidden Sizes (JSON)", "zh": "隐藏层大小 (JSON)"},
	"label_activation": {"en": "Activation", "zh": "激活函数"},
	"label_feature_norm": {"en": "Feature Normalization", "zh": "特征归一化"},
	"label_init_method": {"en": "Init Method", "zh": "初始化方法"},
	"label_output_gain": {"en": "Output Gain", "zh": "输出增益"},
	"heading_recurrent_policy": {"en": "### Recurrent Policy", "zh": "### 循环策略"},
	"label_use_recurrent": {"en": "Use Recurrent Policy", "zh": "使用循环策略"},
	"label_naive_recurrent": {"en": "Naive Recurrent", "zh": "简单循环"},
	"label_recurrent_layers": {"en": "Recurrent Layers", "zh": "循环层数"},
	"label_data_chunk_length": {"en": "Data Chunk Length", "zh": "数据分块长度"},
	"heading_learning_rate": {"en": "### Learning Rate", "zh": "### 学习率"},
	"label_actor_lr": {"en": "Actor LR", "zh": "Actor 学习率"},
	"label_critic_lr": {"en": "Critic LR", "zh": "Critic 学习率"},

	"accordion_on_policy": {"en": "On-Policy Algorithm (PPO/HAPPO)", "zh": "On-Policy 算法 (PPO/HAPPO)"},
	"label_ppo_epoch": {"en": "PPO Epoch", "zh": "PPO 迭代次数"},
	"label_critic_epoch": {"en": "Critic Epoch", "zh": "Critic 迭代次数"},
	"label_clip_param": {"en": "Clip Param", "zh": "裁剪参数"},
	"label_entropy_coef": {"en": "Entropy Coef", "zh": "熵系数"},
	"label_value_loss_coef": {"en": "Value Loss Coef", "zh": "值损失系数"},
	"label_max_grad_norm": {"en": "Max Grad Norm", "zh": "最大梯度范数"},
	"label_gamma": {"en": "Gamma", "zh": "折扣因子 Gamma"},
	"label_gae_lambda": {"en": "GAE Lambda", "zh": "GAE Lambda"},
	"label_actor_mini_batch": {"en": "Actor Mini Batch", "zh": "Actor 小批次"},
	"label_critic_mini_batch": {"en": "Critic Mini Batch", "zh": "Critic 小批次"},
	"label_clipped_value_loss": {"en": "Clipped Value Loss", "zh": "裁剪值损失"},
	"label_use_max_grad_norm": {"en": "Use Max Grad Norm", "zh": "使用最大梯度范数"},
	"label_use_gae": {"en": "Use GAE", "zh": "使用 GAE"},
	"label_use_huber_loss": {"en": "Use Huber Loss", "zh": "使用 Huber 损失"},
	"label_huber_delta": {"en": "Huber Delta", "zh": "Huber Delta"},
	"label_action_aggregation": {"en": "Action Aggregation", "zh": "动作聚合"},
	"label_share_param": {"en": "Share Parameters", "zh": "共享参数"},
	"label_fixed_order": {"en": "Fixed Order", "zh": "固定顺序"},

	"accordion_off_policy": {"en": "Off-Policy Algorithm", "zh": "Off-Policy 算法"},
	"label_buffer_size": {"en": "Buffer Size", "zh": "缓冲区大小"},
	"label_batch_size": {"en": "Batch Size", "zh": "批次大小"},
	"label_polyak": {"en": "Polyak (Soft Update)", "zh": "Polyak 系数（软更新）"},
	"label_n_step": {"en": "N-Step Returns", "zh": "N 步回报"},
	"label_warmup_steps": {"en": "Warmup Steps", "zh": "预热步数"},
	"label_train_interval": {"en": "Train Interval", "zh": "训练间隔"},
	"label_updates_per_train": {"en": "Updates Per Train", "zh": "每次训练更新数"},
	"heading_sac_temp": {"en": "### SAC Temperature", "zh": "### SAC 温度参数"},
	"label_auto_alpha": {"en": "Auto Alpha", "zh": "自动 Alpha"},
	"label_alpha": {"en": "Alpha", "zh": "Alpha"},
	"label_alpha_lr": {"en": "Alpha LR", "zh": "Alpha 学习率"},
	"label_huber_loss": {"en": "Huber Loss", "zh": "Huber 损失"},

	"accordion_eval": {"en": "Evaluation", "zh": "评估"},
	"label_enable_eval": {"en": "Enable Evaluation", "zh": "启用评估"},
	"label_eval_threads": {"en": "Eval Threads", "zh": "评估线程数"},
	"label_eval_episodes": {"en": "Eval Episodes", "zh": "评估回合数"},

	# =====================================================================
	# District Dispatch tab (district_dispatch_tab.py)
	# =====================================================================
	"tab_district_dispatch": {"en": "District Dispatch", "zh": "分区调度"},
	"dd_accordion_zone_config": {"en": "Zone Configuration", "zh": "分区配置"},
	"dd_label_n_districts": {"en": "Number of Districts", "zh": "分区数量"},
	"dd_info_n_districts": {"en": "Number of cooperative dispatch zones", "zh": "协同调度分区的数量"},
	"dd_label_connection_mode": {"en": "Connection Mode", "zh": "连接模式"},
	"dd_info_connection_mode": {
		"en": "Inter-zone connection type: transformer, tieline, or mixed",
		"zh": "区间连接类型：变压器、联络线或混合",
	},
	"dd_info_episode_steps": {
		"en": "96 = 15min resolution over 24h",
		"zh": "96 = 24小时内15分钟分辨率",
	},
	"dd_accordion_observation": {"en": "Observation Settings", "zh": "观测设置"},
	"dd_label_use_neighbor_obs": {"en": "Use Neighbor Observations", "zh": "使用邻区观测"},
	"dd_info_use_neighbor_obs": {
		"en": "Include neighboring zone observations in agent state",
		"zh": "在智能体状态中包含邻区观测信息",
	},
	"dd_label_max_neighbors": {"en": "Max Neighbors", "zh": "最大邻区数"},
	"dd_accordion_physical": {"en": "Physical Constraints", "zh": "物理约束"},
	"dd_heading_voltage": {"en": "#### Voltage Limits", "zh": "#### 电压限制"},
	"dd_heading_soc": {"en": "#### Battery SOC Limits", "zh": "#### 电池 SOC 限制"},
	"dd_label_soc_min": {"en": "SOC Min", "zh": "最小 SOC"},
	"dd_label_soc_max": {"en": "SOC Max", "zh": "最大 SOC"},
	"dd_label_soc_init": {"en": "Initial SOC", "zh": "初始 SOC"},
	"dd_label_charge_eff": {"en": "Charge Efficiency", "zh": "充电效率"},
	"dd_accordion_market": {"en": "Market Parameters", "zh": "市场参数"},
	"dd_label_base_price": {"en": "Base Price (yuan/kWh)", "zh": "基础电价 (元/kWh)"},
	"dd_label_carbon_intensity": {"en": "Carbon Intensity (kg/kWh)", "zh": "碳强度 (kg/kWh)"},
	"dd_label_carbon_price": {"en": "Carbon Price (yuan/kg)", "zh": "碳价 (元/kg)"},
	"dd_accordion_reward_weights": {"en": "Reward Weights", "zh": "奖励权重"},
	"dd_reward_weights_desc": {
		"en": "Weights for the 6 reward components in multi-zone dispatch.",
		"zh": "多分区调度中 6 个奖励分量的权重。",
	},
	"dd_label_rw_economic": {"en": "Economic", "zh": "经济性"},
	"dd_label_rw_voltage": {"en": "Voltage", "zh": "电压"},
	"dd_label_rw_loss": {"en": "Loss", "zh": "损耗"},
	"dd_label_rw_carbon": {"en": "Carbon", "zh": "碳排放"},
	"dd_label_rw_exchange": {"en": "Exchange", "zh": "功率交换"},
	"dd_label_rw_storage": {"en": "Storage", "zh": "储能"},
	"dd_label_load_noise": {"en": "Load Noise", "zh": "负荷噪声"},
	"dd_label_noise_std": {"en": "Noise Std", "zh": "噪声标准差"},

	# =====================================================================
	# Launch CLI (launch.py)
	# =====================================================================
	"cli_description": {"en": "PowerZoo Training Manager", "zh": "PowerZoo 训练管理系统"},
	"cli_port_help": {"en": "Server port", "zh": "服务端口"},
	"cli_host_help": {"en": "Server host", "zh": "服务地址"},
	"cli_share_help": {"en": "Create public link", "zh": "创建公共链接"},

}


def t(key: str) -> str:
	"""Get translated text for the current language.

	Args:
		key: Translation key defined in ``_TEXTS``.

	Returns:
		Translated string. Falls back to English, then to the key itself.
	"""
	entry = _TEXTS.get(key, {})
	return entry.get(LANG, entry.get("en", key))
