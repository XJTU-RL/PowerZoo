/**
 * PowerZoo Training Frontend - 配置状态管理
 */
import { create } from 'zustand';
import type {
	AlgorithmInfo,
	EnvironmentInfo,
	PresetConfig,
	TrainingConfig,
	TrainingTask,
	ValidationResult,
} from '@/types';
import { defaultConfig } from '@/types';
import * as api from '@/services/api';

interface ConfigStore {
	// 数据
	algorithms: {
		multi_agent: AlgorithmInfo[];
		single_agent: AlgorithmInfo[];
	};
	environments: EnvironmentInfo[];
	presets: PresetConfig[];
	tasks: TrainingTask[];

	// 当前配置
	config: TrainingConfig;
	validation: ValidationResult | null;

	// 加载状态
	loading: boolean;
	error: string | null;

	// 操作
	loadInitialData: () => Promise<void>;
	setConfig: (config: Partial<TrainingConfig>) => void;
	resetConfig: () => void;
	loadPreset: (presetName: string) => void;
	validateConfig: () => Promise<ValidationResult>;
	startTraining: () => Promise<string>;
	loadTasks: () => Promise<void>;
	stopTask: (taskId: string) => Promise<void>;
	deleteTask: (taskId: string) => Promise<void>;
	setAlgoAndEnv: (algo: string, env: string) => Promise<void>;
}

export const useConfigStore = create<ConfigStore>((set, get) => ({
	// 初始状态
	algorithms: { multi_agent: [], single_agent: [] },
	environments: [],
	presets: [],
	tasks: [],
	config: { ...defaultConfig },
	validation: null,
	loading: false,
	error: null,

	// 加载初始数据
	loadInitialData: async () => {
		set({ loading: true, error: null });
		try {
			const [algorithms, environments, presets, tasks] = await Promise.all([
				api.getAlgorithms(),
				api.getEnvironments(),
				api.getPresets(),
				api.getTasks(),
			]);
			set({ algorithms, environments, presets, tasks, loading: false });
		} catch (error) {
			set({ error: String(error), loading: false });
		}
	},

	// 设置配置
	setConfig: (partial) => {
		set((state) => ({
			config: { ...state.config, ...partial },
			validation: null, // 清除验证结果
		}));
	},

	// 重置配置
	resetConfig: () => {
		set({ config: { ...defaultConfig }, validation: null });
	},

	// 加载预设
	loadPreset: (presetName) => {
		const preset = get().presets.find((p) => p.name === presetName);
		if (preset) {
			set({ config: { ...preset.config }, validation: null });
		}
	},

	// 验证配置
	validateConfig: async () => {
		const result = await api.validateConfig(get().config);
		set({ validation: result });
		return result;
	},

	// 启动训练
	startTraining: async () => {
		const result = await api.startTraining(get().config);
		await get().loadTasks();
		return result.task_id;
	},

	// 加载任务列表
	loadTasks: async () => {
		const tasks = await api.getTasks();
		set({ tasks });
	},

	// 停止任务
	stopTask: async (taskId) => {
		await api.stopTask(taskId);
		await get().loadTasks();
	},

	// 删除任务
	deleteTask: async (taskId) => {
		await api.deleteTask(taskId);
		await get().loadTasks();
	},

	// 设置算法和环境，并加载默认配置
	setAlgoAndEnv: async (algo, env) => {
		try {
			const defaultCfg = await api.getDefaultConfig(algo, env);
			set({ config: defaultCfg, validation: null });
		} catch {
			set((state) => ({
				config: { ...state.config, algo, env },
				validation: null,
			}));
		}
	},
}));
