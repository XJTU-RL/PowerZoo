/**
 * PowerZoo Training Frontend - API服务层
 */
import axios from 'axios';
import type {
	AlgorithmsResponse,
	EnvironmentInfo,
	PresetConfig,
	TrainingConfig,
	TrainingTask,
	ValidationResult,
} from '@/types';

const api = axios.create({
	baseURL: '/api',
	timeout: 30000,
});

// 算法相关
export async function getAlgorithms(): Promise<AlgorithmsResponse> {
	const response = await api.get('/algorithms');
	return response.data;
}

export async function getAlgorithm(name: string) {
	const response = await api.get(`/algorithms/${name}`);
	return response.data;
}

// 环境相关
export async function getEnvironments(): Promise<EnvironmentInfo[]> {
	const response = await api.get('/environments');
	return response.data;
}

export async function getEnvironment(name: string): Promise<EnvironmentInfo> {
	const response = await api.get(`/environments/${name}`);
	return response.data;
}

// 预设相关
export async function getPresets(): Promise<PresetConfig[]> {
	const response = await api.get('/presets');
	return response.data;
}

export async function getPreset(name: string): Promise<PresetConfig> {
	const response = await api.get(`/presets/${name}`);
	return response.data;
}

// 配置相关
export async function getDefaultConfig(algo: string, env: string): Promise<TrainingConfig> {
	const response = await api.get(`/config/default/${algo}/${env}`);
	return response.data;
}

export async function validateConfig(config: TrainingConfig): Promise<ValidationResult> {
	const response = await api.post('/config/validate', config);
	return response.data;
}

// 训练任务相关
export async function startTraining(config: TrainingConfig): Promise<{ task_id: string; status: string; pid: number }> {
	const response = await api.post('/training/start', config);
	return response.data;
}

export async function getTasks(): Promise<TrainingTask[]> {
	const response = await api.get('/training/tasks');
	return response.data;
}

export async function getTask(taskId: string): Promise<TrainingTask> {
	const response = await api.get(`/training/tasks/${taskId}`);
	return response.data;
}

export async function getTaskLogs(taskId: string, lines: number = 100): Promise<{ logs: string }> {
	const response = await api.get(`/training/tasks/${taskId}/logs`, { params: { lines } });
	return response.data;
}

export async function stopTask(taskId: string): Promise<{ task_id: string; status: string }> {
	const response = await api.post(`/training/tasks/${taskId}/stop`);
	return response.data;
}

export async function deleteTask(taskId: string): Promise<{ task_id: string; status: string }> {
	const response = await api.delete(`/training/tasks/${taskId}`);
	return response.data;
}

export default api;
