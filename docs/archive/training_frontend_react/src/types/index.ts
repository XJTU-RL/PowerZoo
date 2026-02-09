/**
 * PowerZoo Training Frontend - TypeScript类型定义
 */

export interface AlgorithmInfo {
	name: string;
	display_name: string;
	type: 'on_policy' | 'off_policy' | 'two_timescale';
	category: 'ha' | 'ma' | 'qmix' | 'single' | 'special';
	runner: string;
	config_file: string;
	description: string;
}

export interface EnvironmentInfo {
	name: string;
	display_name: string;
	type: string;
	config_file: string;
	description: string;
	default_agents: number;
}

export interface TrainingConfig {
	algo: string;
	env: string;
	exp_name: string;

	// 种子配置
	seed_specify: boolean;
	seed: number;

	// 设备配置
	cuda: boolean;
	cuda_deterministic: boolean;
	torch_threads: number;

	// 训练配置
	n_rollout_threads: number;
	num_env_steps: number;
	episode_length: number;
	log_interval: number;
	eval_interval: number;
	save_interval: number;
	use_valuenorm: boolean;
	use_linear_lr_decay: boolean;

	// 评估配置
	use_eval: boolean;
	n_eval_rollout_threads: number;
	eval_episodes: number;

	// 模型配置
	hidden_sizes: number[];
	activation_func: string;
	use_feature_normalization: boolean;
	use_recurrent_policy: boolean;
	recurrent_n: number;
	data_chunk_length: number;

	// 算法配置
	lr: number;
	critic_lr: number;
	gamma: number;
	gae_lambda: number;
	entropy_coef: number;
	max_grad_norm: number;
	ppo_epoch: number;
	clip_param: number;
	actor_num_mini_batch: number;
	critic_num_mini_batch: number;

	// Off-policy特定
	buffer_size: number;
	batch_size: number;
	polyak: number;

	// 环境特定配置
	env_args: Record<string, unknown>;
}

export interface TrainingTask {
	task_id: string;
	config: TrainingConfig;
	status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
	created_at: string;
	started_at: string | null;
	ended_at: string | null;
	pid: number | null;
	log_file: string | null;
}

export interface PresetConfig {
	name: string;
	description: string;
	config: TrainingConfig;
}

export interface ValidationResult {
	valid: boolean;
	errors: string[];
	warnings: string[];
}

export interface AlgorithmsResponse {
	multi_agent: AlgorithmInfo[];
	single_agent: AlgorithmInfo[];
}

// 默认配置
export const defaultConfig: TrainingConfig = {
	algo: 'happo',
	env: 'smartgrid',
	exp_name: 'experiment_1',

	seed_specify: true,
	seed: 12345,

	cuda: true,
	cuda_deterministic: true,
	torch_threads: 4,

	n_rollout_threads: 4,
	num_env_steps: 2000000,
	episode_length: 360,
	log_interval: 1,
	eval_interval: 2,
	save_interval: 5,
	use_valuenorm: true,
	use_linear_lr_decay: false,

	use_eval: true,
	n_eval_rollout_threads: 2,
	eval_episodes: 10,

	hidden_sizes: [128, 128],
	activation_func: 'relu',
	use_feature_normalization: true,
	use_recurrent_policy: true,
	recurrent_n: 1,
	data_chunk_length: 60,

	lr: 5e-4,
	critic_lr: 5e-4,
	gamma: 0.99,
	gae_lambda: 0.95,
	entropy_coef: 0.08,
	max_grad_norm: 10.0,
	ppo_epoch: 5,
	clip_param: 0.25,
	actor_num_mini_batch: 1,
	critic_num_mini_batch: 1,

	buffer_size: 100000,
	batch_size: 256,
	polyak: 0.005,

	env_args: {},
};
