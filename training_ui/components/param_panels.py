"""Parameter panel factory functions for training configuration.

Each factory builds a ``gr.Accordion`` containing logically grouped Gradio
components.  Every factory returns a ``dict[str, gr.Component]`` whose keys
correspond to the YAML config key names consumed by ``config_builder.py``.
"""

import gradio as gr


# ---------------------------------------------------------------------------
# Seed & Device
# ---------------------------------------------------------------------------

def build_seed_device_panel() -> dict[str, gr.components.Component]:
	"""Build seed and device configuration panel.

	Returns:
		Dict with keys: seed_specify, seed, cuda, cuda_deterministic,
		torch_threads.
	"""
	with gr.Accordion("Seed & Device", open=False):
		with gr.Row():
			seed_specify = gr.Checkbox(label="Specify Seed", value=True)
			seed = gr.Number(label="Seed", value=12345, precision=0)
		with gr.Row():
			cuda = gr.Checkbox(label="Use CUDA", value=True)
			cuda_deterministic = gr.Checkbox(label="CUDA Deterministic", value=True)
			torch_threads = gr.Slider(
				label="Torch Threads",
				minimum=1,
				maximum=16,
				step=1,
				value=4,
			)
	return {
		"seed_specify": seed_specify,
		"seed": seed,
		"cuda": cuda,
		"cuda_deterministic": cuda_deterministic,
		"torch_threads": torch_threads,
	}


# ---------------------------------------------------------------------------
# Training Settings
# ---------------------------------------------------------------------------

def build_training_panel() -> dict[str, gr.components.Component]:
	"""Build training configuration panel.

	Returns:
		Dict with keys: n_rollout_threads, num_env_steps, episode_length,
		log_interval, eval_interval, save_interval, use_valuenorm,
		use_linear_lr_decay.
	"""
	with gr.Accordion("Training Settings", open=True):
		with gr.Row():
			n_rollout_threads = gr.Slider(
				label="Rollout Threads",
				minimum=1,
				maximum=64,
				step=1,
				value=4,
			)
			num_env_steps = gr.Number(
				label="Total Env Steps",
				value=2000000,
				precision=0,
			)
			episode_length = gr.Number(
				label="Episode Length",
				value=24,
				precision=0,
			)
		with gr.Row():
			log_interval = gr.Number(
				label="Log Interval (episodes)",
				value=5,
				precision=0,
			)
			eval_interval = gr.Number(
				label="Eval Interval (episodes)",
				value=25,
				precision=0,
			)
			save_interval = gr.Number(
				label="Save Interval (episodes)",
				value=50,
				precision=0,
			)
		with gr.Row():
			use_valuenorm = gr.Checkbox(label="Value Normalization", value=True)
			use_linear_lr_decay = gr.Checkbox(label="Linear LR Decay", value=False)
	return {
		"n_rollout_threads": n_rollout_threads,
		"num_env_steps": num_env_steps,
		"episode_length": episode_length,
		"log_interval": log_interval,
		"eval_interval": eval_interval,
		"save_interval": save_interval,
		"use_valuenorm": use_valuenorm,
		"use_linear_lr_decay": use_linear_lr_decay,
	}


# ---------------------------------------------------------------------------
# Network Architecture
# ---------------------------------------------------------------------------

def build_network_panel() -> dict[str, gr.components.Component]:
	"""Build network architecture panel.

	Returns:
		Dict with keys: hidden_sizes, activation_func,
		use_feature_normalization, initialization_method, gain,
		use_recurrent_policy, use_naive_recurrent_policy, recurrent_n,
		data_chunk_length, lr, critic_lr.
	"""
	with gr.Accordion("Network Architecture", open=False):
		with gr.Row():
			hidden_sizes = gr.Textbox(
				label="Hidden Sizes (JSON)",
				value="[128, 128]",
				placeholder='e.g. [64, 64] or [256, 128, 64]',
			)
			activation_func = gr.Dropdown(
				label="Activation",
				choices=["relu", "tanh", "sigmoid", "leaky_relu", "selu"],
				value="relu",
			)
		with gr.Row():
			use_feature_normalization = gr.Checkbox(
				label="Feature Normalization",
				value=True,
			)
			initialization_method = gr.Dropdown(
				label="Init Method",
				choices=["orthogonal_", "xavier_uniform_", "kaiming_uniform_"],
				value="orthogonal_",
			)
			gain = gr.Number(label="Output Gain", value=0.2)
		gr.Markdown("### Recurrent Policy")
		with gr.Row():
			use_recurrent_policy = gr.Checkbox(
				label="Use Recurrent Policy",
				value=True,
			)
			use_naive_recurrent_policy = gr.Checkbox(
				label="Naive Recurrent",
				value=False,
			)
			recurrent_n = gr.Slider(
				label="Recurrent Layers",
				minimum=1,
				maximum=4,
				step=1,
				value=1,
			)
			data_chunk_length = gr.Number(
				label="Data Chunk Length",
				value=60,
				precision=0,
			)
		gr.Markdown("### Learning Rate")
		with gr.Row():
			lr = gr.Number(label="Actor LR", value=0.0001)
			critic_lr = gr.Number(label="Critic LR", value=0.0003)
	return {
		"hidden_sizes": hidden_sizes,
		"activation_func": activation_func,
		"use_feature_normalization": use_feature_normalization,
		"initialization_method": initialization_method,
		"gain": gain,
		"use_recurrent_policy": use_recurrent_policy,
		"use_naive_recurrent_policy": use_naive_recurrent_policy,
		"recurrent_n": recurrent_n,
		"data_chunk_length": data_chunk_length,
		"lr": lr,
		"critic_lr": critic_lr,
	}


# ---------------------------------------------------------------------------
# On-Policy Algorithm (PPO / HAPPO family)
# ---------------------------------------------------------------------------

def build_on_policy_algo_panel() -> dict[str, gr.components.Component]:
	"""Build on-policy algorithm specific panel (PPO/HAPPO).

	Returns:
		Dict with keys: _panel, ppo_epoch, critic_epoch, clip_param,
		entropy_coef, value_loss_coef, use_clipped_value_loss,
		use_max_grad_norm, max_grad_norm, use_gae, gamma, gae_lambda,
		use_huber_loss, huber_delta, action_aggregation, share_param,
		fixed_order, actor_num_mini_batch, critic_num_mini_batch.
	"""
	with gr.Accordion("On-Policy Algorithm (PPO/HAPPO)", open=False) as panel:
		with gr.Row():
			ppo_epoch = gr.Slider(
				label="PPO Epoch",
				minimum=1,
				maximum=30,
				step=1,
				value=5,
			)
			critic_epoch = gr.Slider(
				label="Critic Epoch",
				minimum=1,
				maximum=30,
				step=1,
				value=5,
			)
			clip_param = gr.Slider(
				label="Clip Param",
				minimum=0.05,
				maximum=0.5,
				step=0.01,
				value=0.25,
			)
		with gr.Row():
			entropy_coef = gr.Slider(
				label="Entropy Coef",
				minimum=0.0,
				maximum=0.5,
				step=0.005,
				value=0.08,
			)
			value_loss_coef = gr.Slider(
				label="Value Loss Coef",
				minimum=0.1,
				maximum=5.0,
				step=0.1,
				value=1.0,
			)
			max_grad_norm = gr.Number(label="Max Grad Norm", value=3.0)
		with gr.Row():
			gamma = gr.Slider(
				label="Gamma",
				minimum=0.9,
				maximum=1.0,
				step=0.001,
				value=0.99,
			)
			gae_lambda = gr.Slider(
				label="GAE Lambda",
				minimum=0.8,
				maximum=1.0,
				step=0.01,
				value=0.95,
			)
		with gr.Row():
			actor_num_mini_batch = gr.Slider(
				label="Actor Mini Batch",
				minimum=1,
				maximum=32,
				step=1,
				value=4,
			)
			critic_num_mini_batch = gr.Slider(
				label="Critic Mini Batch",
				minimum=1,
				maximum=32,
				step=1,
				value=4,
			)
		with gr.Row():
			use_clipped_value_loss = gr.Checkbox(
				label="Clipped Value Loss",
				value=True,
			)
			use_max_grad_norm = gr.Checkbox(
				label="Use Max Grad Norm",
				value=True,
			)
			use_gae = gr.Checkbox(label="Use GAE", value=True)
			use_huber_loss = gr.Checkbox(label="Use Huber Loss", value=True)
		with gr.Row():
			huber_delta = gr.Number(label="Huber Delta", value=10.0)
			action_aggregation = gr.Dropdown(
				label="Action Aggregation",
				choices=["prod", "mean"],
				value="prod",
			)
		with gr.Row():
			share_param = gr.Checkbox(label="Share Parameters", value=False)
			fixed_order = gr.Checkbox(label="Fixed Order", value=False)
	return {
		"_panel": panel,
		"ppo_epoch": ppo_epoch,
		"critic_epoch": critic_epoch,
		"clip_param": clip_param,
		"entropy_coef": entropy_coef,
		"value_loss_coef": value_loss_coef,
		"use_clipped_value_loss": use_clipped_value_loss,
		"use_max_grad_norm": use_max_grad_norm,
		"max_grad_norm": max_grad_norm,
		"use_gae": use_gae,
		"gamma": gamma,
		"gae_lambda": gae_lambda,
		"use_huber_loss": use_huber_loss,
		"huber_delta": huber_delta,
		"action_aggregation": action_aggregation,
		"share_param": share_param,
		"fixed_order": fixed_order,
		"actor_num_mini_batch": actor_num_mini_batch,
		"critic_num_mini_batch": critic_num_mini_batch,
	}


# ---------------------------------------------------------------------------
# Off-Policy Algorithm
# ---------------------------------------------------------------------------

def build_off_policy_algo_panel() -> dict[str, gr.components.Component]:
	"""Build off-policy algorithm specific panel.

	The accordion is hidden by default and should be toggled visible when an
	off-policy algorithm family is selected.

	Returns:
		Dict with keys: _panel, buffer_size, batch_size, polyak, n_step,
		warmup_steps, train_interval, update_per_train, gamma, auto_alpha,
		alpha, alpha_lr, use_huber_loss, huber_delta, share_param,
		fixed_order.
	"""
	with gr.Accordion("Off-Policy Algorithm", open=False, visible=False) as panel:
		with gr.Row():
			buffer_size = gr.Number(
				label="Buffer Size",
				value=10800,
				precision=0,
			)
			batch_size = gr.Number(
				label="Batch Size",
				value=24,
				precision=0,
			)
			polyak = gr.Slider(
				label="Polyak (Soft Update)",
				minimum=0.001,
				maximum=0.1,
				step=0.001,
				value=0.005,
			)
		with gr.Row():
			n_step = gr.Slider(
				label="N-Step Returns",
				minimum=1,
				maximum=50,
				step=1,
				value=20,
			)
			warmup_steps = gr.Number(
				label="Warmup Steps",
				value=1200,
				precision=0,
			)
		with gr.Row():
			train_interval = gr.Number(
				label="Train Interval",
				value=24,
				precision=0,
			)
			update_per_train = gr.Number(
				label="Updates Per Train",
				value=1,
				precision=0,
			)
		with gr.Row():
			gamma = gr.Slider(
				label="Gamma",
				minimum=0.9,
				maximum=1.0,
				step=0.001,
				value=0.99,
			)
		gr.Markdown("### SAC Temperature")
		with gr.Row():
			auto_alpha = gr.Checkbox(label="Auto Alpha", value=False)
			alpha = gr.Number(label="Alpha", value=0.001)
			alpha_lr = gr.Number(label="Alpha LR", value=0.0003)
		with gr.Row():
			use_huber_loss = gr.Checkbox(label="Huber Loss", value=True)
			huber_delta = gr.Number(label="Huber Delta", value=10.0)
		with gr.Row():
			share_param = gr.Checkbox(label="Share Parameters", value=False)
			fixed_order = gr.Checkbox(label="Fixed Order", value=False)
	return {
		"_panel": panel,
		"buffer_size": buffer_size,
		"batch_size": batch_size,
		"polyak": polyak,
		"n_step": n_step,
		"warmup_steps": warmup_steps,
		"train_interval": train_interval,
		"update_per_train": update_per_train,
		"gamma": gamma,
		"auto_alpha": auto_alpha,
		"alpha": alpha,
		"alpha_lr": alpha_lr,
		"use_huber_loss": use_huber_loss,
		"huber_delta": huber_delta,
		"share_param": share_param,
		"fixed_order": fixed_order,
	}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_eval_panel() -> dict[str, gr.components.Component]:
	"""Build evaluation configuration panel.

	Returns:
		Dict with keys: use_eval, n_eval_rollout_threads, eval_episodes.
	"""
	with gr.Accordion("Evaluation", open=False):
		use_eval = gr.Checkbox(label="Enable Evaluation", value=True)
		with gr.Row():
			n_eval_rollout_threads = gr.Slider(
				label="Eval Threads",
				minimum=1,
				maximum=8,
				step=1,
				value=2,
			)
			eval_episodes = gr.Number(
				label="Eval Episodes",
				value=10,
				precision=0,
			)
	return {
		"use_eval": use_eval,
		"n_eval_rollout_threads": n_eval_rollout_threads,
		"eval_episodes": eval_episodes,
	}
