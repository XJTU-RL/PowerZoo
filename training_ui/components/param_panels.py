"""Parameter panel factory functions for training configuration.

Each factory builds a ``gr.Accordion`` containing logically grouped Gradio
components.  Every factory returns a ``dict[str, gr.Component]`` whose keys
correspond to the YAML config key names consumed by ``config_builder.py``.
"""

import gradio as gr

from training_ui.i18n import t


# ---------------------------------------------------------------------------
# Seed & Device
# ---------------------------------------------------------------------------

def build_seed_device_panel() -> dict[str, gr.components.Component]:
	"""Build seed and device configuration panel.

	Returns:
		Dict with keys: seed_specify, seed, cuda, cuda_deterministic,
		torch_threads.
	"""
	with gr.Accordion(t("accordion_seed_device"), open=False):
		with gr.Row():
			seed_specify = gr.Checkbox(label=t("label_specify_seed"), value=True)
			seed = gr.Number(label=t("label_seed"), value=12345, precision=0)
		with gr.Row():
			cuda = gr.Checkbox(label=t("label_use_cuda"), value=True)
			cuda_deterministic = gr.Checkbox(label=t("label_cuda_deterministic"), value=True)
			torch_threads = gr.Slider(
				label=t("label_torch_threads"),
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
	with gr.Accordion(t("accordion_training_settings"), open=True):
		with gr.Row():
			n_rollout_threads = gr.Slider(
				label=t("label_rollout_threads"),
				minimum=1,
				maximum=64,
				step=1,
				value=4,
			)
			num_env_steps = gr.Number(
				label=t("label_total_env_steps"),
				value=2000000,
				precision=0,
			)
			episode_length = gr.Number(
				label=t("label_episode_length"),
				value=24,
				precision=0,
			)
		with gr.Row():
			log_interval = gr.Number(
				label=t("label_log_interval"),
				value=5,
				precision=0,
			)
			eval_interval = gr.Number(
				label=t("label_eval_interval"),
				value=25,
				precision=0,
			)
			save_interval = gr.Number(
				label=t("label_save_interval"),
				value=50,
				precision=0,
			)
		with gr.Row():
			use_valuenorm = gr.Checkbox(label=t("label_value_norm"), value=True)
			use_linear_lr_decay = gr.Checkbox(label=t("label_linear_lr_decay"), value=False)
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
	with gr.Accordion(t("accordion_network"), open=False):
		with gr.Row():
			hidden_sizes = gr.Textbox(
				label=t("label_hidden_sizes"),
				value="[128, 128]",
				placeholder='e.g. [64, 64] or [256, 128, 64]',
			)
			activation_func = gr.Dropdown(
				label=t("label_activation"),
				choices=["relu", "tanh", "sigmoid", "leaky_relu", "selu"],
				value="relu",
			)
		with gr.Row():
			use_feature_normalization = gr.Checkbox(
				label=t("label_feature_norm"),
				value=True,
			)
			initialization_method = gr.Dropdown(
				label=t("label_init_method"),
				choices=["orthogonal_", "xavier_uniform_", "kaiming_uniform_"],
				value="orthogonal_",
			)
			gain = gr.Number(label=t("label_output_gain"), value=0.2)
		gr.Markdown(t("heading_recurrent_policy"))
		with gr.Row():
			use_recurrent_policy = gr.Checkbox(
				label=t("label_use_recurrent"),
				value=True,
			)
			use_naive_recurrent_policy = gr.Checkbox(
				label=t("label_naive_recurrent"),
				value=False,
			)
			recurrent_n = gr.Slider(
				label=t("label_recurrent_layers"),
				minimum=1,
				maximum=4,
				step=1,
				value=1,
			)
			data_chunk_length = gr.Number(
				label=t("label_data_chunk_length"),
				value=60,
				precision=0,
			)
		gr.Markdown(t("heading_learning_rate"))
		with gr.Row():
			lr = gr.Number(label=t("label_actor_lr"), value=0.0001)
			critic_lr = gr.Number(label=t("label_critic_lr"), value=0.0003)
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
	with gr.Accordion(t("accordion_on_policy"), open=False) as panel:
		with gr.Row():
			ppo_epoch = gr.Slider(
				label=t("label_ppo_epoch"),
				minimum=1,
				maximum=30,
				step=1,
				value=5,
			)
			critic_epoch = gr.Slider(
				label=t("label_critic_epoch"),
				minimum=1,
				maximum=30,
				step=1,
				value=5,
			)
			clip_param = gr.Slider(
				label=t("label_clip_param"),
				minimum=0.05,
				maximum=0.5,
				step=0.01,
				value=0.25,
			)
		with gr.Row():
			entropy_coef = gr.Slider(
				label=t("label_entropy_coef"),
				minimum=0.0,
				maximum=0.5,
				step=0.005,
				value=0.08,
			)
			value_loss_coef = gr.Slider(
				label=t("label_value_loss_coef"),
				minimum=0.1,
				maximum=5.0,
				step=0.1,
				value=1.0,
			)
			max_grad_norm = gr.Number(label=t("label_max_grad_norm"), value=3.0)
		with gr.Row():
			gamma = gr.Slider(
				label=t("label_gamma"),
				minimum=0.9,
				maximum=1.0,
				step=0.001,
				value=0.99,
			)
			gae_lambda = gr.Slider(
				label=t("label_gae_lambda"),
				minimum=0.8,
				maximum=1.0,
				step=0.01,
				value=0.95,
			)
		with gr.Row():
			actor_num_mini_batch = gr.Slider(
				label=t("label_actor_mini_batch"),
				minimum=1,
				maximum=32,
				step=1,
				value=4,
			)
			critic_num_mini_batch = gr.Slider(
				label=t("label_critic_mini_batch"),
				minimum=1,
				maximum=32,
				step=1,
				value=4,
			)
		with gr.Row():
			use_clipped_value_loss = gr.Checkbox(
				label=t("label_clipped_value_loss"),
				value=True,
			)
			use_max_grad_norm = gr.Checkbox(
				label=t("label_use_max_grad_norm"),
				value=True,
			)
			use_gae = gr.Checkbox(label=t("label_use_gae"), value=True)
			use_huber_loss = gr.Checkbox(label=t("label_use_huber_loss"), value=True)
		with gr.Row():
			huber_delta = gr.Number(label=t("label_huber_delta"), value=10.0)
			action_aggregation = gr.Dropdown(
				label=t("label_action_aggregation"),
				choices=["prod", "mean"],
				value="prod",
			)
		with gr.Row():
			share_param = gr.Checkbox(label=t("label_share_param"), value=False)
			fixed_order = gr.Checkbox(label=t("label_fixed_order"), value=False)
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
	with gr.Accordion(t("accordion_off_policy"), open=False, visible=False) as panel:
		with gr.Row():
			buffer_size = gr.Number(
				label=t("label_buffer_size"),
				value=10800,
				precision=0,
			)
			batch_size = gr.Number(
				label=t("label_batch_size"),
				value=24,
				precision=0,
			)
			polyak = gr.Slider(
				label=t("label_polyak"),
				minimum=0.001,
				maximum=0.1,
				step=0.001,
				value=0.005,
			)
		with gr.Row():
			n_step = gr.Slider(
				label=t("label_n_step"),
				minimum=1,
				maximum=50,
				step=1,
				value=20,
			)
			warmup_steps = gr.Number(
				label=t("label_warmup_steps"),
				value=1200,
				precision=0,
			)
		with gr.Row():
			train_interval = gr.Number(
				label=t("label_train_interval"),
				value=24,
				precision=0,
			)
			update_per_train = gr.Number(
				label=t("label_updates_per_train"),
				value=1,
				precision=0,
			)
		with gr.Row():
			gamma = gr.Slider(
				label=t("label_gamma"),
				minimum=0.9,
				maximum=1.0,
				step=0.001,
				value=0.99,
			)
		gr.Markdown(t("heading_sac_temp"))
		with gr.Row():
			auto_alpha = gr.Checkbox(label=t("label_auto_alpha"), value=False)
			alpha = gr.Number(label=t("label_alpha"), value=0.001)
			alpha_lr = gr.Number(label=t("label_alpha_lr"), value=0.0003)
		with gr.Row():
			use_huber_loss = gr.Checkbox(label=t("label_huber_loss"), value=True)
			huber_delta = gr.Number(label=t("label_huber_delta"), value=10.0)
		with gr.Row():
			share_param = gr.Checkbox(label=t("label_share_param"), value=False)
			fixed_order = gr.Checkbox(label=t("label_fixed_order"), value=False)
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
	with gr.Accordion(t("accordion_eval"), open=False):
		use_eval = gr.Checkbox(label=t("label_enable_eval"), value=True)
		with gr.Row():
			n_eval_rollout_threads = gr.Slider(
				label=t("label_eval_threads"),
				minimum=1,
				maximum=8,
				step=1,
				value=2,
			)
			eval_episodes = gr.Number(
				label=t("label_eval_episodes"),
				value=10,
				precision=0,
			)
	return {
		"use_eval": use_eval,
		"n_eval_rollout_threads": n_eval_rollout_threads,
		"eval_episodes": eval_episodes,
	}
