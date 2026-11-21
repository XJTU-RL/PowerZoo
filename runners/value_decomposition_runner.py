"""QMIX and VDN on the common VVC vector environment and operational logger."""

import json
from pathlib import Path

import numpy as np
import torch

from algorithms.actors.discrete_value_decomposition import DiscreteValueDecomposition
from envs import LOGGER_REGISTRY
from runners.value_decomposition_replay import EpisodeReplay, ReplayFrame, Transition
from utils.configs_tools import init_dir, save_config
from utils.discrete_marl_config import require_discrete_spaces, validate_discrete_vvc
from utils.envs_tools import make_eval_env, make_train_env, set_seed
from utils.models_tools import init_device
from utils.physics_manifest import validate_checkpoint


class ValueDecompositionRunner:
    """One discrete Q policy per physical device and one team TD target per transition."""

    def __init__(self, args, algo_args, env_args):
        validate_discrete_vvc(args["algo"], args["env"], algo_args, env_args)
        self.args = args
        self.train_args, self.eval_args = algo_args["train"], algo_args["eval"]
        threads = self.train_args["n_rollout_threads"]
        set_seed(algo_args["seed"])
        seed = algo_args["seed"]["seed"]
        self.device = init_device(algo_args["device"])
        self.envs = make_train_env(args["env"], seed, threads, env_args)
        self.eval_envs = None
        try:
            self._setup(env_args, algo_args)
        except BaseException:
            self.close()
            raise

    def _setup(self, env_args, algo_args):
        env = self.envs
        spaces = env.action_space
        require_discrete_spaces(spaces)
        self.num_agents = len(spaces)
        self.settings = settings = algo_args["algo"]
        self.learner = DiscreteValueDecomposition(
            self.args["algo"],
            algo_args["model"],
            settings,
            env.observation_space,
            spaces,
            env.share_observation_space[0].shape[0],
            self.device,
        )
        self.replay = EpisodeReplay(
            settings["buffer_size"], self.train_args["n_rollout_threads"]
        )
        self.total_steps = self.total_updates = 0
        self.reward_sum = self.reward_count = 0
        self.physics_manifest = env.get_physics_manifest()
        self.run_dir, self.log_dir, self.save_dir, self.writer = init_dir(
            self.args["env"],
            env_args,
            self.args["algo"],
            self.args["exp_name"],
            algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(self.args, algo_args, env_args, self.run_dir)
        (Path(self.run_dir) / "physics.manifest.json").write_text(
            json.dumps(
                self.physics_manifest, ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
        self.logger = LOGGER_REGISTRY[self.args["env"]](
            self.args, algo_args, env_args, self.num_agents, self.writer, self.run_dir
        )
        if self.train_args["model_dir"]:
            self.restore(self.train_args["model_dir"])
        if self.eval_args["use_eval"]:
            self.eval_envs = make_eval_env(
                self.args["env"],
                algo_args["seed"]["seed"],
                self.eval_args["n_eval_rollout_threads"],
                env_args,
            )

    def predict(self, obs, masks, *, explore=False):
        step = self.total_steps if explore else None
        return self.learner.predict(obs, masks, exploration_step=step)

    def _insert(self, frame, actions, result):
        obs, state, available = frame
        next_obs, next_state, reward, done, infos, next_available = result
        if not np.allclose(reward, reward[:, :1], rtol=1e-6, atol=1e-7):
            raise ValueError("QMIX/VDN requires a shared team reward")
        if not np.all(done == done[:, :1]):
            raise ValueError("QMIX/VDN requires simultaneous device termination")
        for worker, (ended, worker_infos) in enumerate(zip(done[:, 0], infos)):
            info = worker_infos[0]
            if (info.get("terminated") or info.get("physical_failure")) and not ended:
                raise ValueError("A terminal event must end the transition")
            truncated = info.get("TimeLimit.truncated") and not (
                info.get("terminated") or info.get("physical_failure")
            )
            current = ReplayFrame.from_worker(
                obs[worker], state[worker], available[worker]
            )
            if ended:
                following = ReplayFrame.from_worker(
                    info["original_obs"],
                    info["original_state"],
                    info["original_avail_actions"],
                )
            else:
                following = ReplayFrame.from_worker(
                    next_obs[worker], next_state[worker], next_available[worker]
                )
            self.replay.append(
                worker,
                Transition(
                    current,
                    actions[worker, :, 0].copy(),
                    float(reward[worker, 0, 0]),
                    following,
                    bool(ended and not truncated),
                ),
                ended=ended,
            )
        self.reward_sum += float(reward[:, 0, 0].sum())
        self.reward_count += len(reward)

    def run(self):
        frame = self.envs.reset()
        args = self.train_args
        steps_per_rollout = args["episode_length"] * args["n_rollout_threads"]
        rollouts = args["num_env_steps"] // steps_per_rollout
        self.logger.init(rollouts)
        update_info = {}
        for rollout in range(1, rollouts + 1):
            self.logger.episode_init(rollout)
            for _ in range(args["episode_length"]):
                actions = self.predict(frame[0], frame[2], explore=True)
                result = self.envs.step(actions)
                self._insert(frame, actions, result)
                self.logger.per_step((*result, None, actions, None, None, None))
                frame = result[0], result[1], result[5]
                self.total_steps += args["n_rollout_threads"]
                vector_steps = self.total_steps // args["n_rollout_threads"]
                if (
                    vector_steps % args["train_interval"] == 0
                    and self.replay.size >= self.settings["batch_size"]
                ):
                    update_info = self.learner.train_batch(
                        self.replay.sample(self.settings["batch_size"])
                    )
                    self.total_updates += 1
            if rollout % args["log_interval"] == 0:
                self.logger.episode_log(
                    [{} for _ in range(self.num_agents)],
                    {
                        **update_info,
                        "updates": self.total_updates,
                    },
                    None,
                    self,
                )
                self.reward_sum = self.reward_count = 0
            if self.eval_envs is not None and rollout % args["eval_interval"] == 0:
                self.eval(rollout)
            if rollout % args["save_interval"] == 0:
                self.save(Path(self.save_dir) / f"checkpoint_episode_{rollout}")
        self.save(Path(self.save_dir))

    def get_mean_rewards(self):
        return self.reward_sum / self.reward_count

    def eval(self, rollout):
        self.logger.eval_init()
        obs, _, masks = self.eval_envs.reset()
        completed = 0
        while completed < self.eval_args["eval_episodes"]:
            result = self.eval_envs.step(self.predict(obs, masks))
            self.logger.eval_per_step(result)
            obs, masks = result[0], result[5]
            for worker in np.flatnonzero(result[3].all(axis=1)):
                if completed < self.eval_args["eval_episodes"]:
                    self.logger.eval_thread_done(worker)
                    completed += 1
        self.logger.eval_log(rollout)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "value_decomposition.pt"
        temporary = destination.with_suffix(".tmp")
        model = self.learner
        torch.save(
            {
                "algo": self.args["algo"],
                "policies": model.policies.state_dict(),
                "target_policies": model.target_policies.state_dict(),
                "mixer": model.mixer.state_dict(),
                "target_mixer": model.target_mixer.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "steps": self.total_steps,
                "updates": self.total_updates,
            },
            temporary,
        )
        temporary.replace(destination)

    def restore(self, directory):
        validate_checkpoint(directory, self.physics_manifest)
        checkpoint = torch.load(
            Path(directory) / "value_decomposition.pt",
            map_location=self.device,
            weights_only=True,
        )
        if checkpoint["algo"] != self.args["algo"]:
            raise ValueError(
                "Checkpoint uses a different value decomposition algorithm"
            )
        model = self.learner
        model.policies.load_state_dict(checkpoint["policies"])
        model.target_policies.load_state_dict(checkpoint["target_policies"])
        model.mixer.load_state_dict(checkpoint["mixer"])
        model.target_mixer.load_state_dict(checkpoint["target_mixer"])
        model.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps, self.total_updates = (
            checkpoint["steps"],
            checkpoint["updates"],
        )

    def get_result(self):
        return self.logger.get_result()

    def close(self):
        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()
        if hasattr(self, "logger"):
            self.logger.close()
        if hasattr(self, "writer"):
            self.writer.export_scalars_to_json(str(Path(self.log_dir) / "summary.json"))
            self.writer.close()
