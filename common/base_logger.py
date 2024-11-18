"""Base logger."""

import time
import os
import numpy as np


class BaseLogger:
    """Base logger class.
    Used for logging information in the on-policy training pipeline.用于记录在基于策略的训练流程中的信息
    """

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize the logger."""
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.writter = writter
        self.run_dir = run_dir
  
        # 打开一个文件，用于记录训练进度
        self.log_file = open(
            os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8"#join的作用是创建文件路径
        )
        
        # 打开一个文件，用于记录训练信息
        self.log_training_info = open(
            os.path.join(run_dir, "train_info.txt"), "w", encoding="utf-8"
        )
        # 打开一个文件，用于记录评估信息
        self.log_eval_info = open(
            os.path.join(run_dir, "eval_info.txt"), "w", encoding="utf-8"
        )

        # 初始化一个空字符串
        text = ""
        # 打印算法参数
        print(algo_args)
        # 遍历算法参数
        for section, params in algo_args.items():
            # 将section添加到text中
            text += f"{section}:\n"
            # 遍历params中的key和value
            for key, value in params.items():
                # 如果value是列表，则将其替换为...
                if isinstance(value, list):
                    value = "..."
                # 将key和value添加到text中
                text += f"\t{key}: {value}\n"+ '\n'

        # 将文本添加到 self.writter 中
        self.writter.add_text("algo_hyperparameters", text)
        text = ""
        for key, value in env_args.items():
            text += f"{key}: {value}\n" + '\n'

        # 将文本添加到 self.writter 中
        self.writter.add_text("env_parameters", text)

    def get_task_name(self):
        """Get the task name."""
        raise NotImplementedError

    def init(self, episodes):
        """Initialize the logger."""
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )#意思是algo_args["train"]类下n_rollout_threads的值，此处创建了一个大小为参数cogfig中设置的训练线程数量的rewards的值，用于存放每个线程的episode_rewards.
        self.done_episodes_rewards = []

    def episode_init(self, episode):
        """Initialize the logger for each episode."""
        self.episode = episode

    def per_step(self, data):
        """Process data per step."""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        #print("log_dones:",dones)#TODO:打印dones
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()#rewards 是一个包含多个环境的奖励值的数组（单步），其中每行代表一个环境的奖励序列,对每行求平均（此处每行只有一个元素）
        self.train_episode_rewards += reward_env#将单步奖励求和，变成episode奖励
        for t in range(self.algo_args["train"]["n_rollout_threads"]):#这个循环变的是episode的数量
            if dones_env[t]:#如果某个线程中一个episode结束了
                self.done_episodes_rewards.append(self.train_episode_rewards[t])#用于存放每一个环境完成episode的奖励值以列表的形式（没有对多线程中的episode奖励求平均），其中self.train_episode_rewards[t]用于存放第t个线程的一个episode的累计奖励
                self.train_episode_rewards[t] = 0

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "环境： {} 任务 {} 算法 {} 实验名称 {} updates {}/{} episodes, 总时间步数 {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.done_episodes_rewards = []

    def eval_init(self):
        """Initialize the logger for evaluation."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    def eval_thread_done(self, tid):
        """Log evaluation information."""
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        self.one_episode_rewards[tid] = []

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        )
        self.log_file.flush()

    def log_train(self, actor_train_infos, critic_train_info):
        """Log training information."""
        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, self.total_num_steps)
        # log critic
        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            self.writter.add_scalars(critic_k, {critic_k: v}, self.total_num_steps)

    def log_env(self, env_infos):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)

    def close(self):
        """Close the logger."""
        self.log_file.close()
