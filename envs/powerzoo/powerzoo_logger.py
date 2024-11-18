from common.base_logger import BaseLogger
import time
from textwrap import dedent
import numpy as np

class PowerZooLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(PowerZooLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        

    def get_task_name(self):
        return self.env_args["env_name"]
    
    def init(self, episodes):
        """初始化记录器。

        Args:
            episodes (int): 剧集数量。
        """
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        
        #TODO:分离奖励log,powerloss_reward
        self.train_episode_powerloss_reward = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_powerloss_reward = []
        
        #TODO:分离奖励log,ctrl_reward
        self.train_episode_ctrl_reward = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_ctrl_reward = []
        
        #TODO:分离奖励log,voltage_reward
        self.train_episode_voltage_reward = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_voltage_reward = []

    def episode_init(self, episode):
        """初始化每个episode的记录器。"""
        self.episode = episode

    def per_step(self, data):
        """处理每步的数据。

        Args:
            data (tuple): 每步的数据。包含以下元素：
                - obs (np.ndarray): 观测值。
                - share_obs (np.ndarray): 共享观测值。
                - rewards (np.ndarray): 奖励。
                - dones (np.ndarray): 是否完成。
                - infos (list): 其他信息。
                - available_actions (np.ndarray): 可用动作。
                - values (np.ndarray): 价值。
                - actions (np.ndarray): 动作。
                - action_log_probs (np.ndarray): 动作对数概率。
                - rnn_states (np.ndarray): RNN状态。
                - rnn_states_critic (np.ndarray): RNN状态（critic）。
        """
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
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()

        # 解析 powerloss_reward
        powerloss_reward = [
            [d[0].get('power_loss_ratio', 0) if isinstance(d, list) and len(d) > 0 and d[0] else 0 for d in infos]
        ]
        powerloss_reward_cleaned = powerloss_reward if powerloss_reward else [[0]]
        powerloss_reward_env = np.mean(powerloss_reward_cleaned, axis=1).flatten()

        # 处理其他奖励（类似处理方式）
        voltage_reward = [
            [d[0].get('vol_reward', 0) if isinstance(d, list) and len(d) > 0 and d[0] else 0 for d in infos]
        ]
        voltage_reward_cleaned = voltage_reward if voltage_reward else [[0]]
        voltage_reward_env = np.mean(voltage_reward_cleaned, axis=1).flatten()

        ctrl_reward = [
            [d[0].get('ctrl_reward', 0) if isinstance(d, list) and len(d) > 0 and d[0] else 0 for d in infos]
        ]
        ctrl_reward_cleaned = ctrl_reward if ctrl_reward else [[0]]
        ctrl_reward_env = np.mean(ctrl_reward_cleaned, axis=1).flatten()

        # 更新训练奖励
        self.train_episode_rewards += reward_env
        self.train_episode_powerloss_reward += powerloss_reward_env
        self.train_episode_voltage_reward += voltage_reward_env
        self.train_episode_ctrl_reward += ctrl_reward_env

        # 更新完成的 episode 奖励
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0

                # 更新 power_loss 奖励
                self.done_episodes_powerloss_reward.append(self.train_episode_powerloss_reward[t] / 24)
                self.train_episode_powerloss_reward[t] = 0

                # 更新 voltage 奖励
                self.done_episodes_voltage_reward.append(self.train_episode_voltage_reward[t])
                self.train_episode_voltage_reward[t] = 0

                # 更新 ctrl 奖励
                self.done_episodes_ctrl_reward.append(self.train_episode_ctrl_reward[t])
                self.train_episode_ctrl_reward[t] = 0

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """记录episode的日志信息。

        Args:
            actor_train_infos (_type_): actor 训练信息
            critic_train_info (_type_): critic 训练信息
            actor_buffer (_type_): actor 缓冲区
            critic_buffer (_type_): critic 缓冲区
        """
        # 计算总步数
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        # 记录结束时间
        self.end = time.time()
        
        # 计算平均步数奖励
        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        training_info = dedent(f"""
            环境：{self.args["env"]} 网络: {self.task_name} 算法: {self.args["algo"]} 实验名称: {self.args["exp_name"]} 
            更新次数 {self.episode}/{self.episodes} episodes,总时间步数 {self.total_num_steps}/{self.algo_args["train"]["num_env_steps"]}, FPS {int(self.total_num_steps / (self.end - self.start))}. 
            平均步数奖励为 {critic_train_info["average_step_rewards"]}.\n""")
        
        # 打印日志信息
        print(training_info)
        self.log_training_info.write(training_info)
        self.log_training_info.flush()

        # 记录平均奖励值
        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            #TODO:power_loss,voltage_loss,ctrl_loss
            aver_episode_powerloss_rewards = np.mean(self.done_episodes_powerloss_reward)
            aver_episode_voltage_rewards = np.mean(self.done_episodes_voltage_reward)
            aver_episode_ctrl_rewards = np.mean(self.done_episodes_ctrl_reward)
            
            log_info = f"平均奖励值为： {aver_episode_rewards}.\n"
            print(log_info)
            self.log_training_info.write(log_info)
            self.log_training_info.flush()

            # 记录到tensorboard
            self.writter.add_scalars(
                "train_episode_rewards",#TODO:
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            
            self.writter.add_scalars(
                "train_episode_powerloss_rewards",#TODO:
                {"aver_rewards": aver_episode_powerloss_rewards},
                self.total_num_steps,
            )
            
            self.writter.add_scalars(
                "train_episode_voltage_rewards",#TODO:
                {"aver_rewards": aver_episode_voltage_rewards},
                self.total_num_steps,
            )
            
            self.writter.add_scalars(
                "train_episode_ctrl_rewards",#TODO:
                {"aver_rewards": aver_episode_ctrl_rewards},
                self.total_num_steps,
            )
            
            self.done_episodes_rewards = []
            #TODO:power_loss,voltage_loss,ctrl_loss
            self.done_episodes_powerloss_reward = []
            self.done_episodes_voltage_reward=[]
            self.done_episodes_ctrl_reward = []
            
  
    def eval_init(self):
        """初始化评估过程."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.eval_env_reward_infos = {} #记录评估环境的奖励信息
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        
        #TODO:分离eval的三个奖励，powerloss
        self.eval_powerloss_episode_rewards = []
        self.one_powerloss_episode_rewards = []
        #TODO:分离eval的三个奖励，voltageloss
        self.eval_voltage_episode_rewards = []
        self.one_voltage_episode_rewards = []
        #TODO:分离eval的三个奖励，ctrlloss
        self.eval_ctrl_episode_rewards = []
        self.one_ctrl_episode_rewards = []
        


        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])
            #TODO:分离eval的三个奖励
            self.one_powerloss_episode_rewards.append([])
            self.eval_powerloss_episode_rewards.append([])
            
            self.one_voltage_episode_rewards.append([])
            self.eval_voltage_episode_rewards.append([])
            
            self.one_ctrl_episode_rewards.append([])
            self.eval_ctrl_episode_rewards.append([])
            
            
            
    def eval_per_step(self, eval_data):
        """记录评估过程中的奖励.

        Args:
            eval_data (tuple): (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions)
        """
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,#info包含信息：
            eval_available_actions,
        ) = eval_data
        
        #TODO:分离reward,powerloss
        eval_powerloss_reward=[[[d[0]['power_loss_ratio'] for d in eval_infos]]]
        eval_voltage_reward=[[[d[0]['vol_reward'] for d in eval_infos]]]
        eval_ctrl_reward=[[[d[0]['ctrl_reward'] for d in eval_infos]]]
        
        
        # 计算评估环境中的功率损耗奖励的平均值，并将其展平为一维数组
        eval_powerloss_reward_env= np.mean(eval_powerloss_reward, axis=1).flatten()
        # 将展平的一维数组重新调整为 (self.algo_args["eval"]["n_eval_rollout_threads"],1,1) 的形状
        eval_powerloss_reward_env=eval_powerloss_reward_env.reshape((self.algo_args["eval"]["n_eval_rollout_threads"],1,1))
        
        # 计算评估环境中的电压奖励的平均值，并将其展平为一维数组
        eval_voltage_reward_env= np.mean(eval_voltage_reward, axis=1).flatten()
        # 将展平的一维数组重新调整为 (self.algo_args["eval"]["n_eval_rollout_threads"],1,1) 的形状
        eval_voltage_reward_env=eval_voltage_reward_env.reshape((self.algo_args["eval"]["n_eval_rollout_threads"],1,1))
        
        # 计算评估环境中的控制奖励的平均值，并将其展平为一维数组
        eval_ctrl_reward_env= np.mean(eval_ctrl_reward, axis=1).flatten()
        # 将展平的一维数组重新调整为 (self.algo_args["eval"]["n_eval_rollout_threads"],1,1) 的形状
        eval_ctrl_reward_env=eval_ctrl_reward_env.reshape((self.algo_args["eval"]["n_eval_rollout_threads"],1,1))
        
        #将每个线程的reward、powerloss、voltage、ctrl reward分别存储到对应的列表中
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
            self.one_powerloss_episode_rewards[eval_i].append(eval_powerloss_reward_env[eval_i])#TODO:POWERLOSS_EVAL
            self.one_voltage_episode_rewards[eval_i].append(eval_voltage_reward_env[eval_i])
            self.one_ctrl_episode_rewards[eval_i].append(eval_ctrl_reward_env[eval_i])
            
        # 将eval_infos赋值给self.eval_infos
        self.eval_infos = eval_infos

        
    def eval_thread_done(self, tid):
        """记录每个评估线程的结束信息.
        Args:
            tid (int): 线程ID
        """
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        self.eval_powerloss_episode_rewards[tid].append(
            np.sum(self.one_powerloss_episode_rewards[tid], axis=0)/24
        )
        
        self.eval_voltage_episode_rewards[tid].append(
            np.sum(self.one_voltage_episode_rewards[tid], axis=0)
        )
        
        self.eval_ctrl_episode_rewards[tid].append(
            np.sum(self.one_ctrl_episode_rewards[tid], axis=0)
        )
        self.one_episode_rewards[tid] = []
        self.one_powerloss_episode_rewards[tid] = []
        self.one_voltage_episode_rewards[tid] = []
        self.one_ctrl_episode_rewards[tid] = []
        
    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        
        self.eval_powerloss_episode_rewards = np.concatenate(
            [powerloss for powerloss in self.eval_powerloss_episode_rewards if powerloss]
        )
        
        self.eval_voltage_episode_rewards = np.concatenate(
            [voltage for voltage in self.eval_voltage_episode_rewards if voltage]
        )
        
        self.eval_ctrl_episode_rewards = np.concatenate(
            [ctrl for ctrl in self.eval_ctrl_episode_rewards if ctrl]
        )
        
        
        eval_env_reward_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "eval_powerloss_average_episode_rewards": self.eval_powerloss_episode_rewards,
            "eval_voltage_average_episode_rewards": self.eval_voltage_episode_rewards,
            "eval_ctrl_average_episode_rewards": self.eval_ctrl_episode_rewards,
        }
        self.eval_env_reward_infos.update(eval_env_reward_infos)
        self.log_env(eval_env_reward_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        
        # 构建步数和奖励值的日志信息
        log_info = f"{self.total_num_steps},{eval_avg_rew}\n"
        # 打印日志信息到控制台
        print(f"当前步数: {self.total_num_steps}, Evaluation 平均 episode 奖励: {eval_avg_rew}")
        # 写入到日志文件 Progress.txt
        self.log_file.write(log_info)
        self.log_file.flush()
        

    def log_train(self, actor_train_infos, critic_train_info):
        """记录训练信息。"""
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
        """记录环境信息."""
        # 遍历env_infos中的键值对
        for k, v in env_infos.items():
            # 如果v的长度大于0
            if len(v) > 0:
                # 使用writter添加标量，键为k，值为v的平均值，步数为self.total_num_steps
                self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)

    def close(self):
        """Close the logger."""
        self.log_file.close()

    def get_result(self):
        """获取训练和评估的结果"""
        result = {
            "train_avg_reward": np.mean(self.done_episodes_rewards) if self.done_episodes_rewards else 0,
            "eval_avg_reward": np.mean(self.eval_env_reward_infos["eval_average_episode_rewards"]) if "eval_average_episode_rewards" in self.eval_env_reward_infos else 0,
            "eval_powerloss_avg_reward": np.mean(self.eval_env_reward_infos["eval_powerloss_average_episode_rewards"]) if "eval_powerloss_average_episode_rewards" in self.eval_env_reward_infos else 0,
            "eval_voltage_avg_reward": np.mean(self.eval_env_reward_infos["eval_voltage_average_episode_rewards"]) if "eval_voltage_average_episode_rewards" in self.eval_env_reward_infos else 0,
            "eval_ctrl_avg_reward": np.mean(self.eval_env_reward_infos["eval_ctrl_average_episode_rewards"]) if "eval_ctrl_average_episode_rewards" in self.eval_env_reward_infos else 0,
        }
        return result
        
