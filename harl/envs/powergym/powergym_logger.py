from harl.common.base_logger import BaseLogger
import time
import numpy as np

class powergymLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(powergymLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        
        
    def get_task_name(self):
        return self.env_args["env_name"]
    
    def init(self, episodes):
        """Initialize the logger."""
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
        #print("log_dones*********",dones)
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        
        #print(infos)
        #TODO:分离reward,powerloss
        powerloss_reward=[[[d[0]['power_loss_ratio'] for d in infos]]] 
        #print(powerloss_reward)
        powerloss_reward_env= np.mean(powerloss_reward, axis=1).flatten()
        #TODO:分离reward,voltage
        voltage_reward=[[[d[0]['vol_reward'] for d in infos]]] 
        #print(voltage_reward)
        voltage_reward_env= np.mean(voltage_reward, axis=1).flatten()
        
        #TODO:分离reward,ctrl_reward
        ctrl_reward=[[[d[0]['ctrl_reward'] for d in infos]]] 
        #print(ctrl_reward)
        ctrl_reward_env= np.mean(ctrl_reward, axis=1).flatten()
        
           
        self.train_episode_rewards += reward_env
        self.train_episode_powerloss_reward += powerloss_reward_env
        self.train_episode_voltage_reward += voltage_reward_env
        self.train_episode_ctrl_reward += ctrl_reward_env
        
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0
                
                #TODO:power_loss
                self.done_episodes_powerloss_reward.append(self.train_episode_powerloss_reward[t]/24)
                self.train_episode_powerloss_reward[t] = 0
                
                #TODO:Voltage_reward
                self.done_episodes_voltage_reward.append(self.train_episode_voltage_reward[t])
                self.train_episode_voltage_reward[t] = 0
                
                #TODO:Voltage_reward
                self.done_episodes_ctrl_reward.append(self.train_episode_ctrl_reward[t])
                self.train_episode_ctrl_reward[t] = 0

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
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
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
            #TODO:power_loss,voltage_loss,ctrl_loss
            aver_episode_powerloss_rewards = np.mean(self.done_episodes_powerloss_reward)
            aver_episode_voltage_rewards = np.mean(self.done_episodes_voltage_reward)
            aver_episode_ctrl_rewards = np.mean(self.done_episodes_ctrl_reward)
            
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
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
        """Initialize the logger for evaluation."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
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
        """Log evaluation information per step."""
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
        
        #print(eval_powerloss_reward)
        eval_powerloss_reward_env= np.mean(eval_powerloss_reward, axis=1).flatten()
        eval_powerloss_reward_env=eval_powerloss_reward_env.reshape((self.algo_args["eval"]["n_eval_rollout_threads"],1,1))
        
        eval_voltage_reward_env= np.mean(eval_voltage_reward, axis=1).flatten()
        eval_voltage_reward_env=eval_voltage_reward_env.reshape((self.algo_args["eval"]["n_eval_rollout_threads"],1,1))
        
        eval_ctrl_reward_env= np.mean(eval_ctrl_reward, axis=1).flatten()
        eval_ctrl_reward_env=eval_ctrl_reward_env.reshape((self.algo_args["eval"]["n_eval_rollout_threads"],1,1))
        
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
            self.one_powerloss_episode_rewards[eval_i].append(eval_powerloss_reward_env[eval_i])#TODO:POWERLOSS_EVAL
            self.one_voltage_episode_rewards[eval_i].append(eval_voltage_reward_env[eval_i])#TODO:POWERLOSS_EVAL
            self.one_ctrl_episode_rewards[eval_i].append(eval_ctrl_reward_env[eval_i])#TODO:POWERLOSS_EVAL
        self.eval_infos = eval_infos

        
    def eval_thread_done(self, tid):
        """Log evaluation information."""
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
        
        
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
            "eval_powerloss_average_episode_rewards": self.eval_powerloss_episode_rewards,
            "eval_voltage_average_episode_rewards": self.eval_voltage_episode_rewards,
            "eval_ctrl_average_episode_rewards": self.eval_ctrl_episode_rewards,
        }
        
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"#这个是在progress文件里写的
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
