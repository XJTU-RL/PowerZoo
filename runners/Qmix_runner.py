#import wandb
import numpy as np
from itertools import chain
import torch
import time
from utils.trans_tools import is_multidiscrete
from runners.Qmix_base_runner import MlpRunner

class QMIXRunner(MlpRunner):
    def __init__(self,args,algo_args,env_args):
        """Runner class for the Multi-Agent Particle Env (MPE)  environment. See parent class for more information."""
        super(QMIXRunner, self).__init__(args,algo_args,env_args)
        self.collecter = self.shared_collect_rollout if self.share_policy else self.separated_collect_rollout
        # fill replay buffer with random actions
        self.finish_first_train_reset = False
        
        self.start = time.time()
        self.log_clear()
        if algo_args["render"]["use_render"]:
            self.use_render()
        else:
            num_warmup_episodes = max((int(self.batch_size//self.episode_length) + 1, self.args.num_random_episodes))
            self.warmup(num_warmup_episodes)


    @torch.no_grad()
    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()
        eval_infos = {}
        eval_infos['average_episode_rewards'] = []
        eval_infos['powerloss_average_episode_rewards']=[]
        eval_infos['voltage_average_episode_rewards']=[]
        eval_infos['ctrl_average_episode_rewards']=[]
        
        eval_infos['power_loss_kw']=[]
        eval_infos['power_loss_kvar']=[]
        eval_infos['total_power_kw']=[]
        eval_infos['total_power_kvar']=[]
        
        eval_infos['capacitor_control']=[]
        eval_infos['regulator_control']=[]
        eval_infos['discharge_control']=[]
        
        

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False)
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")#todo:self.total_env_step也得变，total_env_step =warmup_epsode*72,更新self.lastlog和self.last_eval_T

    @torch.no_grad()
    def use_render(self):
        """Collect episodes to evaluate the policy."""
        #self.trainer.prep_rollout()
        render_infos = {}
        render_infos['average_episode_rewards'] = []
        render_infos['powerloss_average_episode_rewards']=[]
        render_infos['voltage_average_episode_rewards']=[]
        render_infos['ctrl_average_episode_rewards']=[]
        
        render_infos['power_loss_kw']=[]
        render_infos['power_loss_kvar']=[]
        render_infos['total_power_kw']=[]
        render_infos['total_power_kvar']=[]
        
        render_infos['capacitor_control']=[]
        render_infos['regulator_control']=[]
        render_infos['discharge_control']=[]
        
        self.collecter( explore=False, training_episode=False, warmup=False, render=True)


    # for mpe-simple_spread and mpe-simple_reference
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy. Do training steps when appropriate
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if explore else self.eval_env
        n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        if not explore:
            obs = env.reset()
            share_obs = obs.reshape(n_rollout_threads, -1)
        else:
            if self.finish_first_train_reset:
                obs = self.obs
                share_obs = self.share_obs
            else:
                obs = env.reset()
                share_obs = obs.reshape(n_rollout_threads, -1)
                self.finish_first_train_reset = True

        # init
        episode_rewards = []
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        valid_transition = {}
        step_avail_acts = {}
        step_next_avail_acts = {}

        for step in range(self.episode_length):
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, _ = policy.get_actions(obs_batch,
                                                    t_env=self.total_env_steps,
                                                    explore=explore)

            if not isinstance(acts_batch, np.ndarray):
                acts_batch = acts_batch.cpu().detach().numpy()
            env_acts = np.split(acts_batch, n_rollout_threads)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)

            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs = env.reset()

            if not explore and np.all(dones_env):
                average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
                env_info['average_episode_rewards'] = average_episode_rewards
                return env_info

            next_share_obs = next_obs.reshape(n_rollout_threads, -1)

            step_obs[p_id] = obs
            step_share_obs[p_id] = share_obs
            step_acts[p_id] = env_acts
            step_rewards[p_id] = rewards
            step_next_obs[p_id] = next_obs
            step_next_share_obs[p_id] = next_share_obs
            step_dones[p_id] = np.zeros_like(dones)
            step_dones_env[p_id] = dones_env
            valid_transition[p_id] = np.ones_like(dones)
            step_avail_acts[p_id] = None
            step_next_avail_acts[p_id] = None

            obs = next_obs
            share_obs = next_share_obs

            if explore:
                self.obs = obs
                self.share_obs = share_obs
                # push all episodes collected in this rollout step to the buffer
                self.buffer.insert(n_rollout_threads,
                                   step_obs,
                                   step_share_obs,
                                   step_acts,
                                   step_rewards,
                                   step_next_obs,
                                   step_next_share_obs,
                                   step_dones,
                                   step_dones_env,
                                   valid_transition,
                                   step_avail_acts,
                                   step_next_avail_acts)

            # train
            if training_episode:
                self.total_env_steps += n_rollout_threads
                if (self.last_train_T == 0 or ((self.total_env_steps - self.last_train_T) / self.train_interval) >= 1):
                    self.train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps
            
        average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards

        return env_info

    # for mpe-simple_speaker_listener 
    def separated_collect_rollout(self, explore=True, training_episode=True, warmup=False,render=False):
        """
        Collect a rollout and store it in the buffer. Each agent has its own policy.. Do training steps when appropriate.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        if render:
            env= self.env
            n_rollout_threads=1
        else:
            env = self.env if explore else self.eval_env
            n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        if not explore:
            obs,_,_ = env.reset()
            share_obs = []
            if render: 
               share_obs=obs[0]
               share_obs = np.array(share_obs) 
            else:
                for o in obs:
                    share_obs.append(list(chain(*o)))
                share_obs = np.array(share_obs)
        else:
            if self.finish_first_train_reset:
                obs = self.obs
                share_obs = self.share_obs
            else:
                obs,_,_ = env.reset()
                share_obs = []
                # for o in obs:
                #     share_obs.append(list(chain(*o)))
                share_obs=obs[:, 0, :]
                share_obs = np.array(share_obs)
                self.finish_first_train_reset = True

        agent_obs = []
            
        for agent_id in range(self.num_agents):
            env_obs = []
            if render:
                env_obs=obs[agent_id]
            else:
                for o in obs:
                    env_obs.append(o[agent_id])
            env_obs = np.array(env_obs)
            agent_obs.append(env_obs)

        # [agents, parallel envs, dim]
        episode_rewards = []
        episode_powerloss_reward = []
        episode_ctrl_reward = []
        episode_violation_reward = []
        
        render_episode_rewards=[]
        
        episode_plkw=[]
        episode_plkv=[]
        episode_tpkw=[]
        episode_tpkv=[]
        
        episode_cap_ctrl=[]
        episode_reg_ctrl=[]
        episode_dis_ctrl=[]
        
        
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        valid_transition = {}
        step_avail_acts = {}
        step_next_avail_acts = {}

        acts = []
        for p_id in self.policy_ids:
            if is_multidiscrete(self.policy_info[p_id]['act_space']):
                self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
            else:
                self.sum_act_dim = self.policy_act_dim[p_id]
            temp_act = np.zeros((n_rollout_threads, self.sum_act_dim))
            acts.append(temp_act)

        for step in range(self.episode_length):
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                # get actions for all agents to step the env
                if warmup:#在此处补充log文件，log从此处开始，步数也得做变更
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs[agent_id])
                elif render:
                    act, _ = policy.get_actions(agent_obs[agent_id].reshape(1,len(agent_obs[agent_id])),
                            t_env=step,
                            explore=explore)
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    act, _ = policy.get_actions(agent_obs[agent_id],
                                                t_env=self.total_env_steps,
                                                explore=explore)

                if not isinstance(act, np.ndarray):
                    act = act.cpu().detach().numpy()
                acts[agent_id] = act

            env_acts = []
            for i in range(n_rollout_threads):
                env_act = []
                for agent_id in range(self.num_agents):
                    env_act.append(np.argmax(acts[agent_id][i]))
                env_acts.append(env_act)

            # env step and store the relevant episode information
            next_obs,_, rewards, dones, infos,_ = env.step(np.array(env_acts))
            ##记录每一步的奖励值分离
            # 使用嵌套的列表推导式来获取所有'power_loss_ratio'的值
            if render:
                powerloss_reward = infos[0]['power_loss_ratio']
                vol_reward = infos[0]['vol_reward']
                ctrl_reward =infos[0]['ctrl_reward']
                
                        #分离真实的powerloss
                power_loss_kw=infos[0]['power_loss_kw']
                power_loss_kvar=infos[0]['power_loss_kvar']
                total_power_kw=infos[0]['total_power_kw']
                total_power_kvar=infos[0]['total_power_kvar']
                
                cap_ctrl=infos[0]['capacitor_ctrl']
                reg_ctrl=infos[0]['regulator_ctrl']
                dis_ctrl=infos[0]['discharge_ctrl']
            else:
                powerloss_reward = [d['power_loss_ratio'] for sublist in infos for d in sublist if 'power_loss_ratio' in d]
                vol_reward = [d['vol_reward'] for sublist in infos for d in sublist if 'vol_reward' in d]
                ctrl_reward = [d['ctrl_reward'] for sublist in infos for d in sublist if 'ctrl_reward' in d]
                
                        #分离真实的powerloss
                power_loss_kw=[d['power_loss_kw'] for sublist in infos for d in sublist if 'power_loss_kw' in d]
                power_loss_kvar=[d['power_loss_kvar'] for sublist in infos for d in sublist if 'power_loss_kvar' in d]
                total_power_kw=[d['total_power_kw'] for sublist in infos for d in sublist if 'total_power_kw' in d]
                total_power_kvar=[d['total_power_kvar'] for sublist in infos for d in sublist if 'total_power_kvar' in d]
                
                cap_ctrl=[d['capacitor_ctrl'] for sublist in infos for d in sublist if 'capacitor_ctrl' in d]
                reg_ctrl=[d['regulator_ctrl'] for sublist in infos for d in sublist if 'regulator_ctrl' in d]
                dis_ctrl=[d['discharge_ctrl'] for sublist in infos for d in sublist if 'discharge_ctrl' in d]
            
            render_infos = {}
            if render:
                render_infos['step_rewards'] = rewards[0][0]
                render_infos['powerloss_step_rewards']=powerloss_reward
                render_infos['voltage_step_rewards']=vol_reward
                render_infos['ctrl_step_rewards']=ctrl_reward
                
                render_infos['power_loss_kw']=power_loss_kw
                render_infos['power_loss_kvar']=power_loss_kvar
                render_infos['total_power_kw']=total_power_kw
                render_infos['total_power_kvar']=total_power_kvar
                
                render_infos['capacitor_control']=cap_ctrl
                render_infos['regulator_control']=reg_ctrl
                render_infos['discharge_control']=dis_ctrl
                
                suffix="render_"
                for k, v in render_infos.items():
                    suffix_k = k if suffix is None else suffix + k
                    print(suffix_k + " is " + str(v))
                    if self.use_wandb:
                        wandb.log({suffix_k: v}, step=step)
                    else:
                        self.writter.add_scalar(suffix_k, v, step)
                        self.writter.add_scalar("example_metric", step * 0.1, step)
                render_episode_rewards.append(rewards)
                if step==23:
                     average_render_episode_rewards=np.sum(render_episode_rewards)
                     print("部署累计奖励为：", average_render_episode_rewards)   
                continue
            
            
            
            
        
            
            episode_rewards.append(rewards)
            episode_powerloss_reward.append(powerloss_reward)
            episode_ctrl_reward.append(ctrl_reward)
            episode_violation_reward.append(vol_reward)
            
            #分离eval实际的powerloss
            episode_plkw.append(power_loss_kw)
            episode_plkv.append(power_loss_kvar)
            episode_tpkw.append(total_power_kw)
            episode_tpkv.append(total_power_kvar)
            
            
            episode_cap_ctrl.append(cap_ctrl)
            episode_reg_ctrl.append(reg_ctrl)
            episode_dis_ctrl.append(dis_ctrl)
            if render:
                dones_env = np.all(dones)
            else:
                dones_env = np.all(dones, axis=1)

            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs,_,_ = env.reset()

            if not explore and np.all(dones_env):
                average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
                average_episode_powerloss=np.mean(np.sum(episode_powerloss_reward, axis=0))/24
                average_episode_ctrl_reward=np.mean(np.sum(episode_ctrl_reward, axis=0))
                average_episode_violation_reward=np.mean(np.sum(episode_violation_reward, axis=0))
                
                #分离实际的powerloss
                average_episode_plkw=np.mean(np.sum(episode_plkw, axis=0))/24
                average_episode_plkv=np.mean(np.sum(episode_plkv, axis=0))/24
                average_episode_tpkw=np.mean(np.sum(episode_tpkw, axis=0))/24
                average_episode_tpkv=np.mean(np.sum(episode_tpkv, axis=0))/24
                
                average_episode_cap_ctrl=np.mean(np.sum(episode_cap_ctrl, axis=0))
                average_episode_reg_ctrl=np.mean(np.sum(episode_reg_ctrl, axis=0))
                average_episode_dis_ctrl=np.mean(np.sum(episode_dis_ctrl, axis=0))
                
                
                
                
                env_info['average_episode_rewards'] = average_episode_rewards
                env_info['powerloss_average_episode_rewards']=average_episode_powerloss
                env_info['voltage_average_episode_rewards']=average_episode_ctrl_reward
                env_info['ctrl_average_episode_rewards']=average_episode_violation_reward
                
                env_info['power_loss_kw']=average_episode_plkw
                env_info['power_loss_kvar']=average_episode_plkv
                env_info['total_power_kw']=average_episode_tpkw
                env_info['total_power_kvar']=average_episode_tpkv
                
                env_info['capacitor_control']=average_episode_cap_ctrl
                env_info['regulator_control']=average_episode_reg_ctrl
                env_info['discharge_control']=average_episode_dis_ctrl
                
                
                return env_info

            next_share_obs = []
            # for no in next_obs:
            #     next_share_obs.append(list(chain(*no)))
            # next_share_obs = np.array(next_share_obs)
            next_share_obs= np.array(next_obs[:,0,:])

            next_agent_obs = []
            for agent_id in range(self.num_agents):
                next_env_obs = []
                for no in next_obs:
                    next_env_obs.append(no[agent_id])
                next_env_obs = np.array(next_env_obs)
                next_agent_obs.append(next_env_obs)
            ##############
            rewards = np.repeat(rewards, self.num_agents, axis=1)
            ############

            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                step_obs[p_id] = np.expand_dims(agent_obs[agent_id], axis=1)
                step_share_obs[p_id] = share_obs
                step_acts[p_id] = np.expand_dims(acts[agent_id], axis=1)
                step_rewards[p_id] = np.expand_dims(rewards[:, agent_id], axis=1)
                step_next_obs[p_id] = np.expand_dims(next_agent_obs[agent_id], axis=1)
                step_next_share_obs[p_id] = next_share_obs
                step_dones[p_id] = np.expand_dims(np.zeros_like(np.expand_dims(dones[:, agent_id], axis=1)),axis=-1)
                step_dones_env[p_id] =np.expand_dims(dones_env, axis=-1)
                valid_transition[p_id] = np.expand_dims(np.ones_like(np.expand_dims(dones[:, agent_id], axis=1)),axis=-1)
                step_avail_acts[p_id] = None
                step_next_avail_acts[p_id] = None

            obs = next_obs
            agent_obs = next_agent_obs
            share_obs = next_share_obs

            if explore:#在此处插入warmup的数据
                self.obs = obs
                self.share_obs = share_obs
                self.buffer.insert(n_rollout_threads,
                                   step_obs,
                                   step_share_obs,
                                   step_acts,
                                   step_rewards,
                                   step_next_obs,
                                   step_next_share_obs,
                                   step_dones,
                                   step_dones_env,
                                   valid_transition,
                                   step_avail_acts,
                                   step_next_avail_acts)
                
            #记录warmup的步数
            if warmup:
                self.total_env_steps += n_rollout_threads

            # train
            if training_episode:
                self.total_env_steps += n_rollout_threads
                if (self.last_train_T == 0 or ((self.total_env_steps - self.last_train_T) / self.train_interval) >= 1):
                    self.train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps

        average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
        
        average_episode_powerloss=np.mean(np.sum(episode_powerloss_reward, axis=0))/24
        average_episode_ctrl_reward=np.mean(np.sum(episode_ctrl_reward, axis=0))
        average_episode_violation_reward=np.mean(np.sum(episode_violation_reward, axis=0))
        
        #分离eval实际的powerloss
        average_episode_plkw=np.mean(np.sum(episode_plkw, axis=0))/24
        average_episode_plkv=np.mean(np.sum(episode_plkv, axis=0))/24
        average_episode_tpkw=np.mean(np.sum(episode_tpkw, axis=0))/24
        average_episode_tpkv=np.mean(np.sum(episode_tpkv, axis=0))/24
        
        average_episode_cap_ctrl=np.mean(np.sum(episode_cap_ctrl, axis=0))
        average_episode_reg_ctrl=np.mean(np.sum(episode_reg_ctrl, axis=0))
        average_episode_dis_ctrl=np.mean(np.sum(episode_dis_ctrl, axis=0))
        
        
        env_info['average_episode_rewards'] = average_episode_rewards
        env_info['powerloss_average_episode_rewards']=average_episode_powerloss
        env_info['voltage_average_episode_rewards']=average_episode_ctrl_reward
        env_info['ctrl_average_episode_rewards']=average_episode_violation_reward
        
        env_info['power_loss_kw']=average_episode_plkw
        env_info['power_loss_kvar']=average_episode_plkv
        env_info['total_power_kw']=average_episode_tpkw
        env_info['total_power_kvar']=average_episode_tpkv
        
        env_info['capacitor_control']=average_episode_cap_ctrl
        env_info['regulator_control']=average_episode_reg_ctrl
        env_info['discharge_control']=average_episode_dis_ctrl
        

        return env_info 

    def log(self):
        """See parent class."""
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}.\n"
              .format(self.args.env_name,
                      self.algorithm_name,
                      self.env_args.env_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_env(self, env_info, suffix=None):
        """See parent class."""
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))
                if self.use_wandb:
                    wandb.log({suffix_k: v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos['average_episode_rewards'] = []
        self.env_infos['powerloss_average_episode_rewards']=[]
        self.env_infos['voltage_average_episode_rewards']=[]
        self.env_infos['ctrl_average_episode_rewards']=[]

        self.env_infos['power_loss_kw']=[]
        self.env_infos['power_loss_kvar']=[]
        self.env_infos['total_power_kw']=[]
        self.env_infos['total_power_kvar']=[]
        
        self.env_infos['capacitor_control']=[]
        self.env_infos['regulator_control']=[]
        self.env_infos['discharge_control']=[]



    
    @torch.no_grad()
    def warmup(self, num_warmup_episodes):
        # fill replay buffer with enough episodes to begin training
        self.trainer.prep_rollout()
        warmup_rewards = []
        
        eval_infos = {}
        eval_infos['average_episode_rewards'] = []
        eval_infos['powerloss_average_episode_rewards']=[]
        eval_infos['voltage_average_episode_rewards']=[]
        eval_infos['ctrl_average_episode_rewards']=[]
        
        eval_infos['power_loss_kw']=[]
        eval_infos['power_loss_kvar']=[]
        eval_infos['total_power_kw']=[]
        eval_infos['total_power_kvar']=[]
        
        eval_infos['capacitor_control']=[]
        eval_infos['regulator_control']=[]
        eval_infos['discharge_control']=[]
        
        
        
        
        print("warm up...")
        for _ in range(int(num_warmup_episodes // self.num_envs) + 1):
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)#一次返回了一个episode的平均值，一个batch_size是72，所以一共是144个
            warmup_rewards.append(env_info['average_episode_rewards'])
            
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")#todo:self.total_env_step也得变，total_env_step =warmup_epsode*72,更新self.lastlog和self.last_eval_T
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: {}".format(warmup_reward))