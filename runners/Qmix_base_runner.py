import os
#import wandb
import numpy as np
from itertools import chain
from tensorboardX import SummaryWriter
import torch
import time
from utils.models_tools import init_device
from utils.mlp_buffer import MlpReplayBuffer, PrioritizedMlpReplayBuffer
from utils.trans_tools import is_discrete, is_multidiscrete, DecayThenFlatSchedule,get_dim_from_space,get_cent_act_dim
from utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from utils.configs_tools import init_dir, save_config, get_task_name
import setproctitle

class DictToClass:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

class MlpRunner(object):

    def __init__(self,args,algo_args,env_args):
        """
        Base class for training MLP policies.
        在init中完成环境的创建,算法的指定,buffer的创建,writer的创建,以及wandb的初始化
           Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        # non-tunable hyperparameters are in args
        self.parse_args = args
        self.args = DictToClass(algo_args)
        
        #self.args = algo_args
        self.env_args = DictToClass(env_args)
        
        self.device =init_device(algo_args["device"])
        self.q_learning = ["mqmix","mvdn"]
        
        self.render=algo_args["render"]["use_render"]
        self.num_render_envs=algo_args["render"]["render_episodes"]

        # set tunable hyperparameters
        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.env_args.env_name
        self.num_env_steps = self.args.train["num_env_steps"]
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval = self.args.hard_update_interval
        self.train_interval = self.args.train_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        self.total_env_steps = 0  # total environment interactions collected during training
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_T = 0
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0
        self.last_hard_update_T = 0
        
        #config包含的信息比algo_args更多,qmix源代码采取的逻辑是把线程和存储地址都设置好了，再进入runner函数
        '''
            config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "use_same_share_obs": all_args.use_same_share_obs,
              "run_dir": run_dir
              }
        '''
        

        # if config.__contains__("take_turn"):
        #     self.take_turn = config["take_turn"]
        # else:
        #     self.take_turn = False

        if algo_args.__contains__("use_same_share_obs"):
            self.use_same_share_obs = algo_args["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if algo_args.__contains__("use_available_actions"):
            self.use_avail_acts = algo_args["use_available_actions"]
        else:
            self.use_avail_acts = False

        self.episode_length = self.args.train["episode_length"]
        
        
        
        self.task_name = get_task_name(args["env"], env_args)
    #if not algo_args["render"]["use_render"]:
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"],
            env_args,
            args["algo"],
            args["exp_name"],
            algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)
        
        self.log_file = open(
            os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
        )
            
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # env
        if algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_deaths = np.zeros(
            (algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
        )

        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)

        print("share_observation_space.shape: ", len(self.envs.share_observation_space))
        print("observation_space.shape: ", len(self.envs.observation_space))
        print("action_space.shape ", len(self.envs.action_space))
        
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(self.envs.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(self.envs.action_space),#这是总体的action_space,拼接得到的action_space
                                        "obs_space": self.envs.observation_space[agent_id],#每个智能体单独的observation_space
                                        "share_obs_space": self.envs.share_observation_space[agent_id],#和上面的相同
                                        "act_space": self.envs.action_space[agent_id]}#每个智能体单独的action_space
            for agent_id in range(self.num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)
        
        
        
        

        self.policy_info = policy_info
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = policy_mapping_fn

        #self.num_agents = self.num_agents
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = self.envs
        if not algo_args["render"]["use_render"]:
            self.eval_env = self.eval_envs
        if algo_args["render"]["use_render"]:
            self.num_envs=1
        else:
            self.num_envs = algo_args["train"]["n_rollout_threads"]
        self.num_eval_envs = algo_args["eval"]["n_eval_rollout_threads"]

        #dir
        self.model_dir = self.args.model_dir
        # if self.use_wandb:
        #     self.save_dir = str(wandb.run.dir)
        # else:
        #     self.run_dir = self.run_dir
        #     self.log_dir = str(self.run_dir / 'logs')
        #     if not os.path.exists(self.log_dir):
        #         os.makedirs(self.log_dir)
        #     self.writter = SummaryWriter(self.log_dir)
        #     self.save_dir = str(self.run_dir / 'models')
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "mqmix":
            from models.policy_models.m_qmix_policy import M_QMixPolicy as Policy
            from algorithms.actors.m_Qmix import M_QMix as TrainAlgo
        # elif self.algorithm_name == "mvdn":
        #     from offpolicy.algorithms.mvdn.algorithm.mVDNPolicy import M_VDNPolicy as Policy
        #     from offpolicy.algorithms.mvdn.mvdn import M_VDN as TrainAlgo
        else:
            raise NotImplementedError

        self.collecter = self.collect_rollout
        self.saver = self.save_q if self.algorithm_name in self.q_learning else self.save
        self.train = self.batch_train_q if self.algorithm_name in self.q_learning else self.batch_train
        self.restorer = self.restore_q if self.algorithm_name in self.q_learning else self.restore

        self.policies = {p_id: Policy(self.args,self.device, self.policy_info[p_id]) for p_id in self.policy_ids}

        if self.model_dir is not None:
            self.restorer()

        # initialize class for updating policies
        self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,
                                 device=self.device)


        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_iters = self.num_env_steps / self.train_interval
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_iters, decay="linear")

        if self.use_per:
            self.buffer = PrioritizedMlpReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization)
        else:
            self.buffer = MlpReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.buffer_size,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization)
        

    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        if self.render:
            self.env.close()
        else:
            num_warmup_episodes = max((int(self.batch_size//self.episode_length) + 1, self.args.num_random_episodes))
            warm_up_step=(num_warmup_episodes // self.num_envs) + 1
            total_num_steps = 0+warm_up_step*self.batch_size#可以在此处将其变为warmup_step
            while total_num_steps < self.num_env_steps:
            # collect data
                self.trainer.prep_rollout()
                env_info = self.collecter(explore=True, training_episode=True, warmup=False)
                for k, v in env_info.items():
                    self.env_infos[k].append(v)

                # save
                if (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
                    self.saver()
                    self.last_save_T = self.total_env_steps

                # log
                if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
                    self.log()
                    self.last_log_T = self.total_env_steps

                # eval
                if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
                    self.eval()
                    self.last_eval_T = self.total_env_steps
                total_num_steps=self.total_env_steps
            self.envs.close()
            if self.args.use_eval and (self.eval_env is not self.env):
                self.eval_env.close()

        # if all_args.use_wandb:
        #     run.finish()
        # else:
        # self.writter.export_scalars_to_json(
        #     str(self.log_dir + '/summary.json'))
        self.writter.close()

        return self.total_env_steps

    def batch_train(self):
        """Do a gradient update for all policies."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []
        update_actor = True
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            update = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update(p_id, sample)
            update_actor = train_info['update_actor']

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.total_env_steps - self.last_hard_update_T) / self.hard_update_interval) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_T = self.total_env_steps

    def batch_train_q(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample, self.use_same_share_obs)

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.total_env_steps - self.last_hard_update_T) / self.hard_update_interval >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_T = self.total_env_steps

    def save(self):
        """Save all policies to the path specified by the config."""
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(),
                       critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(),
                       actor_save_path + '/actor.pt')

    def save_q(self):
        """Save all policies to the path specified by the config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.trainer.mixer.state_dict(),
                   self.save_dir + '/mixer.pt')

    def restore(self):
        """Load policies policies from pretrained models specified by path in config."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def restore_q(self):
        """Load policies policies from pretrained models specified by path in config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_q_state_dict = torch.load(path + '/q_network.pt')           
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)
            
        #policy_mixer_state_dict = torch.load(str(self.model_dir) + '/mixer.pt')
        #self.trainer.mixer.load_state_dict(policy_mixer_state_dict)

    @torch.no_grad()
    def warmup(self, num_warmup_episodes):
        """
        Fill replay buffer with enough episodes to begin training.

        :param: num_warmup_episodes (int): number of warmup episodes to collect.
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range(int(num_warmup_episodes // self.num_envs) + 1):
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info['average_step_rewards'])
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average step rewards: {}".format(warmup_reward))

    def log(self):
        raise NotImplementedError

    def log_clear(self):
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging.
        """
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))
                if self.use_wandb:
                    wandb.log({suffix_k: v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)

    def log_train(self, policy_id, train_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(policy_k, {policy_k: v}, self.total_env_steps)

    def collect_rollout(self):
        """Collect a rollout and store the transitions in the buffer."""
        raise NotImplementedError
    
    def close(self):
        """Close environment, writter, and log file."""
        # post process
        if self.args.render["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.args.eval["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.log_file.close()