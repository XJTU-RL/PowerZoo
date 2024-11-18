import copy
import gym
from gym.spaces import Discrete,Box,MultiDiscrete
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
from envs.powerzoo.powerzoo.env_register import make_env, remove_parallel_dss

import argparse
import random
import itertools
import sys, os
import multiprocessing as mp

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Argument Parser')
#     parser.add_argument('--env_name', default='13Bus')
#     parser.add_argument('--seed', type=int, default=123456, metavar='N',
#                          help='random seed')
#     parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
#                          help='maximum number of steps')
#     parser.add_argument('--num_workers', type=int, default=3, metavar='N', 
#                          help='number of parallel processes')
#     parser.add_argument('--use_plot', type=lambda x: str(x).lower()=='true', default=False)#这个功能呢暂时不打算加，因为训练的时间步太多了，实验预想可以考虑，事故处理
#     parser.add_argument('--do_testing', type=lambda x: str(x).lower()=='true', default=False)
#     parser.add_argument('--mode', type=str, default='single',
#                         help="running mode, random, parallele, episodic or dss")
#     args = parser.parse_args()
#     return args


def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class PowerZooEnv:
    def __init__(self, args,rank=None):#TODO: ranks是线程数 
        
        self.args = copy.deepcopy(args)
        self.env = make_env(args['env_name'], worker_idx=rank)#args
        self.env.seed(args['seed'] + 0)
        #智能体数量是电容、有载调压器、电池数量之和
        CRB_num = self.env.cap_num+self.env.reg_num+self.env.bat_num
        agents = [i for i in range(0,CRB_num)]
        self.agents=agents
        self.n_agents = len(agents)
        #排序顺序
        self.cap_names =self.env.cap_names
        self.reg_names = self.env.reg_names
        self.bat_names = self.env.bat_names
        
        self.rank=rank#线程编号
        self.env_name=args['env_name']#便于实现多线程
        
        agents_names = self.cap_names+self.reg_names+self.bat_names
        #print(self.agents_names)
        #排序函数
        # 使用sorted函数进行排序，根据自定义排序函数排序，这里是把名字和顺序结合起来了
        # update_orders=list(range(0,self.n_agents))
        # sorted_pairs = sorted(zip(agents_names, update_orders), key=lambda x: self.custom_sort(x[0]))

        # # 分离排序后的元组对以获取排序后的列表,得到想要的更新顺序
        # self.agents_names, self.update_orders = zip(*sorted_pairs)
        # print("agents_names: ",self.agents_names)
        # print("update_orders:",self.update_orders)
        self.env.use_render=args['use_render']
        self.env.useS=args['useS']
        self.env.record_node = args['record_node']
        if args['useS']==True:
            update_orders=list(range(0,self.n_agents))
            self.ordered_agents_pairs = dict(zip(agents_names, update_orders))
            self.agents_bus=self.env.agents_bus 
        else:
            self.ordered_agents_pairs = None
            self.agents_bus=None
        self.share_observation_space = self.repeat(self.env.observation_space)
        self.observation_space = self.unwrap(self.env.observation_space)
        
        self.action_space = self.get_env_action_space(self.env.action_space)#把每个智能体的动作空间拆解出来了
        self.avail_actions = self.get_avail_actions()
        if self.env.action_space.__class__.__name__ == "Box":
            self.discrete = False
        else:
            self.discrete = True # 对所有的动作空间进行离散化处理，需要对电池进行处理，使其离散化
        
    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        #print("step=",np.array(actions)) #TODO:选择是否打印每一步的动作
        if self.discrete:
            obs, rew, done, info = self.env.step(actions.flatten())
        else:
            obs, rew, done, info = self.env.step(actions[0])
        if done:
            if (
                "TimeLimit.truncated" in info.keys()
                and info["TimeLimit.truncated"] == True
            ):
                info["bad_transition"] = True
        return self.unwrap(obs), self.unwrap(obs), [[rew]], self.unwrap(done), [info], self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        #self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.reset(load_profile_idx=self.rank))
        s_obs = copy.deepcopy(obs)
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.array(avail_actions,dtype=object).tolist()
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n
        

    def render(self):#函数需要修改
        #self.env.render()
        pass

    def close(self):
        #self.env.close()
        remove_parallel_dss(self.env_name, self.rank)
        print("Closing the environment")

    def seed(self, seed):
        #self.env.seed(seed)
        self.env.seed(seed)#use default seed
        
    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d)
        return l#pettingzoo是从这里面挑自己的观察空间
    
    def get_env_action_space(self,env):#把混合动作空间分给每个单独的智能体
        #print("env =",env)
        discrete_list = [Discrete(n) for n in env.nvec]
        #print("discrete_list =",discrete_list)
        return discrete_list
    
    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    # def custom_sort(self, item):#brc
    # # 根据元素名称中是否包含'reg'或'cap'进行排序,函数功能：把cap放在最前面按从小到大的顺序，把reg放中间，把bat放最后
    #     if 'Capacitor' in item:
    #        return (2, item)
    #     elif 'Regulator' in item:
    #        return (1, item)
    #     else:#bat
    #        return (0, item)
    # # 自定义排序函数
    # def custom_sort(item):
    #     # 将'Capacitor.cap1'放在第一个位置
    #     if item == 'Capacitor.cap1':
    #         return (-1, item)
    #     else:
    #         # 其他元素按默认规则排序
    #         if 'reg' in item:
    #             return (0, item)
    #         elif 'cap' in item:
    #             return (2, item)
    #         else:#bat
    #             return (1, item)
    # 自定义排序函数
