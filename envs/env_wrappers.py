"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import copy
import logging

logger = logging.getLogger(__name__)

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self, num_envs, observation_space, share_observation_space, action_space
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space)
            )
        elif cmd == "render_vulnerability":
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == "get_num_agents":
            # 兼容不同环境类型，优先使用vvc_env.py的VVCEnv包装器
            if hasattr(env, 'n_agents'):
                remote.send((env.n_agents))
            else:
                # 对于原始Env类，根据设备数量计算智能体数量
                n_agents = getattr(env, 'cap_num', 0) + getattr(env, 'reg_num', 0) + getattr(env, 'bat_num', 0)
                # 如果启用PV控制，添加PV智能体
                if hasattr(env, 'pv_control_enabled') and env.pv_control_enabled:
                    n_agents += getattr(env, 'pv_num', 0)
                remote.send((n_agents))
        # elif cmd == "get_agents_orders":
        #     remote.send((env.update_orders))
        elif cmd == "get_ordered_agents_pairs":
            remote.send((env.ordered_agents_pairs))
        elif cmd == "get_agents_bus":
            remote.send((env.agents_bus))
            
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        
        # 创建nenvs个数量的管道
        # work_remote和remote是一对由Pipe()创建的通信管道的两端。work_remote用于在子进程中，而remote用于在主进程中
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)]) 
        
        # 创建 process list
        self.ps = [
            Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ] 
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start() #启动所有进程

        #关闭每个进程的 work_remote端
        for remote in self.work_remotes:
            remote.close() 

        # 关闭work_remote端点是一种标准做法，旨在提高效率、安全性和代码的清晰度。
        # 这种做法在使用Python的multiprocessing模块进行多进程通信时尤其常见
            

        #命令是一个元组，其中第一个元素是字符串"get_num_agents"，指示子进程返回环境中的代理数量。
        self.remotes[0].send(("get_num_agents", None)) 
        #接收来自子进程的响应。这里，响应应该是环境中的代理数量，该值被存储在self.n_agents中
        self.n_agents = self.remotes[0].recv() 
        #向第一个子进程发送另一个命令，请求环境的空间信息。这通常包括观测空间和动作空间的定义。
        self.remotes[0].send(("get_spaces", None)) 
        #接收并解包从子进程返回的空间信息。这些信息通常包括观测空间（环境提供的每个观测的格式或结构）、
        # 共享观测空间（如果环境中有多个代理需要共享信息），以及动作空间（代理可以执行的动作的格式或结构）。
        observation_space, share_observation_space, action_space = self.remotes[
            0
        ].recv()
        
        
        #自定义更新顺序
        # self.remotes[0].send(("get_agents_orders", None)) 
        # #接收来自子进程的响应。这里，响应应该是环境中的代理数量，该值被存储在self.update_orders中
        # self.update_orders = self.remotes[0].recv() 
        
        #传递更新顺序
        self.remotes[0].send(("get_ordered_agents_pairs", None)) 
        #接收来自子进程的响应。这里，响应应该是环境中的代理数量，该值被存储在self.original_orders中
        self.ordered_agents_pairs = self.remotes[0].recv()
        
        #传递智能体和网络关系
        self.remotes[0].send(("get_agents_bus", None)) 
        #接收来自子进程的响应。这里，响应应该是环境中的代理数量，该值被存储在self.original_orders中
        self.agents_bus = self.remotes[0].recv() 
        
        
               
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )

    def step_async(self, actions):
        # 先发送给所有子进程执行命令,而不是一个一个地发送完等待回复再进行下一步
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        # 显式地等待
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        
        # HAPPO兼容性：强制数据形状标准化
        try:
            # Step 1: 标准化所有数据的形状
            dones_processed = self._standardize_dones(dones)
            obs_processed = self._standardize_observations(obs)
            share_obs_processed = self._standardize_observations(share_obs)
            rews_processed = self._standardize_rewards(rews)
            
            # Step 2: 验证形状一致性（HAPPO关键要求）
            self._validate_shapes(obs_processed, share_obs_processed, rews_processed, dones_processed)
            
            return (
                np.stack(obs_processed),
                np.stack(share_obs_processed),
                np.stack(rews_processed),
                np.stack(dones_processed),
                infos,
                list(available_actions),
            )
            
        except Exception as e:
            logger.error(f"HAPPO并行环境数据处理失败: {e}")
            # 降级到安全模式
            return self._get_safe_step_results(len(dones))
    
    def _standardize_dones(self, dones):
        """标准化done信号为HAPPO兼容格式"""
        processed_dones = []
        target_shape = None
        
        # 首先确定目标形状
        for done_env in dones:
            if isinstance(done_env, (list, tuple, np.ndarray)):
                target_shape = (len(done_env),) if target_shape is None else target_shape
                break
        
        if target_shape is None:
            # 如果都是标量，假设单智能体环境
            target_shape = (1,)
        
        # 标准化所有done信号
        for done_env in dones:
            if isinstance(done_env, (list, tuple)):
                done_array = np.array(done_env, dtype=bool)
            elif isinstance(done_env, np.ndarray):
                done_array = done_env.astype(bool)
            else:
                # 标量done值，扩展到目标形状
                done_array = np.full(target_shape, bool(done_env), dtype=bool)
            
            # 确保形状匹配
            if done_array.shape != target_shape:
                # 形状不匹配时的修正策略
                if done_array.size == 1 and target_shape[0] > 1:
                    # 标量扩展到多智能体
                    done_array = np.full(target_shape, done_array.item(), dtype=bool)
                elif len(done_array) > target_shape[0]:
                    # 截断到目标长度
                    done_array = done_array[:target_shape[0]]
                elif len(done_array) < target_shape[0]:
                    # 扩展到目标长度
                    padding = target_shape[0] - len(done_array)
                    done_array = np.concatenate([done_array, np.full(padding, done_array[-1], dtype=bool)])
            
            processed_dones.append(done_array)
        
        return processed_dones
    
    def _standardize_observations(self, observations):
        """标准化观测数据为HAPPO兼容格式"""
        if not observations:
            return []
        
        # 检查是否所有观测都有相同的结构
        reference_obs = observations[0]
        processed_obs = []
        
        for obs in observations:
            if isinstance(obs, list):
                # 多智能体观测列表
                if isinstance(reference_obs, list) and len(obs) != len(reference_obs):
                    logger.warning(f"观测长度不一致: {len(obs)} vs {len(reference_obs)}")
                    # 填充或截断到参考长度
                    target_len = len(reference_obs)
                    if len(obs) < target_len:
                        obs = obs + [obs[-1]] * (target_len - len(obs))
                    elif len(obs) > target_len:
                        obs = obs[:target_len]
                processed_obs.append(obs)
            else:
                processed_obs.append(obs)
        
        return processed_obs
    
    def _standardize_rewards(self, rewards):
        """标准化奖励数据为HAPPO兼容格式

        每个环境返回的奖励应为 (n_agents, 1) 形状的 numpy 数组。
        最终 stack 后形状为 (n_envs, n_agents, 1)。
        """
        processed_rewards = []

        for rew in rewards:
            rew_arr = np.asarray(rew)
            if rew_arr.ndim == 2:
                # 已经是 (n_agents, 1) 格式，直接使用
                processed_rewards.append(rew_arr)
            elif rew_arr.ndim == 1:
                # (n_agents,) 格式，扩展为 (n_agents, 1)
                processed_rewards.append(rew_arr[:, np.newaxis])
            elif rew_arr.ndim == 0:
                # 标量奖励，扩展为 (1, 1)
                processed_rewards.append(rew_arr.reshape(1, 1))
            else:
                # 高维数组，压缩多余维度
                processed_rewards.append(rew_arr.squeeze())
                if processed_rewards[-1].ndim < 2:
                    processed_rewards[-1] = processed_rewards[-1].reshape(-1, 1)

        return processed_rewards
    
    def _validate_shapes(self, obs, share_obs, rewards, dones):
        """验证所有数据形状的HAPPO兼容性"""
        n_envs = len(obs)
        
        # 验证环境数量一致性
        assert len(share_obs) == n_envs, f"share_obs环境数不匹配: {len(share_obs)} vs {n_envs}"
        assert len(rewards) == n_envs, f"rewards环境数不匹配: {len(rewards)} vs {n_envs}"
        assert len(dones) == n_envs, f"dones环境数不匹配: {len(dones)} vs {n_envs}"
        
        # 验证done信号形状一致性（HAPPO关键要求）
        done_shapes = [d.shape for d in dones]
        if len(set(done_shapes)) > 1:
            raise ValueError(f"Done信号形状不一致: {done_shapes}")
        
        # 验证观测形状一致性
        if isinstance(obs[0], list):
            obs_lens = [len(o) for o in obs]
            if len(set(obs_lens)) > 1:
                logger.warning(f"观测长度不一致: {obs_lens}")
    
    def _get_safe_step_results(self, n_envs):
        """在错误情况下返回安全的结果"""
        # 动态获取智能体数量，避免硬编码
        n_agents = getattr(self, 'n_agents', 1)

        # 动态获取观测维度
        if hasattr(self, 'observation_space') and len(self.observation_space) > 0:
            obs_space = self.observation_space[0]
            obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else 10
        else:
            obs_dim = 10  # 最小安全默认值

        safe_obs = [np.zeros((n_agents, obs_dim)) for _ in range(n_envs)]
        safe_share_obs = [np.zeros((n_agents, obs_dim)) for _ in range(n_envs)]
        safe_rewards = [[[0.0]] for _ in range(n_envs)]
        safe_dones = [np.array([False] * n_agents, dtype=bool) for _ in range(n_envs)]
        safe_infos = [{"error": True} for _ in range(n_envs)]
        safe_avail_actions = [None for _ in range(n_envs)]

        return (
            np.stack(safe_obs),
            np.stack(safe_share_obs),
            np.stack(safe_rewards),
            np.stack(safe_dones),
            safe_infos,
            safe_avail_actions
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        obs = np.stack(obs)
        share_obs = np.stack(share_obs)
        # available_actions 是不同长度的列表，不能直接 stack
        # 保持为列表格式
        available_actions = list(available_actions)
        return obs, share_obs, available_actions

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


# single env
class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        try:
            self.n_agents = env.n_agents
            self.ordered_agents_pairs=env.ordered_agents_pairs
            self.agents_bus=env.agents_bus
        except:
            pass

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        
        # Convert to lists first to handle potential shape mismatches during reset
        obs = list(obs)
        share_obs = list(share_obs)
        
        rews = np.array(rews)
        dones = np.array(dones)
        infos = np.array(infos)
        # available_actions 是不同长度的列表，保持为列表格式
        available_actions = list(available_actions)

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        
        # Convert back to numpy arrays after all resets are done
        obs = np.array(obs)
        share_obs = np.array(share_obs)
        
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        #print("reset_result=================================",results)#打印reset的结果
               
        obs, share_obs, available_actions = zip(*results)
        obs = np.array(obs)
        share_obs = np.array(share_obs)
        # available_actions是不同长度的列表，不能直接转换为numpy数组
        # 保持为列表格式
        available_actions = list(available_actions)

        return obs, share_obs, available_actions
        

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
