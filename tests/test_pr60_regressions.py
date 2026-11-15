"""PR60 采样契约与真实时序懒加载的回归；不启动 MATLAB。"""

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from envs.env_wrappers import ShareDummyVecEnv
from envs.lfc.lfc_env import LFCEnv
from envs.powergrid.smartds_timeseries import SmartDSEpisodeDataset
from runners.on_policy_base_runner import OnPolicyBaseRunner


class _LFCPhysicsStub:
    """仅替换联合仿真，保留真实 LFC 和向量包装器的采样契约。"""

    def reset(self):
        return np.zeros(3, dtype=np.float32)

    def step(self, actions):
        return np.ones(3, dtype=np.float32), 2.0, False, False, {}

    def close(self):
        pass


def _runner(agents, threads, discrete=False):
    """使用真实 actor buffer 构造只依赖采样状态的运行器。"""
    args = {
        "episode_length": 2, "n_rollout_threads": threads,
        "hidden_sizes": [4], "recurrent_n": 1,
    }
    runner = OnPolicyBaseRunner.__new__(OnPolicyBaseRunner)
    runner.num_agents = agents
    runner.recurrent_n, runner.rnn_hidden_size = 1, 4
    runner.state_type = "EP"
    runner.algo_args = {"train": args}
    spaces = [Discrete(3) if discrete and i == 0 else Box(-1, 1, (1,))
              for i in range(agents)]
    runner.actor_buffer = [
        OnPolicyActorBuffer(args, Box(-1, 1, (3,)), space) for space in spaces
    ]
    runner.critic_buffer = SimpleNamespace(
        share_obs=np.zeros((3, threads, 3)), insert=lambda *args: None,
    )
    return runner


def _data(obs, shared, rewards, dones, infos, available):
    """补齐运行器实际 insert 输入的策略/价值数组。"""
    threads, agents = obs.shape[:2]
    return (
        obs, shared, rewards, dones, infos, available,
        np.zeros((threads, 1)), np.zeros((threads, agents, 1)),
        np.zeros((threads, agents, 1)), np.zeros((threads, agents, 1, 4)),
        np.zeros((threads, 1, 4)),
    )


@pytest.mark.parametrize("threads", [1, 2])
def test_lfc_none_masks_reach_real_actor_buffers(threads):
    """LFC 的每环境 None 掩码须通过 warmup 和首步写入。"""
    def make_env():
        env = LFCEnv.__new__(LFCEnv)
        env.env, env.n_agents, env.discrete = _LFCPhysicsStub(), 10, False
        env.observation_space = [Box(-1, 1, (3,))] * 10
        env.share_observation_space = env.observation_space
        env.action_space = [Box(-1, 1, (1,))] * 10
        return env

    vector = ShareDummyVecEnv([make_env] * threads)
    runner = _runner(10, threads)
    runner.envs = vector
    try:
        runner.warmup()
        output = vector.step(np.zeros((threads, 10, 1)))
        assert output[-1] == [None] * threads
        runner.insert(_data(*output))
        for buffer in runner.actor_buffer:
            assert buffer.step == 1
            assert buffer.available_actions is None
            np.testing.assert_array_equal(buffer.obs[1], np.ones((threads, 3)))
    finally:
        vector.close()


@pytest.mark.parametrize("available", [None, [None, None], [[None, None]] * 2])
def test_continuous_sampling_accepts_all_no_mask_forms(available):
    """整体、逐环境和逐智能体的无掩码表示均不生成 object 数组。"""
    runner = _runner(2, 2)
    obs = np.ones((2, 2, 3))
    runner.insert(_data(obs, obs, np.ones((2, 2, 1)),
                        np.zeros((2, 2), dtype=bool), [[{}, {}]] * 2, available))
    assert all(buffer.step == 1 for buffer in runner.actor_buffer)


def test_mixed_sampling_preserves_discrete_mask_and_continuous_none():
    """混合动作的离散 mask 数值应完整写入，连续槽仍为 None。"""
    runner = _runner(2, 2, discrete=True)
    obs = np.ones((2, 2, 3))
    masks = [[np.array([1, 0, 1]), None], [np.array([0, 1, 1]), None]]
    runner.insert(_data(obs, obs, np.ones((2, 2, 1)),
                        np.zeros((2, 2), dtype=bool), [[{}, {}]] * 2, masks))
    np.testing.assert_array_equal(runner.actor_buffer[0].available_actions[1],
                                  [[1, 0, 1], [0, 1, 1]])
    assert runner.actor_buffer[1].available_actions is None


def test_episode_metadata_is_lazy_and_selected_days_are_cached():
    """300 天配置不应在查长度和选一天时遍历全年逐设备数据。"""
    calls = []

    def row(index):
        calls.append(index)
        return {"loads": {"load": (float(index + 1), 0.2)}, "pv": {}}

    dataset = SmartDSEpisodeDataset(
        "/tmp", "master.dss", {"train": list(range(300)), "validation": [364]},
    )
    dataset._series = SimpleNamespace(row=row)
    episodes = dataset.episodes
    assert len(episodes) == 300
    assert calls == []
    entry, profile = episodes[12]
    assert entry["day"] == 12
    assert calls == list(range(12 * 96, 13 * 96))
    assert episodes[12][1] is profile
    assert len(calls) == 96
    assert len(episodes[:1]) == 1
    assert len(calls) == 192
    with pytest.raises(IndexError):
        _ = episodes[300]


@pytest.mark.parametrize("days", [{"train": None}, {"train": "0"}, {"test": [1]}])
def test_invalid_holdout_containers_fail_before_workers(days):
    """配置容器错误同样应给出清晰 ValueError，而非迭代时 TypeError。"""
    with pytest.raises(ValueError):
        SmartDSEpisodeDataset("/tmp", "master.dss", days)


def test_partial_worker_masks_are_rejected_instead_of_silently_ignored():
    """同一离散 agent 的 mask 部分缺失不能被当作全部无约束。"""
    with pytest.raises(ValueError, match="部分缺失"):
        OnPolicyBaseRunner._agent_available_actions([None, [np.ones(3)]], 0)


def test_split_digests_share_content_scan_but_preserve_distinct_identities():
    """合并两种设计时保留 GLM 的分片身份，不为每个 split 重扫曲线。"""
    scans = []

    def content_digest():
        scans.append(True)
        return "frozen-content"

    dataset = SmartDSEpisodeDataset(
        "/tmp", "master.dss", {"train": [0, 1], "validation": [2]},
    )
    dataset._series = SimpleNamespace(content_digest=content_digest)
    training = dataset.sha256
    validation = dataset.sha256_for_split("validation")
    assert training != validation
    assert dataset.sha256_for_split("train") == training
    assert dataset.sha256_for_split("validation") == validation
    assert len(scans) == 1
