# -*- coding: utf-8 -*-
"""MAPPO 分组参数共享的 FP 优势切片与日志回填回归测试。

针对 4d8af3b5 复审缺陷 1（P1）：share_param="group" 时 runner 把全队
advantages 配上组内 buffer 传给 MAPPO.share_param_train，后者用组内序号切
agent 维导致串位；训练信息按组顺序 extend 拼接导致日志错配。

验收要点：
- FP 非连续组 [0,2]/[1]：每个成员的网络收到自己 agent 维的优势；
- EP：保持全队共享优势语义，不因分组被切片；
- 训练信息按实际 agent ID 回填（组 [0,2]/[1] 应为 [A,B,A] 而非 [A,A,B]）；
- 全队共享与不共享路径不受影响；
- 真实组件：FP 非连续组数据上每个独立网络的参数都发生更新。
"""

import numpy as np
import pytest
import torch
from gym.spaces import Box, Discrete

from runners.on_policy_ma_runner import OnPolicyMARunner


class _SpyActor:
    """记录 share_param_train / train 调用参数的探针网络。"""

    def __init__(self, name):
        self.name = name
        self.share_calls = []
        self.solo_calls = []

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        self.share_calls.append(
            {
                "buffers": list(actor_buffer),
                "advantages": advantages.copy(),
                "num_agents": num_agents,
                "state_type": state_type,
            }
        )
        return {"policy_loss": self.name}

    def train(self, actor_buffer, advantages, state_type):
        self.solo_calls.append(
            {
                "buffer": actor_buffer,
                "advantages": advantages.copy(),
                "state_type": state_type,
            }
        )
        return {"policy_loss": self.name}


class _SpyCritic:
    def train(self, critic_buffer, value_normalizer):
        return {"value_loss": 0.0}


def _make_fake_runner(state_type, share_param, num_agents=3, episode_length=6, threads=2):
    """绕过 __init__ 构造仅含 train() 依赖属性的 runner。

    share_param="group"/True 时 actor 列表 [A, B, A] 使 _shared_groups()
    产出非连续组 [0,2]/[1]；不共享时用三个独立 spy。
    critic 的 returns/value_preds 让各 agent 维优势取明显不同的 1/2/3。
    """
    runner = OnPolicyMARunner.__new__(OnPolicyMARunner)
    steps = episode_length + 1
    if state_type == "FP":
        returns = np.zeros((steps, threads, num_agents), dtype=np.float32)
        value_preds = np.zeros((steps, threads, num_agents), dtype=np.float32)
        value_preds[:] = -np.arange(1, num_agents + 1, dtype=np.float32)
        raw_advantages = returns[:-1] - value_preds[:-1]
    else:
        returns = np.ones((steps, threads), dtype=np.float32)
        value_preds = np.zeros((steps, threads), dtype=np.float32)
        raw_advantages = returns[:-1] - value_preds[:-1]

    runner.state_type = state_type
    runner.share_param = share_param
    runner.num_agents = num_agents
    runner.value_normalizer = None
    runner.critic_buffer = _Namespace(
        returns=returns, value_preds=value_preds, raw_advantages=raw_advantages
    )
    runner.actor_buffer = [
        _Namespace(active_masks=np.ones((steps, threads, 1), dtype=np.float32))
        for _ in range(num_agents)
    ]
    if share_param:
        net_a, net_b = _SpyActor("A"), _SpyActor("B")
        runner.actor = [net_a, net_b, net_a]
    else:
        runner.actor = [_SpyActor(name) for name in ("A", "B", "C")]
        net_a, net_b = runner.actor[0], runner.actor[1]
    runner.critic = _SpyCritic()
    return runner, net_a, net_b


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_group_fp_preserves_real_critic_value_axis():
    """真实 FP critic 的 [T,N,A,1] 布局须保留值轴并按成员切片。"""
    runner, net_a, net_b = _make_fake_runner("FP", "group")
    runner.critic_buffer.returns = runner.critic_buffer.returns[..., None]
    runner.critic_buffer.value_preds = runner.critic_buffer.value_preds[..., None]
    runner.actor_buffer[2].active_masks[0] = 0
    raw = runner.critic_buffer.returns[:-1] - runner.critic_buffer.value_preds[:-1]
    active = np.stack([b.active_masks[:-1] for b in runner.actor_buffer], axis=2)
    expected = (raw - raw[active != 0].mean()) / (raw[active != 0].std() + 1e-5)
    runner.train()
    np.testing.assert_allclose(net_a.share_calls[0]["advantages"], expected[:, :, [0, 2]])
    np.testing.assert_allclose(net_b.share_calls[0]["advantages"], expected[:, :, [1]])


def _expected_fp_normalized(raw_advantages, active_masks_list=None):
    """按 runner 的 FP 归一化公式计算期望（active 样本统计量）。"""
    advantages_copy = raw_advantages.copy()
    if active_masks_list is not None:
        masks = np.stack(active_masks_list, axis=2)[:-1]
        advantages_copy[masks.squeeze(axis=-1) == 0.0] = np.nan
        mean = np.nanmean(advantages_copy)
        std = np.nanstd(advantages_copy)
    else:
        mean = raw_advantages.mean()
        std = raw_advantages.std()
    return (raw_advantages - mean) / (std + 1e-5)


@pytest.mark.unit
class TestGroupFpAdvantages:
    def test_group_fp_slices_advantages_by_member_ids(self):
        """非连续组 [0,2]/[1]：网络收到的应是成员维切片而非全队张量。"""
        runner, net_a, net_b = _make_fake_runner("FP", "group")
        expected = _expected_fp_normalized(runner.critic_buffer.raw_advantages)

        runner.train()

        assert len(net_a.share_calls) == 1, "组 [0,2] 的网络应被调用一次"
        assert len(net_b.share_calls) == 1, "组 [1] 的网络应被调用一次"
        call_a = net_a.share_calls[0]
        call_b = net_b.share_calls[0]
        assert call_a["num_agents"] == 2
        assert call_b["num_agents"] == 1
        # 修复前这里收到的是全队 advantages，组内序号 1 切到 agent1 的优势
        np.testing.assert_allclose(
            call_a["advantages"], expected[:, :, [0, 2]], rtol=1e-6
        )
        np.testing.assert_allclose(call_b["advantages"], expected[:, :, [1]], rtol=1e-6)

    def test_group_fp_normalization_with_inactive_masks(self):
        """FP 归一化：真实形状 [T+1,N,1] 的 active_masks 可正常掩蔽。

        复现：stack 后的 4 维布尔掩码索引 3 维 advantages 直接 IndexError，
        FP 分组路径在到达串位逻辑前就会崩溃。
        """
        runner, net_a, net_b = _make_fake_runner("FP", "group")
        masks = [buf.active_masks for buf in runner.actor_buffer]
        # agent1 的前一半步设为 inactive，掩蔽样本不参与归一化统计
        masks[1][:3] = 0.0
        raw = runner.critic_buffer.raw_advantages
        expected = _expected_fp_normalized(raw, masks)

        runner.train()  # 修复前此处 IndexError

        np.testing.assert_allclose(
            net_a.share_calls[0]["advantages"], expected[:, :, [0, 2]], rtol=1e-6
        )
        np.testing.assert_allclose(
            net_b.share_calls[0]["advantages"], expected[:, :, [1]], rtol=1e-6
        )

    def test_group_ep_keeps_shared_advantages(self):
        """EP 分组训练仍传完整全队共享优势，不被成员切片。"""
        runner, net_a, net_b = _make_fake_runner("EP", "group")

        runner.train()

        raw = runner.critic_buffer.raw_advantages
        for call in [net_a.share_calls[0], net_b.share_calls[0]]:
            np.testing.assert_allclose(call["advantages"], raw, rtol=1e-6)


@pytest.mark.unit
class TestGroupActorTrainInfos:
    def test_group_infos_backfilled_by_agent_id(self):
        """组 [0,2]/[1] 的日志应为 [A,B,A]，而非按组顺序拼接的 [A,A,B]。

        用 EP 场景使该缺陷独立于 FP 归一化路径直接可见。
        """
        runner, net_a, net_b = _make_fake_runner("EP", "group")

        actor_train_infos, _ = runner.train()

        assert len(actor_train_infos) == 3
        assert actor_train_infos[0] is actor_train_infos[2]
        assert actor_train_infos[0]["policy_loss"] == "A"
        assert actor_train_infos[1]["policy_loss"] == "B"

    def test_shared_all_infos_repeat_for_every_agent(self):
        """全队共享：同一网络的信息重复到每个 agent 的日志位。"""
        runner, net_a, net_b = _make_fake_runner("FP", True)

        actor_train_infos, _ = runner.train()

        assert len(actor_train_infos) == 3
        assert all(info["policy_loss"] == "A" for info in actor_train_infos)

    def test_nonshared_fp_agent_gets_own_advantage(self):
        """不共享 FP：agent_i 单网训练收到 advantages[:,:,i]。"""
        runner, _, _ = _make_fake_runner("FP", False)
        expected = _expected_fp_normalized(runner.critic_buffer.raw_advantages)

        actor_train_infos, _ = runner.train()

        for agent_id, actor in enumerate(runner.actor):
            assert len(actor.solo_calls) == 1
            np.testing.assert_allclose(
                actor.solo_calls[0]["advantages"],
                expected[:, :, agent_id],
                rtol=1e-6,
            )
        assert [info["policy_loss"] for info in actor_train_infos] == ["A", "B", "C"]

    def test_nonshared_ep_agents_share_team_advantages(self):
        """不共享 EP：每个 agent 单网训练收到同一份全队优势。"""
        runner, _, _ = _make_fake_runner("EP", False)

        actor_train_infos, _ = runner.train()

        raw = runner.critic_buffer.raw_advantages
        for actor in runner.actor:
            assert len(actor.solo_calls) == 1
            np.testing.assert_allclose(
                actor.solo_calls[0]["advantages"], raw, rtol=1e-6
            )
        assert [info["policy_loss"] for info in actor_train_infos] == ["A", "B", "C"]


def _make_mappo_and_buffers():
    """构造真实 MAPPO 与组 [0,2] 成员（agent0/agent2）的 buffer。"""
    from algorithms.actors.mappo import MAPPO
    from common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
    from utils.config_pipeline import ConfigLoader

    cfg = ConfigLoader().load(experiment="configs/experiments/vvc_13bus_smoke.yaml")
    args = {**cfg.algo_args["train"], **cfg.algo_args["model"], **cfg.algo_args["algo"]}
    # 用小 episode 覆盖配置值，使 buffer 形状与本测试的优势张量一致
    args["episode_length"] = 6
    args["n_rollout_threads"] = 2
    obs_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    act_space = Discrete(5)

    rng = np.random.default_rng(0)
    buffers = []
    for _ in range(2):
        buffer = OnPolicyActorBuffer(args, obs_space, act_space)
        buffer.obs[1:] = rng.normal(size=buffer.obs[1:].shape).astype(np.float32)
        buffer.actions[:] = rng.integers(0, 5, size=buffer.actions.shape)
        buffer.action_log_probs[:] = -1.0
        buffers.append(buffer)

    nets = [MAPPO(args, obs_space, act_space, device=torch.device("cpu")) for _ in range(2)]
    return nets, buffers, args


def _clone_parameters(net):
    return [p.detach().clone() for p in net.actor.parameters()]


def _parameters_changed(before, after):
    return any(
        not torch.allclose(b, a) for b, a in zip(before, after)
    )


@pytest.mark.unit
class TestMappoComponent:
    def test_fp_group_generator_gets_member_advantages(self):
        """组内 buffer 的数据生成器收到各自成员的优势切片。

        组 [0,2] 传入切片后的 [T,N,2] 优势：成员 1（真实 agent2）的
        generator 收到第 2 列，即全队 advantages[:,:,2] 的值。
        """
        nets, buffers, _ = _make_mappo_and_buffers()
        steps, threads = 6, 2
        full = np.tile(
            np.arange(1.0, 4.0, dtype=np.float32), (steps, threads, 1)
        )
        group_advantages = full[:, :, [0, 2]].copy()

        captured = []
        for buffer in buffers:
            original = buffer.feed_forward_generator_actor

            def spy(advantages, *f_args, **f_kwargs):
                captured.append(advantages.copy())
                return original(advantages, *f_args, **f_kwargs)

            buffer.feed_forward_generator_actor = spy

        nets[0].share_param_train(buffers, group_advantages, 2, "FP")

        assert len(captured) == 2
        np.testing.assert_allclose(captured[0], group_advantages[:, :, 0], rtol=1e-6)
        np.testing.assert_allclose(captured[1], group_advantages[:, :, 1], rtol=1e-6)
        # 成员 1 的 generator 收到的是全队 agent2 维的值，而不是 agent1 的
        np.testing.assert_allclose(captured[1], full[:, :, 2], rtol=1e-6)

    def test_each_group_network_updates(self):
        """FP 非连续组的两个独立网络在各自组数据上都发生参数更新。"""
        nets, buffers, _ = _make_mappo_and_buffers()
        steps, threads = 6, 2
        full = np.tile(
            np.arange(1.0, 4.0, dtype=np.float32), (steps, threads, 1)
        )
        solo_buffer = buffers[1]

        before_a = _clone_parameters(nets[0])
        before_b = _clone_parameters(nets[1])
        nets[0].share_param_train(buffers, full[:, :, [0, 2]].copy(), 2, "FP")
        nets[1].share_param_train([solo_buffer], full[:, :, [1]].copy(), 1, "FP")

        assert _parameters_changed(before_a, _clone_parameters(nets[0])), "组 [0,2] 网络未更新"
        assert _parameters_changed(before_b, _clone_parameters(nets[1])), "组 [1] 网络未更新"
