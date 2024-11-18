"""Tools."""
import torch
import numpy as np
from gym.spaces import Box, Discrete, Tuple


def _t2n(value):
    """Convert torch.Tensor to numpy.ndarray."""
    return value.detach().cpu().numpy()


def _flatten(T, N, value):
    """Flatten the first two dimensions of a tensor."""
    return value.reshape(T * N, *value.shape[2:])


def _sa_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, *dim) to (n_rollout_threads, episode_length, *dim).
    Then it combines the first two dimensions into one dimension.
    """
    return value.transpose(1, 0, 2).reshape(-1, *value.shape[2:])


def _ma_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, num_agents, *dim) to (n_rollout_threads, num_agents, episode_length, *dim).
    Then it combines the first three dimensions into one dimension.
    """
    return value.transpose(1, 2, 0, 3).reshape(-1, *value.shape[3:])

def _n2t(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input

def avail_choose(x, avail_x=None):
    x = _n2t(x)
    if avail_x is not None:
        avail_x = _n2t(avail_x)
        x[avail_x == 0] = -1e10
    return x#FixedCategorical(logits=x)

class DecayThenFlatSchedule():
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / \
                np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

def is_discrete(space):
    if isinstance(space, Discrete) or "MultiDiscrete" in space.__class__.__name__:
        return True
    else:
        return False


def is_multidiscrete(space):
    if "MultiDiscrete" in space.__class__.__name__:
        return True
    else:
        return False


def make_onehot(int_action, action_dim, seq_len=None):
    if type(int_action) == torch.Tensor:
        int_action = int_action.cpu().numpy()
    if not seq_len:
        return np.eye(action_dim)[int_action]
    if seq_len:
        onehot_actions = []
        for i in range(seq_len):
            onehot_action = np.eye(action_dim)[int_action[i]]
            onehot_actions.append(onehot_action)
        return np.stack(onehot_actions)

def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = space.shape[0]
    elif isinstance(space, Discrete):
        dim = space.n
    elif isinstance(space, Tuple):
        dim = sum([get_dim_from_space(sp) for sp in space])
    elif "MultiDiscrete" in space.__class__.__name__:
        return (space.high - space.low) + 1
    elif isinstance(space, list):
        dim = space[0]
    else:
        raise Exception("Unrecognized space: ", type(space))
    return dim

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2


def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)

def get_cent_act_dim(action_space):
    cent_act_dim = 0
    for space in action_space:
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            cent_act_dim += int(sum(dim))
        else:
            cent_act_dim += dim
    return cent_act_dim

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module