import torch
import torch.nn as nn
import gym
from utils.envs_tools import check, get_shape_from_obs_space
from models.base.mlp import MLPBase
from models.base.qmix_act import ACTLayer

class AgentQFunction(nn.Module):
    """
    Individual agent q network (MLP). For the implatation of Qmix
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, input_dim, act_dim, device):
        super(AgentQFunction, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        ######################
        #act_dim_space=gym.spaces.Discrete(act_dim)
        

        self.mlp = MLPBase(vars(args)["model"], [input_dim])
        #self.q = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, gain=self._gain, args=vars(args)["model"])
        self.q = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, gain=self._gain)
        self.to(device)

    def forward(self, x):
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
        x = check(x).to(**self.tpdv).to(self.device)
        x = self.mlp(x)
        # pass outputs through linear layer
        q_value = self.q(x)

        return q_value