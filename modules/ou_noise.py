import numpy as np
import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class OUNoise:

    # https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=None, decay_period=10000, low=-1,
                 high=1):
        if min_sigma is None:
            min_sigma = max_sigma
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.low = low
        self.high = high
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim).to(device) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + (self.sigma * torch.rand(self.action_dim).to(device))
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = torch.tensor(self.evolve_state()).to(device)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.clamp(action + ou_state, self.low, self.high)

    def get_action_batch(self, actions, times):
        noisy_out = []
        for action, t in zip(actions, times):
            noisy_out.append(self.get_action(action, t))

        return torch.stack(noisy_out, 0).to(device)
