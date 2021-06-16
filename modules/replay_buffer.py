import random
import numpy as np
from torch import nn
import torch 

class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.iters = 0
        self.batch_size=100

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, state, action, reward, next_state, done, mask):
        means = next_state.mean(1)
        for i in np.arange(state.shape[0]):
            if mask[i] > 0.0:
                if not np.isnan(means[i]):
                    self.push(state[i], action[i], reward[i], next_state[i], done[i])

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        #      state, action, reward, next_state, done
        return state, action, reward, next_state, done

    def __iter__(self):
        for batch_id in range(len(self.buffer)//self.batch_size):
            low, high = batch_id*self.batch_size,(batch_id+1)*self.batch_size
            yield  map(np.stack, zip(*self.buffer[low:high]))

    def __len__(self):
        return len(self.buffer)

    def buffer_reset(self):
        self.buffer = []
        self.position = 0


class ReplayBufferGPU(nn.Module):

    def __init__(self, capacity, use_cuda=True):
        super(ReplayBufferGPU, self).__init__()
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.to(self.device)
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.position = (self.position + 1) % self.capacity

    def push_batch(self, state, action, reward, next_state, done, mask):
        means = next_state.mean(1)
        for i in torch.arange(state.size(0)):
            if mask[i] > 0.0:
                if not torch.isnan(means[i]):
                    
                    self.push(state[i], action[i], reward[i], next_state[i], done[i])

    def sample(self, batch_size):      
        
        indices = torch.randperm(len(self.buffer))[:batch_size].to(dtype=torch.long,device=self.device)
        batch =[self.buffer[x] for x in indices]
        general_list =[torch.cat([x[i].unsqueeze(0) for x in batch ]) for i in torch.arange(5)]
        
        state, action, reward, next_state, done = general_list[0],general_list[1],general_list[2],general_list[3],general_list[4]
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def buffer_reset(self):
        self.buffer = []
        self.position = 0