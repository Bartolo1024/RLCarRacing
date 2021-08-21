from abc import abstractmethod

import torch

import agents.utils.replay_memory
from agents.utils import epsilon, named_stack
from agents.utils.transform import get_default_transform


class MemoryAgent:
    def __init__(
        self,
        batch_size=64,
        gamma: float = 0.99,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        num_of_actions: int = 3,
        replay_memory_capacity: int = 10000,
        device=torch.device('cuda')
    ):
        self.step = 0
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon.Epsilon(eps_start, eps_end, eps_decay)
        self.memory = agents.utils.replay_memory.ReplayMemory(replay_memory_capacity)
        self.num_of_actions = num_of_actions
        self.state_transform = get_default_transform(self.device)
        self.eval_mode = False

    @abstractmethod
    def act(self, state):
        pass

    def sample_memory(self):
        self.epsilon.update()
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        batch = agents.utils.replay_memory.Transition(*zip(*batch))
        state_batch = named_stack([self.state_transform(s) for s in batch.state])
        action_batch = named_stack(
            [
                torch.tensor(action, device=self.device, dtype=torch.long, requires_grad=False)
                for action in batch.action
            ]
        )
        reward_batch = named_stack(
            [
                torch.tensor(reward, device=self.device, dtype=torch.float32, requires_grad=False)
                for reward in batch.reward
            ]
        )
        return state_batch, action_batch, reward_batch, batch

    def push_transition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False
