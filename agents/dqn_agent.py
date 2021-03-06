import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pytorch_named_dims import nm
from torch import optim

from utils import import_function
from agents.utils import named_stack

from . import memory_agent


class DQNAgent(memory_agent.MemoryAgent):
    def __init__(
        self,
        action_space: Dict[str, Tuple[float, float, float]],
        network_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = .001,
        target_update: int = 8,
        optimizer: str = 'Adam',
        create_model: str = 'agents.models.cnn.FullyCNN',
        loss_fn: str = 'MSELoss',
        **kwargs
    ):
        super(DQNAgent, self).__init__(**kwargs)
        network_kwargs = network_kwargs if network_kwargs is not None else {}
        model_constructor = import_function(create_model)
        self.q_net = model_constructor(num_actions=len(action_space), **network_kwargs).to(self.device)
        self.target_net = model_constructor(num_actions=len(action_space), **network_kwargs).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        optimizer_constructor = getattr(optim, optimizer)
        self.optimizer = optimizer_constructor(self.q_net.parameters(), lr=lr)
        self.num_of_actions = len(action_space)
        self.target_update = target_update
        self.action_space = action_space
        self.loss = getattr(nm, loss_fn)()

    def act(self, state):
        sample = random.random()
        eps_th = self.epsilon()
        if sample > eps_th or self.eval_mode:
            x = self.state_transform(state)
            x = x.unflatten('C', (('N', 1), ('C', -1)))
            with torch.no_grad():
                val, idx = self.q_net(x)[0].max(0)
                return idx.item()
        return np.random.randint(self.num_of_actions)

    def update(self, epoch: int):
        tmp = self.sample_memory()
        if tmp is None:
            print('not update')
            return 0.
        state_batch, action_batch, reward_batch, batch = tmp
        self.q_net.train()
        q_estimate = self.q_net(state_batch).rename(None)

        action_batch = action_batch.unflatten('N', (('C', 1), ('N', -1))).rename(None)
        state_action_values = q_estimate.gather(1, action_batch).rename('C', 'N').flatten(('C', 'N'), 'N')
        with torch.no_grad():
            expected_state_action_values = self.future_reward_estimate(batch) + reward_batch

        loss = self.loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_net.parameters():
            if param.requires_grad:
                param.grad.data.clamp(-1, 1)
        self.optimizer.step()

        if epoch % self.target_update == 0 and epoch != 0:
            self.update_target_net()

        return loss.item()

    def future_reward_estimate(self, batch):
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=self.device
        )
        non_final_next_states = list(filter(lambda el: el is not None, batch.next_state))
        non_final_next_states = named_stack([self.state_transform(s) for s in non_final_next_states])
        next_state_values = torch.zeros(self.batch_size, device=self.device, requires_grad=False)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach().rename(None)
        return next_state_values * self.gamma

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_q_net(self, dir):
        torch.save(self.q_net.state_dict(), dir)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
