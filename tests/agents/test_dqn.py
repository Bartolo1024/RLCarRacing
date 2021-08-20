import torch

from agents.dqn_agent import DQNAgent


def test_train_act():
    agent = DQNAgent(
        action_space={
            '0': (1., .2, 0.),
            '1': (0., 1., 0.),
            '2': (-1., .2, 0.)
        }, device=torch.device('cpu')
    )
    agent.train()
    for _ in range(10):
        state = torch.randn(1, 3, 96, 96, names=('N', 'C', 'H', 'W'))
        action_idx = agent.act(state)
        assert action_idx in range(3)


def test_eval_act():
    agent = DQNAgent(
        action_space={
            '0': (1., .2, 0.),
            '1': (0., 1., 0.),
            '2': (-1., .2, 0.)
        }, device=torch.device('cpu')
    )
    agent.eval()
    state = torch.randn(1, 3, 96, 96, names=('N', 'C', 'H', 'W'))
    first_action_idx = agent.act(state)
    for _ in range(10):
        action_idx = agent.act(state)
        assert action_idx == first_action_idx
