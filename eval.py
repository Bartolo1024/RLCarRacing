from typing import Any, Dict

import click
import gym

import agents.dqn_agent
import utils


@click.command()
@click.argument('config')
@click.argument('weights_path')
@utils.unpack_config
def main(
    weights_path: str,
    agent: Dict[str, Any],
    max_time_steps: int,
    **__
):
    env = gym.make('CarRacing-v0')
    agent = agents.dqn_agent.DQNAgent(**agent)
    agent.load_weights(weights_path)
    agent.eval()
    state = env.reset()
    total_reward = 0.
    for i in range(max_time_steps):
        action = agent.act(state)
        action_arr = agent.action_space[action]
        state, reward, done, _ = env.step(action_arr)
        env.render()
        total_reward += reward
        print(f'Reward: {reward}')
        if done:
            break
    print(f'Total reward: {total_reward}')


if __name__ == '__main__':
    main()
