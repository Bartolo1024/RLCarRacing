from typing import Any, Dict

import click
import gym
from gym.envs.box2d import CarRacing
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events

import agents.dqn_agent
import utils

EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED


def create_reinforce_engine(agent: agents.dqn_agent.DQNAgent, environment: gym.Env, render: bool = False):
    def _run_single_time_step(engine, time_step):
        observation = engine.state.observation
        action = agent.act(observation)
        action_arr = agent.action_space[action]
        engine.state.observation, reward, done, _ = environment.step(action_arr)
        engine.state.total_reward += reward

        if render:
            environment.render()
        agent.push_transition(observation, action, engine.state.observation, reward)

        if done:
            engine.terminate_epoch()
            engine.state.time_step = time_step

    trainer = Engine(_run_single_time_step)

    @trainer.on(EPISODE_STARTED)
    def reset_environment_state(engine):
        agent.train()
        engine.state.observation = environment.reset()
        engine.state.total_reward = 0

    @trainer.on(EPISODE_COMPLETED)
    def _sum_reward(engine: Engine):
        print(f'Reward: {engine.state.total_reward}')
        agent.update(engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def close(_: Engine):
        environment.close()

    def _attach(plugin):
        plugin.attach(trainer)

    trainer.attach = _attach

    return trainer


@click.command()
@click.argument('config')
@utils.unpack_config
def main(agent: Dict[str, Any], max_epochs: int, max_time_steps: int):
    env = CarRacing()
    agent = agents.dqn_agent.DQNAgent(**agent)
    trainer = create_reinforce_engine(agent, env)
    trainer.attach(ProgressBar(persist=False))
    trainer.run(data=range(max_time_steps), max_epochs=max_epochs)


if __name__ == '__main__':
    main()
