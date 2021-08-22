from typing import Any, Dict, List, Union

import click
import gym
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from livelossplot import PlotLossesIgnite
from livelossplot.outputs import BaseOutput

import agents.dqn_agent
import utils
from agents.utils.net_saver import create_best_metric_saver

EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED


def create_reinforce_engine(agent: agents.dqn_agent.DQNAgent, environment: gym.Env, render: bool = False):
    def _run_single_time_step(engine, time_step):
        observation = engine.state.observation
        action = agent.act(observation)
        action_arr = agent.action_space[action]
        engine.state.observation, reward, done, _ = environment.step(action_arr)

        if action > len(agent.action_space) // 2:
            reward += .2
        engine.state.total_reward += reward
        engine.state.negative_ctr = 0 if reward > 0 else engine.state.negative_ctr + 1
        if engine.state.negative_ctr > 50:
            engine.terminate_epoch()

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
        engine.state.metrics = {}
        engine.state.total_reward = 0
        engine.state.negative_ctr = 0

    @trainer.on(EPISODE_COMPLETED)
    def _sum_reward(engine: Engine):
        engine.state.metrics = {'total_reward': engine.state.total_reward}
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
@utils.create_experiment
def main(
    agent: Dict[str, Any],
    max_epochs: int,
    max_time_steps: int,
    render: bool,
    run_dir: str,
    logger_outputs: List[Union[str, BaseOutput]]
):
    env = gym.make('CarRacing-v0')
    agent = agents.dqn_agent.DQNAgent(**agent)
    trainer = create_reinforce_engine(agent, env, render=render)
    trainer.attach(ProgressBar(persist=False))
    net_saver = create_best_metric_saver(
        model=agent.q_net, trainer=trainer, artifacts_dir=run_dir, metric_name='total_reward', mode=max
    )
    trainer.attach(net_saver)
    logger = PlotLossesIgnite(outputs=logger_outputs)
    trainer.attach(logger)

    trainer.run(data=range(max_time_steps), max_epochs=max_epochs)


if __name__ == '__main__':
    main()
