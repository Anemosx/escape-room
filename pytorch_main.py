import os
import gym
import gym.spaces

from pytorch_agents import make_dqn_agent
import pytorch_training
from dotmap import DotMap
import json
import neptune
from datetime import datetime
import escape_room


def main():
    """
    Trains agents with logger (neptune)
    """

    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    # make new directory depending on current date time
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    log_dir = os.path.join(os.getcwd(), 'experiments', '{}'.format(exp_time))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save params
    with open(os.path.join(log_dir, 'params.json'), 'w') as outfile:
        json.dump(params_json, outfile)
    with open(os.path.join(log_dir, 'params.txt'), 'w') as outfile:
        json.dump(params_json, outfile)

    # logging in neptune to view progress and run experiment
    if params.logging:
        neptune.init('arno/escape-room',
                     api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMzA3NmZlNmEtZWFhZC00MDY1LTk4MjEtNDk3MzBlODQ2Mzc3In0=')
        logger = neptune
        with neptune.create_experiment(name='esc_room_trading', params=params_json):
            neptune.append_tag('time-{}'.format(exp_time))
            run_experiment(params, logger, log_dir)
    else:
        logger = None
        run_experiment(params, logger, log_dir)


def run_experiment(params, logger, log_dir):
    """
    Runs experiment and save data.
    :param params: parameters of the run
    :param logger: logger object (neptune)
    :param log_dir: directory path of the current experiment
    """

    # use custom actions on trading
    if params.trading:
        params.use_actions = 1
    else:
        params.use_actions = 0

    # init escape room environment
    env = escape_room.init_escape_room(params)

    # init agents
    agents = []
    observation_shape = list(gym.spaces.Box(0.0, 1.0, shape=(len(env.observations[0]), env.field_width, env.field_height)).shape)
    for i in range(params.nb_agents):
        agent = make_dqn_agent(params, observation_shape, env.nb_actions)
        agents.append(agent)

    # train agents
    pytorch_training.train_trading_dqn(params, agents, env, params.trading, params.train_episodes, params.nb_max_episode_steps, logger)

    # save agents weights
    for i in range(len(agents)):
        agents[i].save_weights(os.path.join(log_dir, 'weights-{}.pth'.format(i)))


if __name__ == '__main__':
    main()