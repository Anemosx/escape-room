import os
import moviepy.editor as mpy
import numpy as np
import json
from dotmap import DotMap
import gym
import gym.spaces

from pytorch_agents import make_dqn_agent
import drawing
import escape_room


class Trade:
    def __init__(self, env, params):
        """
        Enables agents to trade their reward for the other agents action.
        :param env: escape room environment
        :param params: parameters of the run
        """

        # setup valuation net for action trading compensation
        observation_shape = list(gym.spaces.Box(0.0, 1.0, shape=(len(env.observations[0]), env.field_width, env.field_height)).shape)

        self.valuation_net = [make_dqn_agent(params, observation_shape, 4),
                              make_dqn_agent(params, observation_shape, 4)]

        for i in range(len(self.valuation_net)):
            self.valuation_net[i].load_weights(os.path.join(os.getcwd(), 'valuation', 'attitude-{}.pth'.format(i)))

        self.normal_actions = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
        self.env = env
        self.gamma = params.gamma
        self.mark_up = params.mark_up
        self.trading_budget = np.full(env.nb_agents, params.trading_budget)

    def pay(self, offers, actions, observations, step_rewards):
        """
        Executes trade by comparing offers and actions and exchange reward accordingly.
        :param offers: offers of the agents
        :param actions: actions the agents performed
        :param observations: observations before the actions were performed
        :param step_rewards: current step rewards
        :return: step_rewards: rewards after the compensations have been exchanged, succ_trade: number of successful trades
        """

        succ_trade = 0

        # check if offer and action matches
        for i in range(len(offers)):
            if offers[i][0] == actions[(i + 1) % 2][0] and offers[i][1] == actions[(i + 1) % 2][1]:
                if self.trading_budget[i] > 0:
                    if offers[i][0] != 0.0 or offers[i][1] != 0.0:

                        # calculate comparison from the valuation_net
                        action_index = self.normal_actions.index(offers[i])
                        valuation_q_vals = self.valuation_net[(i + 1) % 2].compute_q_values(observations[self.env.attitude[(i + 1) % 2]])[0]
                        compensation = ((np.max(valuation_q_vals) - valuation_q_vals[action_index]) / self.gamma) * self.mark_up

                        # clear differences
                        step_rewards[i] -= compensation
                        step_rewards[(i + 1) % 2] += compensation
                        self.trading_budget[i] -= 1
                        succ_trade += 1

        return step_rewards, succ_trade


def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    exp_time = "20200618-20-35-41"

    # use custom actions on trading
    if params.trading:
        params.use_actions = 1
    else:
        params.use_actions = 0

    # init escape room environment
    env = escape_room.init_escape_room(params)

    # init agents and load weights
    agents = []
    observation_shape = list(gym.spaces.Box(0.0, 1.0, shape=(len(env.observations[0]), env.field_width, env.field_height)).shape)
    for i in range(params.nb_agents):
        agent = make_dqn_agent(params, observation_shape, env.nb_actions)
        agent.load_weights(os.path.join(os.getcwd(), 'experiments', '{}'.format(exp_time), "weights-{}.pth".format(i)))
        agent.epsilon = 0.01
        agents.append(agent)

    # enables escape room render and resets variables to start new episode
    env.render = True
    env.reset()
    observations = env.observations
    current_step = 0
    agent_indices = list(range(0, env.nb_agents))
    episode_return = np.zeros(env.nb_agents)
    steps_per_episode = 100
    done = False

    if params.trading:
        offers = []
        trade = Trade(env, params)

    # setup video frames
    combined_frames = []
    combined_frames = drawing.render_esc_room(combined_frames, env.render_objects, 10)

    # run one episode
    while not done:

        # agents choose action depending on policy
        actions = []
        action_indices = []
        for agent_index in agent_indices:
            action_index = agents[agent_index].policy(observations[agent_index])
            actions.append(env.actions[action_index])
            action_indices.append(action_index)

        # perform step in the environment
        next_observations, step_rewards = env.step(actions)

        # execute trades and get new offers of agents
        if params.trading:
            step_rewards, succ_trade = trade.pay(offers, actions, observations, step_rewards)

            for i in range(env.nb_agents):
                actions[i] = [actions[i][2], actions[i][3]]
            offers = actions

            # agent without attitude can not make any offer
            for i in range(len(actions)):
                if not env.attitude[i]:
                    offers[i] = [0.0, 0.0]

        # episode ends on max steps or environment goal achievement
        if current_step == steps_per_episode or env.escape_room_done:
            done = True

        observations = next_observations
        current_step += 1

        for i in range(env.nb_agents):
            episode_return[i] += step_rewards[i]

        # set new frame for the current state of the environment
        combined_frames = drawing.render_esc_room(combined_frames, env.render_objects, 10)

    # renders video from sequence of frames
    clip = mpy.ImageSequenceClip(combined_frames, fps=30)
    clip.write_videofile('escape room.mp4')


if __name__ == '__main__':
    main()