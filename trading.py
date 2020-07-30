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

        if not params.comp_target_net and params.comp_fixed == 0:
            for i in range(len(self.valuation_net)):
                self.valuation_net[i].load_weights(os.path.join(os.getcwd(), 'valuation', 'attitude-{}.pth'.format(i)))

        self.normal_actions = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
        self.env = env
        self.gamma = params.gamma
        self.mark_up = params.mark_up
        self.trading_budget = np.full(env.nb_agents, params.trading_budget)
        self.params = params

    def pay(self, offers, actions, observations, step_rewards, agents):
        """
        Executes trade by comparing offers and actions and exchange reward accordingly.
        :param offers: offers of the agents
        :param actions: actions the agents performed
        :param observations: observations before the actions were performed
        :param step_rewards: current step rewards
        :param agents: agents for target valuation net
        :return: step_rewards: rewards after the compensations have been exchanged, succ_trade: number of successful trades, accumulated_transfer: amount of reward transferred
        """

        succ_trades = np.zeros(len(agents))
        accumulated_transfer = np.zeros(len(agents))

        # check if offer and action matches
        for i in range(len(offers)):
            if offers[i][0] == actions[(i + 1) % 2][0] and offers[i][1] == actions[(i + 1) % 2][1]:
                if self.trading_budget[i] > 0:
                    if offers[i][0] != 0.0 or offers[i][1] != 0.0:

                        # calculate compensation
                        action_index = self.normal_actions.index(offers[i])

                        if self.params.comp_target_net:
                            valuation_q_vals = agents[(i + 1) % 2].compute_target_q_values(observations[(i + 1) % 2])[0]
                        elif self.params.comp_policy_net:
                            valuation_q_vals = agents[(i + 1) % 2].compute_q_values(observations[(i + 1) % 2])[0]
                        else:
                            if self.params.trading_observations:
                                val_obs = observations[(i + 1) % 2].copy()
                                val_obs = val_obs[:-2]
                                valuation_q_vals = self.valuation_net[self.env.attitude[(i + 1) % 2]].compute_q_values(val_obs)[0]
                            else:
                                valuation_q_vals = self.valuation_net[self.env.attitude[(i + 1) % 2]].compute_q_values(observations[(i + 1) % 2])[0]

                        if self.params.comp_fixed > 0:
                            compensation = self.params.comp_fixed
                        else:
                            compensation = ((np.max(valuation_q_vals) - valuation_q_vals[action_index]) / self.gamma) * self.mark_up

                        # clear differences
                        step_rewards[i] -= compensation
                        step_rewards[(i + 1) % 2] += compensation
                        self.trading_budget[i] -= compensation
                        succ_trades[i] += 1

                        accumulated_transfer[i] += compensation

        return step_rewards, succ_trades, accumulated_transfer

    def trading_observations(self, observations, offers):
        '''
        Add offers to the observations.
        :param observations: observations without the offers
        :param offers: offers of the agents
        :return: observations with included offer indicators
        '''
        trading_observations = []

        # copy observation
        for i in range(len(observations)):
            tr_observation = np.zeros((len(observations[i])+self.env.nb_agents, self.env.field_width, self.env.field_height))
            for i_obs in range(len(observations[i])):
                tr_observation[i_obs] = observations[i][i_obs]

            # add offer to the observations
            if offers[i][0] != 0.0 or offers[i][1] != 0.0:
                if self.env.collision_check(i, offers[((i + 1) % 2)]):
                    offer_pos = [self.env.agents_pos[i][0] + offers[((i + 1) % 2)][0], self.env.agents_pos[i][1] + offers[((i + 1) % 2)][1]]
                else:
                    offer_pos = [self.env.agents_pos[i][0], self.env.agents_pos[i][1]]
                tr_observation[len(observations[i])][int(offer_pos[0])][int(offer_pos[1])] = 1

            trading_observations.append(tr_observation)

        return trading_observations

    # def clear_after_episode(self, transfer, episode_return, agents):
    #     step_rewards = np.zeros(len(agents))
    #     for i in range(len(agents)):
    #         if episode_return[i] > 0 and transfer[i] > 0:
    #             if episode_return[i] - transfer[i] >= 0:
    #                 step_rewards[i] = transfer[i]
    #             else:
    #                 step_rewards[i] = episode_return[i]
    #     return step_rewards


def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    exp_time = "20200718-10-28-50"

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
    if params.trading_observations and params.trading:
        observation_shape = list(gym.spaces.Box(0.0, 1.0, shape=(len(env.observations[0])+env.nb_agents, env.field_width, env.field_height)).shape)
    for i in range(params.nb_agents):
        agent = make_dqn_agent(params, observation_shape, env.nb_actions)
        agent.load_weights(os.path.join(os.getcwd(), 'experiments', '{}'.format(exp_time), 'run 5', "weights-{}.pth".format(i)))
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
        offers = [[0, 0], [0, 0]]
        trade = Trade(env, params)
        if params.trading_observations:
            observations = trade.trading_observations(observations, offers)

    # setup video frames
    combined_frames = []
    combined_frames = drawing.render_esc_room(combined_frames, env.render_objects, 10,
                                              [[[0, 0], [0, 0]], [0, 0], [0, 0]], env)

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
            step_rewards, succ_trades, accumulated_transfer = trade.pay(offers, actions, observations, step_rewards, agents)

            for i in range(env.nb_agents):
                actions[i] = [actions[i][2], actions[i][3]]
            offers = actions

            if params.trading_observations:
                next_observations = trade.trading_observations(next_observations, offers)

        # episode ends on max steps or environment goal achievement
        if current_step == steps_per_episode or env.escape_room_done:
            done = True

        observations = next_observations
        current_step += 1

        for i in range(env.nb_agents):
            episode_return[i] += step_rewards[i]

        # set new frame for the current state of the environment
        if params.trading:
            combined_frames = drawing.render_esc_room(combined_frames, env.render_objects, 10, [offers, succ_trades, accumulated_transfer], env)

    # renders video from sequence of frames
    clip = mpy.ImageSequenceClip(combined_frames, fps=30)
    clip.write_videofile('escape room.mp4')


if __name__ == '__main__':
    main()