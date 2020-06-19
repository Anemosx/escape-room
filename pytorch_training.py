from datetime import datetime
import numpy as np

from trading import Trade


def train_trading_dqn(params, agents, env, trading, training_episodes: int, steps_per_episode: int, logger):
    """
    Trains agents in the environment
    :param params: parameters of the run
    :param agents: agents in the environment
    :param env: escape room environment
    :param trading: 0 if trading enabled, 1 if disabled
    :param training_episodes: amount of training episodes
    :param steps_per_episode: max amount of steps in one episode
    :param logger: logger object (neptune)
    """

    print("{} | training started".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # creates trading object if trading enabled
    if trading:
        trade = Trade(env, params)

    # run episodes
    for episode in range(0, training_episodes):
        # resets variables to start new episode
        env.reset()
        observations = env.observations
        current_step = 0
        agent_indices = list(range(0, env.nb_agents))
        episode_return = np.zeros(env.nb_agents)
        done = False

        if trading:
            offers = []
            amount_trades = 0
            trade.trading_budget = np.full(env.nb_agents, params.trading_budget)

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
            if trading:
                step_rewards, succ_trade = trade.pay(offers, actions, observations, step_rewards)
                amount_trades += succ_trade

                for i in range(env.nb_agents):
                    actions[i] = [actions[i][2], actions[i][3]]
                offers = actions

                # agent without attitude can not make any offer
                for i in range(len(actions)):
                    if not env.attitude[i]:
                        offers[i] = [0.0, 0.0]

            # save transitions of agents
            for agent_index in agent_indices:
                agents[agent_index].save(observations[agent_index],
                                         action_indices[agent_index],
                                         next_observations[agent_index],
                                         step_rewards[agent_index],
                                         done)

            # train agents
            for agent_index in agent_indices:
                if not done:
                    agents[agent_index].train()

            # episode ends on max steps or environment goal achievement
            if current_step == steps_per_episode or env.escape_room_done:
                done = True

            observations = next_observations.copy()
            current_step += 1

            for i in range(env.nb_agents):
                episode_return[i] += step_rewards[i]

        # logs episode stats on logger object (neptune)
        if logger is not None:
            logger.log_metric('episode_return', np.sum(episode_return))
            logger.log_metric('episode_steps', current_step)
            logger.log_metric('episode_return-0', episode_return[0])
            logger.log_metric('episode_return-1', episode_return[1])
            logger.log_metric('episode_return_per_step', np.sum(episode_return) / current_step)
            logger.log_metric('episode_return_per_step-0', episode_return[0] / current_step)
            logger.log_metric('episode_return_per_step-1', episode_return[1] / current_step)
            if trading:
                logger.log_metric('trades', amount_trades)

        if episode > 0 and episode % 25 is 0:
            for i in range(len(agents)):
                print("episode: {}, epsilon: {:.5f}, rewards: {:.3f}".format(episode, agents[i].epsilon, episode_return[i]))

    print("{} | training finished".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))